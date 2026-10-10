// InfiniCore's modular stack: public InfiniOps API, PyTorch-owned memory/stream.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <infini/ops.h>

#include <optional>
#include <utility>
#include <vector>

namespace {
namespace ops = infini::ops;

ops::DataType dtype_from_torch(const at::Tensor &tensor) {
    switch (tensor.scalar_type()) {
    case at::kFloat:
        return ops::DataType::kFloat32;
    case at::kHalf:
        return ops::DataType::kFloat16;
    case at::kBFloat16:
        return ops::DataType::kBFloat16;
    case at::kInt:
        return ops::DataType::kInt32;
    case at::kLong:
        return ops::DataType::kInt64;
    default:
        TORCH_CHECK(false, "unsupported dtype for InfiniOps bridge");
    }
}

ops::Device device_from_torch(const at::Tensor &tensor) {
    TORCH_CHECK(tensor.is_cuda(), "InfiniOps bridge requires accelerator tensors");
#if defined(ENABLE_METAX_API)
    return {ops::Device::Type::kMetax, tensor.get_device()};
#elif defined(ENABLE_NVIDIA_API)
    return {ops::Device::Type::kNvidia, tensor.get_device()};
#else
#error "The modular InfiniOps bridge supports MetaX and NVIDIA"
#endif
}

ops::Tensor wrap(const at::Tensor &tensor) {
    ops::Tensor::Shape shape(tensor.sizes().begin(), tensor.sizes().end());
    ops::Tensor::Strides strides(tensor.strides().begin(), tensor.strides().end());
    return {tensor.data_ptr(), std::move(shape), dtype_from_torch(tensor),
            device_from_torch(tensor), std::move(strides)};
}

// Explicitly select the native implementation. Online tuning synchronizes the
// device, so it must not run inside vLLM's graph capture.
ops::Config native_config() {
    ops::Config config;
    config.set_implementation_index(0);
    return config;
}

template <typename Op, typename... Args>
void launch(const at::Tensor &reference, const Args &...args) {
    const c10::cuda::CUDAGuard guard(reference.device());
    ops::Handle handle;
    handle.set_stream(at::cuda::getCurrentCUDAStream(reference.get_device()).stream());
    Op::Call(handle, native_config(), args...);
}

void check_same_device(const at::Tensor &reference, const at::Tensor &tensor) {
    TORCH_CHECK(reference.device() == tensor.device(),
                "InfiniOps inputs must be on the same device");
}

std::optional<ops::Tensor> optional_tensor(const std::optional<at::Tensor> &tensor,
                                           const at::Tensor &reference) {
    if (!tensor || !tensor->defined())
        return std::nullopt;
    check_same_device(reference, *tensor);
    return wrap(*tensor);
}
} // namespace

at::Tensor linear_current_stream(at::Tensor input, at::Tensor weight,
                                 std::optional<at::Tensor> bias) {
    TORCH_CHECK(input.dim() >= 1 && weight.dim() == 2,
                "expected input [..., in_features] and weight [out_features, in_features]");
    check_same_device(input, weight);
    TORCH_CHECK(input.scalar_type() == weight.scalar_type() && input.size(-1) == weight.size(1),
                "linear input/weight dtype or feature mismatch");
    if (bias && bias->defined()) {
        check_same_device(input, *bias);
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == weight.size(0),
                    "linear bias must have shape [out_features]");
    }
    auto shape = input.sizes().vec();
    shape.back() = weight.size(0);
    auto output = at::empty(shape, input.options());
    if (output.numel() == 0)
        return output;
    auto input_2d = input.reshape({-1, input.size(-1)});
    auto output_2d = output.view({-1, weight.size(0)});
    auto weight_t = weight.transpose(0, 1);
    launch<ops::Gemm>(input, wrap(input_2d), wrap(weight_t), wrap(output_2d));
    if (bias && bias->defined())
        output.add_(*bias);
    return output;
}

at::Tensor embedding_current_stream(at::Tensor input, at::Tensor weight) {
    check_same_device(weight, input);
    TORCH_CHECK(weight.dim() == 2 &&
                    (input.scalar_type() == at::kInt || input.scalar_type() == at::kLong),
                "embedding requires integer indices and 2D weights");
    auto shape = input.sizes().vec();
    shape.push_back(weight.size(1));
    auto output = at::empty(shape, weight.options());
    if (output.numel())
        launch<ops::Embedding>(weight, wrap(input), wrap(weight), wrap(output));
    return output;
}

at::Tensor rms_norm_current_stream(at::Tensor input, at::Tensor weight, double epsilon) {
    check_same_device(input, weight);
    TORCH_CHECK(input.dim() >= 1 && weight.dim() == 1 && weight.size(0) == input.size(-1),
                "RMSNorm requires weight matching the last input dimension");
    auto output = at::empty_like(input);
    if (output.numel()) {
        // InfiniOps accepts [batch, heads, dim] with explicit strides. Q/K
        // are views of packed QKV; flattening their first two dimensions
        // would allocate and copy on every call when batch > 1.
        auto x = input.dim() == 3 ? input : input.reshape({-1, input.size(-1)});
        auto y = input.dim() == 3 ? output : output.view({-1, input.size(-1)});
        launch<ops::RmsNorm>(input, wrap(x), wrap(weight), static_cast<float>(epsilon), wrap(y));
    }
    return output;
}

bool add_rms_norm_supported(at::Tensor input, at::Tensor residual, at::Tensor weight,
                            double /*epsilon*/) {
    return input.dim() >= 2 && input.sizes() == residual.sizes() &&
           input.device() == residual.device() && input.device() == weight.device() &&
           input.scalar_type() == residual.scalar_type() &&
           input.scalar_type() == weight.scalar_type() && weight.dim() == 1 &&
           weight.size(0) == input.size(-1) &&
           !ops::FusedAddRmsNorm::active_implementation_indices(device_from_torch(input).type())
                .empty();
}

void add_rms_norm_inplace_current_stream(at::Tensor input, at::Tensor residual, at::Tensor weight,
                                         double epsilon) {
    TORCH_CHECK(add_rms_norm_supported(input, residual, weight, epsilon),
                "unsupported fused add+RMSNorm input");
    if (input.numel()) {
        auto x = input.view({-1, input.size(-1)});
        auto r = residual.view_as(x);
        const std::optional<ops::Tensor> w{wrap(weight)};
        launch<ops::FusedAddRmsNorm>(input, wrap(x), wrap(r), w, static_cast<float>(epsilon));
    }
}

std::vector<at::Tensor> add_rms_norm_current_stream(at::Tensor input, at::Tensor residual,
                                                    at::Tensor weight, double epsilon) {
    // vllm_infinicore custom ops return fresh outputs. InfiniOps' canonical
    // FusedAddRmsNorm mutates its buffers, so clone before invoking it.
    auto output = input.clone(at::MemoryFormat::Contiguous);
    auto residual_out = residual.clone(at::MemoryFormat::Contiguous);
    add_rms_norm_inplace_current_stream(output, residual_out, weight, epsilon);
    return {output, residual_out};
}

at::Tensor silu_and_mul_current_stream(at::Tensor input) {
    TORCH_CHECK(input.dim() >= 1 && input.size(-1) % 2 == 0,
                "SiLU input last dimension must be even");
    auto shape = input.sizes().vec();
    shape.back() /= 2;
    auto output = at::empty(shape, input.options());
    if (output.numel())
        launch<ops::SiluAndMul>(input, wrap(input), wrap(output));
    return output;
}

void rotary_embedding_inplace_current_stream(at::Tensor positions, at::Tensor query,
                                             std::optional<at::Tensor> key, int64_t head_size,
                                             at::Tensor cache, bool is_neox_style) {
    check_same_device(query, positions);
    check_same_device(query, cache);
    TORCH_CHECK(positions.scalar_type() == at::kLong && positions.dim() == 1,
                "RotaryEmbedding requires 1D int64 positions");
    TORCH_CHECK(head_size > 0 && cache.dim() == 2 && cache.size(1) <= head_size &&
                    cache.size(1) % 2 == 0 && cache.stride(1) == 1,
                "invalid rotary cache/head dimension");
    if (key && key->defined()) {
        check_same_device(query, *key);
    }
    if (positions.numel()) {
        auto q_view = query.view({positions.numel(), -1, head_size});
        std::optional<at::Tensor> k_view;
        if (key && key->defined())
            k_view = key->view({positions.numel(), -1, head_size});
        launch<ops::RotaryEmbedding>(query, wrap(positions), wrap(q_view),
                                     optional_tensor(k_view, query), wrap(cache), head_size,
                                     is_neox_style, int64_t{0}, false);
    }
}

std::vector<at::Tensor> rotary_embedding_current_stream(at::Tensor positions, at::Tensor query,
                                                        std::optional<at::Tensor> key,
                                                        int64_t head_size, at::Tensor cache,
                                                        bool is_neox_style) {
    auto q_out = query.clone(at::MemoryFormat::Contiguous);
    std::optional<at::Tensor> k_out;
    if (key && key->defined())
        k_out = key->clone(at::MemoryFormat::Contiguous);
    rotary_embedding_inplace_current_stream(positions, q_out, k_out, head_size, cache,
                                            is_neox_style);
    return {q_out, k_out.value_or(at::Tensor{})};
}

void store_kv_cache_current_stream(at::Tensor key_cache, at::Tensor value_cache, at::Tensor key,
                                   at::Tensor value, at::Tensor slots) {
    for (const auto &tensor : {key_cache, value_cache, value, slots})
        check_same_device(key, tensor);
    if (slots.numel())
        launch<ops::PagedCachingInfinilm>(key, wrap(key), wrap(value), wrap(slots.flatten()),
                                          wrap(key_cache), wrap(value_cache));
}

void paged_attention_prefill_current_stream(at::Tensor query, at::Tensor key_cache,
                                            at::Tensor value_cache, at::Tensor blocks,
                                            at::Tensor lengths, at::Tensor starts,
                                            std::optional<at::Tensor> alibi, double scale,
                                            at::Tensor output) {
    for (const auto &tensor : {key_cache, value_cache, blocks, lengths, starts, output})
        check_same_device(query, tensor);
    TORCH_CHECK(query.dim() == 3 && query.sizes() == output.sizes(),
                "attention requires matching [tokens, heads, head_dim] query/output");
    if (query.numel())
        launch<ops::PagedAttentionPrefillInfinilm>(
            query, wrap(query), wrap(key_cache), wrap(value_cache), wrap(blocks), wrap(lengths),
            wrap(starts), optional_tensor(alibi, query), static_cast<float>(scale), wrap(output));
}

void paged_attention_decode_out(at::Tensor query, at::Tensor key_cache, at::Tensor value_cache,
                                at::Tensor lengths, at::Tensor blocks,
                                std::optional<at::Tensor> alibi, double scale, int64_t num_tokens,
                                int64_t num_decodes, at::Tensor output) {
    TORCH_CHECK(num_tokens == num_decodes, "InfiniOps bridge does not support speculative decode");
    for (const auto &tensor : {key_cache, value_cache, blocks, lengths, output})
        check_same_device(query, tensor);
    TORCH_CHECK(query.dim() == 3 && num_tokens >= 0 && num_tokens <= query.size(0),
                "invalid decode token count/query shape");
    if (!num_tokens)
        return;
    auto q = query.narrow(0, 0, num_tokens);
    auto out = output.narrow(0, 0, num_tokens).view(q.sizes());
    // This native implementation advertises a workspace for split-KV decode.
    // PyTorch owns it so allocations remain valid through CUDA Graph replay.
    auto q_view = wrap(q);
    auto k = wrap(key_cache);
    auto v = wrap(value_cache);
    auto b = wrap(blocks);
    auto l = wrap(lengths);
    auto a = optional_tensor(alibi, query);
    auto y = wrap(out);
    const auto s = static_cast<float>(scale);
    const c10::cuda::CUDAGuard guard(query.device());
    const auto config = native_config();
    auto op = ops::PagedAttentionInfinilm::Make(config, q_view, std::as_const(k), std::as_const(v),
                                                std::as_const(b), std::as_const(l),
                                                std::as_const(a), s, std::as_const(y));
    const auto bytes = op->workspace_size_in_bytes();
    ops::Handle handle;
    handle.set_stream(at::cuda::getCurrentCUDAStream(query.get_device()).stream());
    at::Tensor workspace;
    if (bytes) {
        workspace = at::empty({static_cast<int64_t>(bytes)}, query.options().dtype(at::kByte));
        handle.set_workspace(workspace.data_ptr());
        handle.set_workspace_size_in_bytes(bytes);
    }
    ops::PagedAttentionInfinilm::Call(handle, config, q_view, k, v, b, l, a, s, y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_current_stream", &linear_current_stream);
    m.def("lm_head", &linear_current_stream);
    m.def("embedding_current_stream", &embedding_current_stream);
    m.def("rms_norm_current_stream", &rms_norm_current_stream);
    m.def("add_rms_norm_current_stream", &add_rms_norm_current_stream);
    m.def("add_rms_norm_inplace_current_stream", &add_rms_norm_inplace_current_stream);
    m.def("add_rms_norm_supported", &add_rms_norm_supported);
    m.def("silu_and_mul_current_stream", &silu_and_mul_current_stream);
    m.def("rotary_embedding_current_stream", &rotary_embedding_current_stream);
    m.def("rotary_embedding_inplace_current_stream", &rotary_embedding_inplace_current_stream);
    m.def("store_kv_cache_current_stream", &store_kv_cache_current_stream);
    m.def("paged_attention_prefill_current_stream", &paged_attention_prefill_current_stream);
    m.def("paged_attention_decode_out", &paged_attention_decode_out);
}
