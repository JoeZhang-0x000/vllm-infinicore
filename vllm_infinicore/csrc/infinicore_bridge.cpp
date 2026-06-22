#include <torch/extension.h>

#if defined(ENABLE_MUSA_API)
#include <torch_musa/csrc/core/MUSAStream.h>
#elif defined(ENABLE_METAX_API) || defined(ENABLE_CUDA_API)
#include <ATen/cuda/CUDAContext.h>
#endif
#include <infiniop/ops/embedding.h>
#include <infiniop/ops/gemm.h>
#include <infiniop/ops/paged_attention.h>
#include <infiniop/ops/paged_caching.h>
#include <infiniop/ops/paged_attention_prefill.h>
#include <infiniop/ops/rms_norm.h>
#include <infiniop/ops/rope.h>
#include <infiniop/ops/silu_and_mul.h>
#include <infiniop/ops/swiglu.h>
#include <infinicore/adaptor/flash_attention_adaptor.hpp>
#include <infinicore/context/context.hpp>
#include <infinicore/device.hpp>
#include <infinicore/dtype.hpp>
#include <infinicore/ops/linear.hpp>
#include <infinicore/tensor.hpp>

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENABLE_FLASH_ATTN) && defined(ENABLE_METAX_API)
#define VLLM_INFINICORE_FLASH_OP(name) ::name
#elif defined(ENABLE_FLASH_ATTN)
#define VLLM_INFINICORE_FLASH_OP(name) flash::name
#endif

namespace {

void check_infini_status(infiniStatus_t status, const char *op_name) {
    if (status != INFINI_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(op_name) + " failed with status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

infinicore::DataType dtype_from_torch(const at::Tensor &tensor) {
    switch (tensor.scalar_type()) {
    case at::kFloat:
        return infinicore::DataType::F32;
    case at::kHalf:
        return infinicore::DataType::F16;
    case at::kBFloat16:
        return infinicore::DataType::BF16;
    case at::kInt:
        return infinicore::DataType::I32;
    case at::kLong:
        return infinicore::DataType::I64;
    default:
        throw std::runtime_error("unsupported torch dtype for InfiniCore bridge");
    }
}

infinicore::Device device_from_torch(const at::Tensor &tensor) {
    auto index = tensor.device().index();
#if defined(ENABLE_MUSA_API)
    if (tensor.device().type() == c10::musa::kMUSA) {
        return infinicore::Device(infinicore::Device::Type::MOORE,
                                  index < 0 ? 0 : static_cast<size_t>(index));
    }
#endif
#if defined(ENABLE_METAX_API) || defined(ENABLE_CUDA_API)
    if (tensor.is_cuda()) {
        return infinicore::Device(infinicore::Device::Type::METAX,
                                  index < 0 ? 0 : static_cast<size_t>(index));
    }
#endif
    return infinicore::Device::cpu();
}

void *current_stream_from_torch(const at::Tensor &tensor) {
    auto index = tensor.device().index();
    auto device_index = index < 0 ? 0 : index;
#if defined(ENABLE_MUSA_API)
    if (tensor.device().type() == c10::musa::kMUSA) {
        return c10::musa::getCurrentMUSAStream(device_index).stream();
    }
#endif
#if defined(ENABLE_METAX_API) || defined(ENABLE_CUDA_API)
    if (tensor.is_cuda()) {
        return at::cuda::getCurrentCUDAStream(device_index).stream();
    }
#endif
    return nullptr;
}

infinicore::Shape shape_from_torch(const at::Tensor &tensor) {
    infinicore::Shape shape;
    shape.reserve(static_cast<size_t>(tensor.dim()));
    for (int64_t size : tensor.sizes()) {
        shape.push_back(static_cast<size_t>(size));
    }
    return shape;
}

infinicore::Strides strides_from_torch(const at::Tensor &tensor) {
    infinicore::Strides strides;
    strides.reserve(static_cast<size_t>(tensor.dim()));
    for (int64_t stride : tensor.strides()) {
        strides.push_back(static_cast<infinicore::Stride>(stride));
    }
    return strides;
}

infinicore::Tensor wrap_strided(const at::Tensor &tensor) {
    return infinicore::Tensor::strided_from_blob(
        const_cast<void *>(tensor.data_ptr()),
        shape_from_torch(tensor),
        strides_from_torch(tensor),
        dtype_from_torch(tensor),
        device_from_torch(tensor));
}

std::tuple<at::Tensor, at::Tensor> split_kv_cache_bshd(
    const at::Tensor &kv_cache,
    int64_t num_kv_heads) {
    if (kv_cache.dim() < 5 || kv_cache.numel() == 0) {
        throw std::runtime_error("expected non-empty 5D KV cache");
    }

    at::Tensor key_cache;
    at::Tensor value_cache;
    if (kv_cache.size(0) == 2) {
        key_cache = kv_cache.select(0, 0);
        value_cache = kv_cache.select(0, 1);
    } else if (kv_cache.size(1) == 2) {
        key_cache = kv_cache.select(1, 0);
        value_cache = kv_cache.select(1, 1);
    } else {
        throw std::runtime_error("cannot infer KV cache split axis");
    }

    if (key_cache.size(1) == num_kv_heads) {
        return {key_cache.permute({0, 2, 1, 3}),
                value_cache.permute({0, 2, 1, 3})};
    }
    if (key_cache.size(2) == num_kv_heads) {
        return {key_cache, value_cache};
    }
    throw std::runtime_error("cannot infer KV cache layout");
}

std::tuple<at::Tensor, at::Tensor> split_kv_cache_hbsd(
    const at::Tensor &kv_cache,
    int64_t num_kv_heads) {
    if (kv_cache.dim() < 5 || kv_cache.numel() == 0) {
        throw std::runtime_error("expected non-empty 5D KV cache");
    }

    at::Tensor key_cache;
    at::Tensor value_cache;
    if (kv_cache.size(0) == 2) {
        key_cache = kv_cache.select(0, 0);
        value_cache = kv_cache.select(0, 1);
    } else if (kv_cache.size(1) == 2) {
        key_cache = kv_cache.select(1, 0);
        value_cache = kv_cache.select(1, 1);
    } else {
        throw std::runtime_error("cannot infer KV cache split axis");
    }

    if (key_cache.size(1) == num_kv_heads) {
        return {key_cache, value_cache};
    }
    if (key_cache.size(2) == num_kv_heads) {
        return {key_cache.permute({0, 2, 1, 3}),
                value_cache.permute({0, 2, 1, 3})};
    }
    throw std::runtime_error("cannot infer KV cache layout");
}

int64_t flattened_rows(const at::Tensor &tensor) {
    if (tensor.dim() < 1) {
        throw std::runtime_error("expected non-scalar tensor");
    }
    int64_t rows = 1;
    for (int64_t dim = 0; dim < tensor.dim() - 1; ++dim) {
        rows *= tensor.size(dim);
    }
    return rows;
}

} // namespace

at::Tensor linear_current_stream(at::Tensor input,
                                 at::Tensor weight,
                                 c10::optional<at::Tensor> bias);

void paged_attention_decode_out(at::Tensor query,
                                at::Tensor key,
                                at::Tensor kv_cache,
                                at::Tensor decode_seq_lens,
                                at::Tensor decode_block_table,
                                c10::optional<at::Tensor> alibi_slopes,
                                double scale,
                                int64_t num_decode_tokens,
                                int64_t num_decodes,
                                at::Tensor output) {
    if (num_decode_tokens == 0) {
        return;
    }
    if (num_decode_tokens != num_decodes) {
        throw std::runtime_error("InfiniCore bridge does not support speculative decode");
    }
    if (query.dim() != 3) {
        throw std::runtime_error("expected query shape [tokens, heads, head_dim]");
    }

    auto q = query.narrow(0, 0, num_decode_tokens);
    auto out = output.narrow(0, 0, num_decode_tokens).view(q.sizes());
    auto caches = split_kv_cache_hbsd(kv_cache, key.size(1));

    auto out_tensor = wrap_strided(out);
    auto q_tensor = wrap_strided(q);
    auto k_tensor = wrap_strided(std::get<0>(caches));
    auto v_tensor = wrap_strided(std::get<1>(caches));
    auto block_tensor = wrap_strided(decode_block_table);
    auto len_tensor = wrap_strided(decode_seq_lens);
    std::optional<infinicore::Tensor> alibi;
    if (alibi_slopes.has_value() && alibi_slopes.value().defined()) {
        alibi = wrap_strided(alibi_slopes.value());
    }

    infiniopPagedAttentionDescriptor_t desc = nullptr;
    auto handle = infinicore::context::getInfiniopHandle(device_from_torch(query));
    check_infini_status(
        infiniopCreatePagedAttentionDescriptor(
            handle,
            &desc,
            out_tensor->desc(),
            q_tensor->desc(),
            k_tensor->desc(),
            v_tensor->desc(),
            block_tensor->desc(),
            len_tensor->desc(),
            alibi.has_value() ? alibi.value()->desc() : nullptr,
            static_cast<float>(scale)),
        "infiniopCreatePagedAttentionDescriptor");

    size_t workspace_size = 0;
    try {
        check_infini_status(
            infiniopGetPagedAttentionWorkspaceSize(desc, &workspace_size),
            "infiniopGetPagedAttentionWorkspaceSize");
        at::Tensor workspace;
        void *workspace_ptr = nullptr;
        if (workspace_size > 0) {
            workspace = at::empty(
                {static_cast<int64_t>(workspace_size)},
                query.options().dtype(at::kByte));
            workspace_ptr = workspace.data_ptr();
        }
        check_infini_status(
            infiniopPagedAttention(
                desc,
                workspace_ptr,
                workspace_size,
                out.data_ptr(),
                q.data_ptr(),
                std::get<0>(caches).data_ptr(),
                std::get<1>(caches).data_ptr(),
                decode_block_table.data_ptr(),
                decode_seq_lens.data_ptr(),
                alibi_slopes.has_value() && alibi_slopes.value().defined()
                    ? alibi_slopes.value().data_ptr()
                    : nullptr,
                current_stream_from_torch(query)),
            "infiniopPagedAttention");
    } catch (...) {
        infiniopDestroyPagedAttentionDescriptor(desc);
        throw;
    }
    check_infini_status(
        infiniopDestroyPagedAttentionDescriptor(desc),
        "infiniopDestroyPagedAttentionDescriptor");
}

#if defined(ENABLE_FLASH_ATTN)
void paged_attention_decode_flash_out(at::Tensor query,
                                      at::Tensor key,
                                      at::Tensor kv_cache,
                                      at::Tensor decode_seq_lens,
                                      at::Tensor decode_block_table,
                                      c10::optional<at::Tensor> alibi_slopes,
                                      double scale,
                                      int64_t num_decode_tokens,
                                      int64_t num_decodes,
                                      int64_t num_splits,
                                      at::Tensor output) {
    if (num_decode_tokens == 0) {
        return;
    }
    if (num_decode_tokens != num_decodes) {
        throw std::runtime_error("InfiniCore flash bridge does not support speculative decode");
    }
    if (query.dim() != 3) {
        throw std::runtime_error("expected query shape [tokens, heads, head_dim]");
    }

    auto q = query.narrow(0, 0, num_decode_tokens);
    auto q_for_fa = q.view({num_decode_tokens, 1, q.size(1), q.size(2)});
    auto out_tensor = output.narrow(0, 0, num_decode_tokens).view(q_for_fa.sizes());
    auto caches = split_kv_cache_bshd(kv_cache, key.size(1));

    std::optional<const at::Tensor> k_new = std::nullopt;
    std::optional<const at::Tensor> v_new = std::nullopt;
    std::optional<const at::Tensor> seqlens_k = decode_seq_lens;
    std::optional<const at::Tensor> rotary_cos = std::nullopt;
    std::optional<const at::Tensor> rotary_sin = std::nullopt;
    std::optional<const at::Tensor> cache_batch_idx = std::nullopt;
    std::optional<const at::Tensor> leftpad_k = std::nullopt;
    std::optional<at::Tensor> block_table = decode_block_table;
    std::optional<at::Tensor> alibi =
        alibi_slopes.has_value() && alibi_slopes.value().defined()
            ? std::optional<at::Tensor>(alibi_slopes.value())
            : std::nullopt;

    const bool use_dynamic_out = q_for_fa.dim() == 4 && std::get<0>(caches).dim() == 4
                              && q_for_fa.size(1) == 1
                              && q_for_fa.size(2) > std::get<0>(caches).size(2)
                              && q_for_fa.size(3) % 8 == 0
                              && !alibi.has_value();
    auto out = use_dynamic_out ? std::optional<at::Tensor>(std::nullopt)
                               : std::optional<at::Tensor>(out_tensor);

    std::optional<at::Tensor> flash_attn_mars_ext = std::nullopt;
    auto result = VLLM_INFINICORE_FLASH_OP(mha_fwd_kvcache)(
        q_for_fa,
        std::get<0>(caches),
        std::get<1>(caches),
        k_new,
        v_new,
        seqlens_k,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        leftpad_k,
        block_table,
        alibi,
        out,
        static_cast<float>(scale),
        true,
        -1,
        -1,
        0.0f,
        false,
        static_cast<int>(num_splits),
        flash_attn_mars_ext);

    if (use_dynamic_out) {
        out_tensor.copy_(result[0]);
    }
}
#endif

at::Tensor lm_head(at::Tensor input,
                   at::Tensor weight,
                   c10::optional<at::Tensor> bias) {
    if (bias.has_value() && bias.value().defined()) {
        at::Tensor out = linear_current_stream(input, weight, c10::optional<at::Tensor>());
        out.add_(bias.value());
        return out;
    }
    return linear_current_stream(input, weight, c10::optional<at::Tensor>());
}

at::Tensor linear_current_stream(at::Tensor input,
                                 at::Tensor weight,
                                 c10::optional<at::Tensor> bias) {
    if (bias.has_value() && bias.value().defined()) {
        throw std::runtime_error("linear_current_stream does not support bias");
    }
    if (input.dim() < 1 || weight.dim() != 2) {
        throw std::runtime_error("expected input [..., in_features] and weight [out_features, in_features]");
    }
    const int64_t in_features = input.size(input.dim() - 1);
    if (weight.size(1) != in_features) {
        throw std::runtime_error("linear_current_stream input/weight feature mismatch");
    }

    std::vector<int64_t> out_shape(input.sizes().begin(), input.sizes().end());
    out_shape.back() = weight.size(0);
    at::Tensor out = at::empty(out_shape, input.options());

    const int64_t rows = flattened_rows(input);
    auto input_2d = input.view({rows, in_features});
    auto out_2d = out.view({rows, weight.size(0)});
    auto weight_t = weight.transpose(0, 1);

    auto c = wrap_strided(out_2d);
    auto a = wrap_strided(input_2d);
    auto b = wrap_strided(weight_t);

    infiniopGemmDescriptor_t desc = nullptr;
    auto handle = infinicore::context::getInfiniopHandle(device_from_torch(input));
    check_infini_status(
        infiniopCreateGemmDescriptor(handle, &desc, c->desc(), a->desc(), b->desc()),
        "infiniopCreateGemmDescriptor");

    size_t workspace_size = 0;
    try {
        check_infini_status(
            infiniopGetGemmWorkspaceSize(desc, &workspace_size),
            "infiniopGetGemmWorkspaceSize");
        at::Tensor workspace;
        void *workspace_ptr = nullptr;
        if (workspace_size > 0) {
            workspace = at::empty(
                {static_cast<int64_t>(workspace_size)},
                input.options().dtype(at::kByte));
            workspace_ptr = workspace.data_ptr();
        }
        void *stream = current_stream_from_torch(input);
        check_infini_status(
            infiniopGemm(
                desc,
                workspace_ptr,
                workspace_size,
                out_2d.data_ptr(),
                input_2d.data_ptr(),
                weight_t.data_ptr(),
                1.0f,
                0.0f,
                stream),
            "infiniopGemm");
    } catch (...) {
        infiniopDestroyGemmDescriptor(desc);
        throw;
    }
    check_infini_status(infiniopDestroyGemmDescriptor(desc), "infiniopDestroyGemmDescriptor");
    return out;
}

at::Tensor embedding_current_stream(at::Tensor input, at::Tensor weight) {
    if (weight.dim() != 2) {
        throw std::runtime_error("embedding_current_stream expects a 2D weight tensor");
    }
    if (input.device() != weight.device()) {
        throw std::runtime_error("embedding_current_stream expects input and weight on the same device");
    }
    if (input.scalar_type() != at::kInt && input.scalar_type() != at::kLong) {
        throw std::runtime_error("embedding_current_stream expects int32 or int64 input");
    }

    std::vector<int64_t> out_shape(input.sizes().begin(), input.sizes().end());
    out_shape.push_back(weight.size(1));
    at::Tensor out = at::empty(out_shape, weight.options());

    auto y = wrap_strided(out);
    auto x = wrap_strided(input);
    auto w = wrap_strided(weight);

    infiniopEmbeddingDescriptor_t desc = nullptr;
    auto handle = infinicore::context::getInfiniopHandle(device_from_torch(weight));
    check_infini_status(
        infiniopCreateEmbeddingDescriptor(handle, &desc, y->desc(), x->desc(), w->desc()),
        "infiniopCreateEmbeddingDescriptor");

    try {
        check_infini_status(
            infiniopEmbedding(
                desc,
                out.data_ptr(),
                input.data_ptr(),
                weight.data_ptr(),
                current_stream_from_torch(weight)),
            "infiniopEmbedding");
    } catch (...) {
        infiniopDestroyEmbeddingDescriptor(desc);
        throw;
    }
    check_infini_status(
        infiniopDestroyEmbeddingDescriptor(desc),
        "infiniopDestroyEmbeddingDescriptor");
    return out;
}

at::Tensor rms_norm_current_stream(at::Tensor input,
                                   at::Tensor weight,
                                   double epsilon) {
    at::Tensor out = at::empty_like(input);
    auto y = wrap_strided(out);
    auto x = wrap_strided(input);
    auto w = wrap_strided(weight);

    infiniopRMSNormDescriptor_t desc = nullptr;
    auto handle = infinicore::context::getInfiniopHandle(device_from_torch(input));
    check_infini_status(
        infiniopCreateRMSNormDescriptor(
            handle,
            &desc,
            y->desc(),
            x->desc(),
            w->desc(),
            static_cast<float>(epsilon)),
        "infiniopCreateRMSNormDescriptor");

    size_t workspace_size = 0;
    try {
        check_infini_status(
            infiniopGetRMSNormWorkspaceSize(desc, &workspace_size),
            "infiniopGetRMSNormWorkspaceSize");
        at::Tensor workspace;
        void *workspace_ptr = nullptr;
        if (workspace_size > 0) {
            workspace = at::empty(
                {static_cast<int64_t>(workspace_size)},
                input.options().dtype(at::kByte));
            workspace_ptr = workspace.data_ptr();
        }
        void *stream = current_stream_from_torch(input);
        check_infini_status(
            infiniopRMSNorm(
                desc,
                workspace_ptr,
                workspace_size,
                out.data_ptr(),
                input.data_ptr(),
                weight.data_ptr(),
                stream),
            "infiniopRMSNorm");
    } catch (...) {
        infiniopDestroyRMSNormDescriptor(desc);
        throw;
    }
    check_infini_status(infiniopDestroyRMSNormDescriptor(desc), "infiniopDestroyRMSNormDescriptor");
    return out;
}

at::Tensor swiglu_current_stream(at::Tensor up, at::Tensor gate) {
    if (up.sizes() != gate.sizes()) {
        throw std::runtime_error("swiglu_current_stream expects matching input shapes");
    }
    at::Tensor out = at::empty_like(up);
    auto c = wrap_strided(out);
    auto a = wrap_strided(up);
    auto b = wrap_strided(gate);

    infiniopSwiGLUDescriptor_t desc = nullptr;
    auto handle = infinicore::context::getInfiniopHandle(device_from_torch(up));
    check_infini_status(
        infiniopCreateSwiGLUDescriptor(handle, &desc, c->desc(), a->desc(), b->desc()),
        "infiniopCreateSwiGLUDescriptor");

    size_t workspace_size = 0;
    try {
        check_infini_status(
            infiniopGetSwiGLUWorkspaceSize(desc, &workspace_size),
            "infiniopGetSwiGLUWorkspaceSize");
        at::Tensor workspace;
        void *workspace_ptr = nullptr;
        if (workspace_size > 0) {
            workspace = at::empty(
                {static_cast<int64_t>(workspace_size)},
                up.options().dtype(at::kByte));
            workspace_ptr = workspace.data_ptr();
        }
        void *stream = current_stream_from_torch(up);
        check_infini_status(
            infiniopSwiGLU(
                desc,
                workspace_ptr,
                workspace_size,
                out.data_ptr(),
                up.data_ptr(),
                gate.data_ptr(),
                stream),
            "infiniopSwiGLU");
    } catch (...) {
        infiniopDestroySwiGLUDescriptor(desc);
        throw;
    }
    check_infini_status(infiniopDestroySwiGLUDescriptor(desc), "infiniopDestroySwiGLUDescriptor");
    return out;
}

at::Tensor silu_and_mul_current_stream(at::Tensor input) {
    if (input.dim() < 1 || input.size(input.dim() - 1) % 2 != 0) {
        throw std::runtime_error("silu_and_mul_current_stream expects input last dimension to be even");
    }
    auto output_sizes = input.sizes().vec();
    output_sizes.back() /= 2;
    at::Tensor out = at::empty(output_sizes, input.options());

    auto y = wrap_strided(out);
    auto x = wrap_strided(input);

    infiniopSiluAndMulDescriptor_t desc = nullptr;
    auto handle = infinicore::context::getInfiniopHandle(device_from_torch(input));
    check_infini_status(
        infiniopCreateSiluAndMulDescriptor(handle, &desc, y->desc(), x->desc()),
        "infiniopCreateSiluAndMulDescriptor");

    size_t workspace_size = 0;
    try {
        check_infini_status(
            infiniopGetSiluAndMulWorkspaceSize(desc, &workspace_size),
            "infiniopGetSiluAndMulWorkspaceSize");
        at::Tensor workspace;
        void *workspace_ptr = nullptr;
        if (workspace_size > 0) {
            workspace = at::empty(
                {static_cast<int64_t>(workspace_size)},
                input.options().dtype(at::kByte));
            workspace_ptr = workspace.data_ptr();
        }
        void *stream = current_stream_from_torch(input);
        check_infini_status(
            infiniopSiluAndMul(
                desc,
                workspace_ptr,
                workspace_size,
                out.data_ptr(),
                input.data_ptr(),
                stream),
            "infiniopSiluAndMul");
    } catch (...) {
        infiniopDestroySiluAndMulDescriptor(desc);
        throw;
    }
    check_infini_status(
        infiniopDestroySiluAndMulDescriptor(desc),
        "infiniopDestroySiluAndMulDescriptor");
    return out;
}

at::Tensor rope_current_stream(at::Tensor input,
                               at::Tensor positions,
                               at::Tensor sin_table,
                               at::Tensor cos_table,
                               bool is_neox_style) {
    if (input.dim() != 3 && input.dim() != 4) {
        throw std::runtime_error("expected RoPE input shape [tokens, heads, dim] or [batch, tokens, heads, dim]");
    }
    at::Tensor out = at::empty_like(input);
    auto y = wrap_strided(out);
    auto x = wrap_strided(input);
    auto pos = wrap_strided(positions);
    auto sin = wrap_strided(sin_table);
    auto cos = wrap_strided(cos_table);

    infiniopRoPEDescriptor_t desc = nullptr;
    auto handle = infinicore::context::getInfiniopHandle(device_from_torch(input));
    auto algo = is_neox_style ? INFINIOP_ROPE_ALGO_GPT_NEOX : INFINIOP_ROPE_ALGO_GPT_J;
    check_infini_status(
        infiniopCreateRoPEDescriptor(
            handle,
            &desc,
            y->desc(),
            x->desc(),
            pos->desc(),
            sin->desc(),
            cos->desc(),
            algo),
        "infiniopCreateRoPEDescriptor");

    size_t workspace_size = 0;
    try {
        check_infini_status(
            infiniopGetRoPEWorkspaceSize(desc, &workspace_size),
            "infiniopGetRoPEWorkspaceSize");
        at::Tensor workspace;
        void *workspace_ptr = nullptr;
        if (workspace_size > 0) {
            workspace = at::empty(
                {static_cast<int64_t>(workspace_size)},
                input.options().dtype(at::kByte));
            workspace_ptr = workspace.data_ptr();
        }
        void *stream = current_stream_from_torch(input);
        check_infini_status(
            infiniopRoPE(
                desc,
                workspace_ptr,
                workspace_size,
                out.data_ptr(),
                input.data_ptr(),
                positions.data_ptr(),
                sin_table.data_ptr(),
                cos_table.data_ptr(),
                stream),
            "infiniopRoPE");
    } catch (...) {
        infiniopDestroyRoPEDescriptor(desc);
        throw;
    }
    check_infini_status(infiniopDestroyRoPEDescriptor(desc), "infiniopDestroyRoPEDescriptor");
    return out;
}

void store_kv_cache_current_stream(at::Tensor kv_cache,
                                   at::Tensor key,
                                   at::Tensor value,
                                   at::Tensor slot_mapping) {
    auto caches = split_kv_cache_hbsd(kv_cache, key.size(1));
    auto flat_slots = slot_mapping.flatten();
    auto k_cache_tensor = wrap_strided(std::get<0>(caches));
    auto v_cache_tensor = wrap_strided(std::get<1>(caches));
    auto key_tensor = wrap_strided(key);
    auto value_tensor = wrap_strided(value);
    auto slot_tensor = wrap_strided(flat_slots);

    infiniopPagedCachingDescriptor_t desc = nullptr;
    auto handle = infinicore::context::getInfiniopHandle(device_from_torch(key));
    check_infini_status(
        infiniopCreatePagedCachingDescriptor(
            handle,
            &desc,
            k_cache_tensor->desc(),
            v_cache_tensor->desc(),
            key_tensor->desc(),
            value_tensor->desc(),
            slot_tensor->desc()),
        "infiniopCreatePagedCachingDescriptor");

    size_t workspace_size = 0;
    try {
        check_infini_status(
            infiniopGetPagedCachingWorkspaceSize(desc, &workspace_size),
            "infiniopGetPagedCachingWorkspaceSize");
        at::Tensor workspace;
        void *workspace_ptr = nullptr;
        if (workspace_size > 0) {
            workspace = at::empty(
                {static_cast<int64_t>(workspace_size)},
                key.options().dtype(at::kByte));
            workspace_ptr = workspace.data_ptr();
        }
        void *stream = current_stream_from_torch(key);
        check_infini_status(
            infiniopPagedCaching(
                desc,
                workspace_ptr,
                workspace_size,
                std::get<0>(caches).data_ptr(),
                std::get<1>(caches).data_ptr(),
                key.data_ptr(),
                value.data_ptr(),
                flat_slots.data_ptr(),
                stream),
            "infiniopPagedCaching");
    } catch (...) {
        infiniopDestroyPagedCachingDescriptor(desc);
        throw;
    }
    check_infini_status(
        infiniopDestroyPagedCachingDescriptor(desc),
        "infiniopDestroyPagedCachingDescriptor");
}

void paged_attention_prefill_current_stream(at::Tensor query,
                                            at::Tensor key_cache,
                                            at::Tensor value_cache,
                                            at::Tensor block_table,
                                            at::Tensor total_kv_lens,
                                            at::Tensor query_start_loc,
                                            c10::optional<at::Tensor> alibi_slopes,
                                            double scale,
                                            at::Tensor output) {
    if (query.numel() == 0) {
        return;
    }
    if (query.dim() != 3 || output.sizes() != query.sizes()) {
        throw std::runtime_error("paged_attention_prefill_current_stream expects query/output [tokens, heads, head_dim]");
    }

    auto out_tensor = wrap_strided(output);
    auto q_tensor = wrap_strided(query);
    auto k_tensor = wrap_strided(key_cache);
    auto v_tensor = wrap_strided(value_cache);
    auto block_tensor = wrap_strided(block_table);
    auto len_tensor = wrap_strided(total_kv_lens);
    auto q_start_tensor = wrap_strided(query_start_loc);
    std::optional<infinicore::Tensor> alibi;
    if (alibi_slopes.has_value() && alibi_slopes.value().defined()) {
        alibi = wrap_strided(alibi_slopes.value());
    }

    infiniopPagedAttentionPrefillDescriptor_t desc = nullptr;
    auto handle = infinicore::context::getInfiniopHandle(device_from_torch(query));
    check_infini_status(
        infiniopCreatePagedAttentionPrefillDescriptor(
            handle,
            &desc,
            out_tensor->desc(),
            q_tensor->desc(),
            k_tensor->desc(),
            v_tensor->desc(),
            block_tensor->desc(),
            len_tensor->desc(),
            q_start_tensor->desc(),
            alibi.has_value() ? alibi.value()->desc() : nullptr,
            static_cast<float>(scale)),
        "infiniopCreatePagedAttentionPrefillDescriptor");

    size_t workspace_size = 0;
    try {
        check_infini_status(
            infiniopGetPagedAttentionPrefillWorkspaceSize(desc, &workspace_size),
            "infiniopGetPagedAttentionPrefillWorkspaceSize");
        at::Tensor workspace;
        void *workspace_ptr = nullptr;
        if (workspace_size > 0) {
            workspace = at::empty(
                {static_cast<int64_t>(workspace_size)},
                query.options().dtype(at::kByte));
            workspace_ptr = workspace.data_ptr();
        }
        check_infini_status(
            infiniopPagedAttentionPrefill(
                desc,
                workspace_ptr,
                workspace_size,
                output.data_ptr(),
                query.data_ptr(),
                key_cache.data_ptr(),
                value_cache.data_ptr(),
                block_table.data_ptr(),
                total_kv_lens.data_ptr(),
                query_start_loc.data_ptr(),
                alibi_slopes.has_value() && alibi_slopes.value().defined()
                    ? alibi_slopes.value().data_ptr()
                    : nullptr,
                current_stream_from_torch(query)),
            "infiniopPagedAttentionPrefill");
    } catch (...) {
        infiniopDestroyPagedAttentionPrefillDescriptor(desc);
        throw;
    }
    check_infini_status(
        infiniopDestroyPagedAttentionPrefillDescriptor(desc),
        "infiniopDestroyPagedAttentionPrefillDescriptor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention_decode_out", &paged_attention_decode_out);
#if defined(ENABLE_FLASH_ATTN)
    m.def("paged_attention_decode_flash_out", &paged_attention_decode_flash_out);
#endif
    m.def("embedding_current_stream", &embedding_current_stream);
    m.def("linear_current_stream", &linear_current_stream);
    m.def("rms_norm_current_stream", &rms_norm_current_stream);
    m.def("swiglu_current_stream", &swiglu_current_stream);
    m.def("silu_and_mul_current_stream", &silu_and_mul_current_stream);
    m.def("rope_current_stream", &rope_current_stream);
    m.def("store_kv_cache_current_stream", &store_kv_cache_current_stream);
    m.def("paged_attention_prefill_current_stream", &paged_attention_prefill_current_stream);
    m.def("lm_head", &lm_head);
}
