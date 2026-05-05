#include <torch/extension.h>

#include <infinicore/device.hpp>
#include <infinicore/dtype.hpp>
#include <infinicore/ops/linear.hpp>
#include <infinicore/ops/mha_kvcache.hpp>
#include <infinicore/tensor.hpp>

#include <optional>
#include <stdexcept>
#include <vector>

namespace {

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
    if (!tensor.is_cuda()) {
        return infinicore::Device::cpu();
    }
    auto index = tensor.device().index();
    return infinicore::Device(infinicore::Device::Type::METAX,
                              index < 0 ? 0 : static_cast<size_t>(index));
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

} // namespace

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
    auto q_for_fa = q.view({num_decode_tokens, 1, q.size(1), q.size(2)});
    auto out = output.narrow(0, 0, num_decode_tokens).view(q_for_fa.sizes());
    auto caches = split_kv_cache_bshd(kv_cache, key.size(1));

    std::optional<infinicore::Tensor> alibi;
    if (alibi_slopes.has_value() && alibi_slopes.value().defined()) {
        alibi = wrap_strided(alibi_slopes.value());
    }

    infinicore::op::mha_kvcache_(
        wrap_strided(out),
        wrap_strided(q_for_fa),
        wrap_strided(std::get<0>(caches)),
        wrap_strided(std::get<1>(caches)),
        wrap_strided(decode_seq_lens),
        wrap_strided(decode_block_table),
        alibi,
        static_cast<float>(scale));
}

at::Tensor lm_head(at::Tensor input,
                   at::Tensor weight,
                   c10::optional<at::Tensor> bias) {
    std::vector<int64_t> out_shape(input.sizes().begin(), input.sizes().end());
    if (out_shape.empty()) {
        throw std::runtime_error("expected non-scalar LMHead input");
    }
    out_shape.back() = weight.size(0);
    at::Tensor out = at::empty(out_shape, input.options());

    std::optional<infinicore::Tensor> infini_bias;
    if (bias.has_value() && bias.value().defined()) {
        infini_bias = wrap_strided(bias.value());
    }

    infinicore::op::linear_(
        wrap_strided(out),
        wrap_strided(input),
        wrap_strided(weight),
        infini_bias);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention_decode_out", &paged_attention_decode_out);
    m.def("lm_head", &lm_head);
}
