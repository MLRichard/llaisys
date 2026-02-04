#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    // Check dimensions
    ASSERT(q->ndim() == 3, "Self Attention: q must be 3-D tensor [seqlen, nhead, d]");
    ASSERT(k->ndim() == 3, "Self Attention: k must be 3-D tensor [total_len, nkvhead, d]");
    ASSERT(v->ndim() == 3, "Self Attention: v must be 3-D tensor [total_len, nkvhead, dv]");
    ASSERT(attn_val->ndim() == 3, "Self Attention: attn_val must be 3-D tensor [seqlen, nhead, dv]");

    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];

    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    ASSERT(k->shape()[2] == d, "Self Attention: k dimension must match q dimension");

    ASSERT(v->shape()[0] == total_len, "Self Attention: v length must match k length");
    ASSERT(v->shape()[1] == nkvhead, "Self Attention: v nkvhead must match k nkvhead");
    size_t dv = v->shape()[2];

    // Check output shape
    ASSERT(attn_val->shape()[0] == seqlen && attn_val->shape()[1] == nhead && attn_val->shape()[2] == dv,
           "Self Attention: attn_val shape must be [seqlen, nhead, dv]");

    // Check grouped query attention constraint
    ASSERT(nhead % nkvhead == 0, "Self Attention: nhead must be divisible by nkvhead for GQA");

    // Check data types
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    // Check contiguous
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "Self Attention: all tensors must be contiguous");

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  attn_val->dtype(), seqlen, total_len, nhead, nkvhead, d, dv, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  attn_val->dtype(), seqlen, total_len, nhead, nkvhead, d, dv, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
