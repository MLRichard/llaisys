#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    // Check dimensions
    ASSERT(in->ndim() == 3, "RoPE: input must be 3-D tensor [seqlen, nhead, d]");
    ASSERT(out->ndim() == 3, "RoPE: output must be 3-D tensor [seqlen, nhead, d]");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1-D tensor [seqlen]");

    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t d = in->shape()[2];

    // Check shapes
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(pos_ids->shape()[0] == seqlen, "RoPE: pos_ids length must equal seqlen");
    ASSERT(d % 2 == 0, "RoPE: dimension d must be even");

    // Check data types
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be Int64");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    // Check contiguous
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, d, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seqlen, nhead, d, theta);
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
