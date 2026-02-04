#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    // Check dimensions
    ASSERT(in->ndim() == 2, "RMS Norm: input must be 2-D tensor");
    ASSERT(out->ndim() == 2, "RMS Norm: output must be 2-D tensor");
    ASSERT(weight->ndim() == 1, "RMS Norm: weight must be 1-D tensor");

    size_t batch = in->shape()[0];
    size_t dim = in->shape()[1];

    // Check shapes
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(weight->shape()[0] == dim, "RMS Norm: weight length must equal input last dimension");

    // Check data types
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    // Check contiguous
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMS Norm: all tensors must be contiguous");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), batch, dim, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), batch, dim, eps);
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
