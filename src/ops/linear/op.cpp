#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // Check dimensions
    ASSERT(in->ndim() == 2, "Linear: input must be 2-D tensor");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2-D tensor");
    ASSERT(out->ndim() == 2, "Linear: output must be 2-D tensor");

    size_t batch = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];

    // Check shapes
    ASSERT(weight->shape()[1] == in_features, "Linear: weight shape[1] must equal input shape[1]");
    ASSERT(out->shape()[0] == batch && out->shape()[1] == out_features,
           "Linear: output shape must be [batch, out_features]");

    // Check bias if provided
    bool has_bias = (bias != nullptr);
    if (has_bias) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        ASSERT(bias->ndim() == 1, "Linear: bias must be 1-D tensor");
        ASSERT(bias->shape()[0] == out_features, "Linear: bias shape must match out_features");
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous");
    } else {
        CHECK_SAME_DEVICE(out, in, weight);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    }

    // Check contiguous
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: out, in, weight must be contiguous");

    const std::byte *bias_data = has_bias ? bias->data() : nullptr;

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data,
                          out->dtype(), batch, in_features, out_features, has_bias);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data,
                          out->dtype(), batch, in_features, out_features, has_bias);
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
