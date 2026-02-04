#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    // Check dimensions
    ASSERT(index->ndim() == 1, "Embedding: index must be 1-D tensor");
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2-D tensor");
    ASSERT(out->ndim() == 2, "Embedding: output must be 2-D tensor");
    // Check shapes
    size_t seq_len = index->shape()[0];
    size_t embed_dim = weight->shape()[1];
    ASSERT(out->shape()[0] == seq_len && out->shape()[1] == embed_dim,
           "Embedding: output shape must be [seq_len, embed_dim]");
    // Check data types
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be Int64");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    // Check contiguous
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), seq_len, embed_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), seq_len, embed_dim);
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
