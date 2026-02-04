#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t batch, size_t dim, float eps) {
    // For each row, compute: Y_i = W_i * X_i / sqrt(mean(X^2) + eps)
    for (size_t b = 0; b < batch; b++) {
        const T *in_row = in + b * dim;
        T *out_row = out + b * dim;

        // Compute mean of squares
        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in_row[i]);
            } else {
                val = static_cast<float>(in_row[i]);
            }
            sum_sq += val * val;
        }

        float mean_sq = sum_sq / static_cast<float>(dim);
        float rms = std::sqrt(mean_sq + eps);

        // Normalize and apply weight
        for (size_t i = 0; i < dim; i++) {
            float x_val, w_val;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                x_val = llaisys::utils::cast<float>(in_row[i]);
                w_val = llaisys::utils::cast<float>(weight[i]);
            } else {
                x_val = static_cast<float>(in_row[i]);
                w_val = static_cast<float>(weight[i]);
            }

            float result = (w_val * x_val) / rms;

            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out_row[i] = llaisys::utils::cast<T>(result);
            } else {
                out_row[i] = static_cast<T>(result);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t batch, size_t dim, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                        reinterpret_cast<const float *>(weight), batch, dim, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                        reinterpret_cast<const llaisys::bf16_t *>(weight), batch, dim, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                        reinterpret_cast<const llaisys::fp16_t *>(weight), batch, dim, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
