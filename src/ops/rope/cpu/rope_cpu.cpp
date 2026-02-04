#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seqlen, size_t nhead, size_t d, float theta) {
    // RoPE: Rotary Position Embedding
    // out shape: [seqlen, nhead, d]
    // in shape: [seqlen, nhead, d]
    // pos_ids shape: [seqlen]

    size_t half_d = d / 2;

    for (size_t s = 0; s < seqlen; s++) {
        int64_t pos = pos_ids[s];

        for (size_t h = 0; h < nhead; h++) {
            for (size_t j = 0; j < half_d; j++) {
                // Compute angle: phi = pos / (theta^(2j/d))
                // Use double precision for more accurate angle calculation
                double exponent = (2.0 * static_cast<double>(j)) / static_cast<double>(d);
                double freq = 1.0 / std::pow(static_cast<double>(theta), exponent);
                double angle = static_cast<double>(pos) * freq;

                float cos_angle = static_cast<float>(std::cos(angle));
                float sin_angle = static_cast<float>(std::sin(angle));

                // Get indices for the pair (a, b)
                size_t idx_a = s * nhead * d + h * d + j;
                size_t idx_b = s * nhead * d + h * d + j + half_d;

                // Get input values
                float a_val, b_val;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a_val = llaisys::utils::cast<float>(in[idx_a]);
                    b_val = llaisys::utils::cast<float>(in[idx_b]);
                } else {
                    a_val = static_cast<float>(in[idx_a]);
                    b_val = static_cast<float>(in[idx_b]);
                }

                // Apply rotation
                // a' = a*cos - b*sin
                // b' = b*cos + a*sin
                float a_out = a_val * cos_angle - b_val * sin_angle;
                float b_out = b_val * cos_angle + a_val * sin_angle;

                // Store results
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[idx_a] = llaisys::utils::cast<T>(a_out);
                    out[idx_b] = llaisys::utils::cast<T>(b_out);
                } else {
                    out[idx_a] = static_cast<T>(a_out);
                    out[idx_b] = static_cast<T>(b_out);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d, float theta) {
    const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                    pos_ptr, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    pos_ptr, seqlen, nhead, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    pos_ptr, seqlen, nhead, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
