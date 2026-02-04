#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                     size_t d, size_t dv, float scale) {
    // Compute attention: attn_val = causal_softmax(Q @ K^T * scale) @ V
    // q: [seqlen, nhead, d]
    // k: [total_len, nkvhead, d]
    // v: [total_len, nkvhead, dv]
    // attn_val: [seqlen, nhead, dv]

    // Handle grouped query attention (GQA)
    size_t n_groups = nhead / nkvhead;

    for (size_t s = 0; s < seqlen; s++) {
        for (size_t h = 0; h < nhead; h++) {
            // Determine which kv head this query head corresponds to
            size_t kv_h = h / n_groups;

            // Compute attention scores: Q[s,h] @ K[:,kv_h]^T
            // scores shape: [total_len]
            float scores[total_len];

            for (size_t t = 0; t < total_len; t++) {
                float score = 0.0f;

                for (size_t i = 0; i < d; i++) {
                    float q_val, k_val;

                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q[s * nhead * d + h * d + i]);
                        k_val = llaisys::utils::cast<float>(k[t * nkvhead * d + kv_h * d + i]);
                    } else {
                        q_val = static_cast<float>(q[s * nhead * d + h * d + i]);
                        k_val = static_cast<float>(k[t * nkvhead * d + kv_h * d + i]);
                    }

                    score += q_val * k_val;
                }

                scores[t] = score * scale;
            }

            // Apply causal mask and softmax
            // For position s, only attend to positions 0...(total_len - seqlen + s)
            size_t causal_end = total_len - seqlen + s + 1;

            // Find max for numerical stability
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t t = 0; t < causal_end; t++) {
                if (scores[t] > max_score) {
                    max_score = scores[t];
                }
            }

            // Compute exp and sum
            float exp_sum = 0.0f;
            for (size_t t = 0; t < causal_end; t++) {
                scores[t] = std::exp(scores[t] - max_score);
                exp_sum += scores[t];
            }

            // Normalize
            for (size_t t = 0; t < causal_end; t++) {
                scores[t] /= exp_sum;
            }

            // Zero out masked positions
            for (size_t t = causal_end; t < total_len; t++) {
                scores[t] = 0.0f;
            }

            // Compute attention output: scores @ V[:,kv_h]
            for (size_t i = 0; i < dv; i++) {
                float output = 0.0f;

                for (size_t t = 0; t < total_len; t++) {
                    float v_val;

                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v[t * nkvhead * dv + kv_h * dv + i]);
                    } else {
                        v_val = static_cast<float>(v[t * nkvhead * dv + kv_h * dv + i]);
                    }

                    output += scores[t] * v_val;
                }

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val[s * nhead * dv + h * dv + i] = llaisys::utils::cast<T>(output);
                } else {
                    attn_val[s * nhead * dv + h * dv + i] = static_cast<T>(output);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                    size_t d, size_t dv, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                              reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                              seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q),
                              reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
                              seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q),
                              reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
                              seqlen, total_len, nhead, nkvhead, d, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
