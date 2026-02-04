#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>

template <typename T>
void argmax_(int64_t *max_idx_ptr, T *max_val_ptr, const T *vals_ptr, size_t numel) {
    T max_value;
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        max_value = llaisys::utils::cast<T>(-std::numeric_limits<float>::infinity());
    } else {
        max_value = -std::numeric_limits<T>::infinity();
    }

    int64_t max_index = 0;

    for (size_t i = 0; i < numel; i++) {
        T current_val = vals_ptr[i];
        bool is_greater;

        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            is_greater = llaisys::utils::cast<float>(current_val) > llaisys::utils::cast<float>(max_value);
        } else {
            is_greater = current_val > max_value;
        }

        if (is_greater) {
            max_value = current_val;
            max_index = static_cast<int64_t>(i);
        }
    }

    *max_idx_ptr = max_index;
    *max_val_ptr = max_value;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    int64_t *max_idx_ptr = reinterpret_cast<int64_t *>(max_idx);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(max_idx_ptr, reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(max_idx_ptr, reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(max_idx_ptr, reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
