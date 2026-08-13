#pragma once
#include "../common.hpp"

namespace turborl {
namespace utils {

// ============================================================================
// TensorUtils — device-agnostic element-wise helpers over turborl::Tensor.
//   Float32 kernels only; other dtypes fall back to a per-element CPU path.
// ============================================================================
class TensorUtils {
public:
    // Deep copy `src` into `dst` (dst is reallocated to match src).
    static Status Copy(const Tensor& src, Tensor* dst);

    // Fill every element with `value`.
    static Status Fill(Tensor* tensor, float value);

    // dst[i] = src[i] + scalar
    static Status AddScalar(Tensor* tensor, float scalar);

    // dst[i] *= scalar
    static Status Scale(Tensor* tensor, float scalar);

    // dst[i] = a[i] + b[i]  (shapes must match)
    static Status Add(const Tensor& a, const Tensor& b, Tensor* out);

    // Return true iff `a` and `b` have identical shape, dtype and device.
    static bool SameLayout(const Tensor& a, const Tensor& b);
};

} // namespace utils
} // namespace turborl
