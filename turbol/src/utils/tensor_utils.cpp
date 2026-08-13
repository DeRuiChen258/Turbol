#include "turbol/utils/tensor_utils.hpp"

namespace turborl {
namespace utils {

Status TensorUtils::Copy(const Tensor& src, Tensor* dst) {
    if (!dst) return Status::InvalidArgument("dst is null");
    *dst = src;
    return Status::Ok();
}

Status TensorUtils::Fill(Tensor* tensor, float value) {
    return Status::Ok();
}

} // namespace utils
} // namespace turbol
