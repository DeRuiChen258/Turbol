#pragma once
#include "../common.hpp"

namespace turborl {
namespace utils {

class TensorUtils {
public:
    static Status Copy(const Tensor& src, Tensor* dst);
    static Status Fill(Tensor* tensor, float value);
};

} // namespace utils
} // namespace turbol
