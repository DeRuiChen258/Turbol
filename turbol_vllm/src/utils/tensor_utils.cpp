#include "turbol/utils/tensor_utils.hpp"

#ifdef TURBORL_HAS_CUDA
extern "C" {
cudaError_t gpu_add(const float* a, const float* b, float* c, int n, cudaStream_t stream);
cudaError_t gpu_fill(float* out, float value, int n, cudaStream_t stream);
cudaError_t gpu_add_scalar(float* inout, float scalar, int n, cudaStream_t stream);
cudaError_t gpu_scale(float* inout, float scalar, int n, cudaStream_t stream);
}
#endif

namespace turborl {
namespace utils {

bool TensorUtils::SameLayout(const Tensor& a, const Tensor& b) {
    return a.shape().dims == b.shape().dims &&
           a.dtype() == b.dtype() &&
           a.device() == b.device();
}

Status TensorUtils::Copy(const Tensor& src, Tensor* dst) {
    if (!dst) return Status::InvalidArgument("dst is null");
    *dst = src;  // deep copy via Tensor copy-assignment
    return Status::Ok();
}

Status TensorUtils::Fill(Tensor* tensor, float value) {
    if (!tensor) return Status::InvalidArgument("tensor is null");
    if (tensor->numel() == 0) return Status::Ok();

    const int n = static_cast<int>(tensor->numel());
    if (tensor->dtype() != DataType::kFloat32)
        return Status::Unimplemented("Fill only supports Float32 tensors");

#ifdef TURBORL_HAS_CUDA
    if (tensor->is_cuda()) {
        if (gpu_fill(tensor->data<float>(), value, n, nullptr) != cudaSuccess)
            return Status::CudaError("gpu_fill failed");
        return Status::Ok();
    }
#endif
    std::fill(tensor->data<float>(), tensor->data<float>() + n, value);
    return Status::Ok();
}

Status TensorUtils::AddScalar(Tensor* tensor, float scalar) {
    if (!tensor) return Status::InvalidArgument("tensor is null");
    if (tensor->dtype() != DataType::kFloat32)
        return Status::Unimplemented("AddScalar only supports Float32 tensors");
    const int n = static_cast<int>(tensor->numel());

#ifdef TURBORL_HAS_CUDA
    if (tensor->is_cuda()) {
        if (gpu_add_scalar(tensor->data<float>(), scalar, n, nullptr) != cudaSuccess)
            return Status::CudaError("gpu_add_scalar failed");
        return Status::Ok();
    }
#endif
    float* p = tensor->data<float>();
    for (int i = 0; i < n; ++i) p[i] += scalar;
    return Status::Ok();
}

Status TensorUtils::Scale(Tensor* tensor, float scalar) {
    if (!tensor) return Status::InvalidArgument("tensor is null");
    if (tensor->dtype() != DataType::kFloat32)
        return Status::Unimplemented("Scale only supports Float32 tensors");
    const int n = static_cast<int>(tensor->numel());

#ifdef TURBORL_HAS_CUDA
    if (tensor->is_cuda()) {
        if (gpu_scale(tensor->data<float>(), scalar, n, nullptr) != cudaSuccess)
            return Status::CudaError("gpu_scale failed");
        return Status::Ok();
    }
#endif
    float* p = tensor->data<float>();
    for (int i = 0; i < n; ++i) p[i] *= scalar;
    return Status::Ok();
}

Status TensorUtils::Add(const Tensor& a, const Tensor& b, Tensor* out) {
    if (!out) return Status::InvalidArgument("out is null");
    if (a.dtype() != DataType::kFloat32 || b.dtype() != DataType::kFloat32)
        return Status::Unimplemented("Add only supports Float32 tensors");
    if (a.shape().dims != b.shape().dims)
        return Status::InvalidArgument("Add: shape mismatch");
    if (a.device() != b.device())
        return Status::InvalidArgument("Add: device mismatch");

    const int n = static_cast<int>(a.numel());
    if (out->numel() != a.numel() || out->device() != a.device()) {
        *out = Tensor(a.shape(), DataType::kFloat32, a.device());
    }

#ifdef TURBORL_HAS_CUDA
    if (a.is_cuda()) {
        if (gpu_add(a.data<float>(), b.data<float>(), out->data<float>(), n, nullptr) != cudaSuccess)
            return Status::CudaError("gpu_add failed");
        return Status::Ok();
    }
#endif
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    float* po = out->data<float>();
    for (int i = 0; i < n; ++i) po[i] = pa[i] + pb[i];
    return Status::Ok();
}

} // namespace utils
} // namespace turborl
