#pragma once

// ============================================================================
// TurboRL <-> libtorch interop
//   Provides zero-copy and copy conversions between turborl::Tensor and
//   torch::Tensor. Only compiled when TURBORL_LIBTORCH is defined.
// ============================================================================

#include "../common.hpp"

#ifdef TURBORL_LIBTORCH
#include <torch/torch.h>

namespace turborl {
namespace torch_interop {

// Map a turborl DataType to the corresponding torch scalar type.
inline torch::ScalarType ToTorchScalarType(DataType dtype) {
    switch (dtype) {
        case DataType::kFloat32:  return torch::kFloat32;
        case DataType::kFloat16:  return torch::kFloat16;
        case DataType::kBFloat16: return torch::kBFloat16;
        case DataType::kInt32:    return torch::kInt32;
        case DataType::kInt64:    return torch::kInt64;
        case DataType::kInt16:    return torch::kInt16;
        case DataType::kInt8:     return torch::kInt8;
        case DataType::kUInt8:    return torch::kUInt8;
        case DataType::kBool:     return torch::kBool;
        default:                  return torch::kFloat32;
    }
}

// Build the torch shape from a turborl Shape.
inline std::vector<int64_t> ToTorchShape(const Shape& shape) {
    return shape.dims;
}

// Zero-copy *borrowed* view of a turborl::Tensor.
//   The caller must guarantee `src` outlives the returned torch::Tensor —
//   no ownership is transferred and no copy is performed.
inline torch::Tensor AsTorchView(Tensor& src) {
    auto sizes = ToTorchShape(src.shape());
    auto options = torch::TensorOptions().dtype(ToTorchScalarType(src.dtype()));
    if (src.is_cuda()) {
        options = options.device(torch::kCUDA, src.device().device_id);
    } else {
        options = options.device(torch::kCPU);
    }
    // from_blob with a no-op deleter: torch does NOT own this memory.
    return torch::from_blob(src.data(), sizes, [](void*) {}, options);
}

// Owned copy of a turborl::Tensor (deep copy into a torch-managed tensor).
inline torch::Tensor ToTorch(const Tensor& src) {
    return AsTorchView(const_cast<Tensor&>(src)).clone();
}

} // namespace torch_interop
} // namespace turbol

#endif // TURBORL_LIBTORCH
