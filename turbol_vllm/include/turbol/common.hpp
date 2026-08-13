#pragma once

// ============================================================================
// TurboRL Common — foundational types shared across the whole library.
//   - Status / StatusCode error model
//   - Shape, Device, DataType
//   - Tensor: device-agnostic (CPU / CUDA) contiguous buffer
// ============================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <array>
#include <functional>
#include <future>
#include <atomic>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <type_traits>

// ============================================================================
// Platform / Compiler / Standard detection
// ============================================================================
#if defined(__linux__)
    #define TURBORL_PLATFORM_LINUX 1
#elif defined(_WIN32)
    #define TURBORL_PLATFORM_WINDOWS 1
#elif defined(__APPLE__)
    #define TURBORL_PLATFORM_MACOS 1
#endif

#if defined(__GNUC__) && !defined(__clang__)
    #define TURBORL_COMPILER_GCC 1
#elif defined(__clang__)
    #define TURBORL_COMPILER_CLANG 1
#elif defined(_MSC_VER)
    #define TURBORL_COMPILER_MSVC 1
#endif

#if defined(__cplusplus) && __cplusplus >= 202302L
    #define TURBORL_CXX23 1
#elif defined(__cplusplus) && __cplusplus >= 202002L
    #define TURBORL_CXX20 1
#elif defined(__cplusplus) && __cplusplus >= 201703L
    #define TURBORL_CXX17 1
#endif

// CUDA detection — CMake defines TURBORL_CUDA for host .cpp, __CUDACC__ for .cu.
#if defined(TURBORL_CUDA) || defined(__CUDACC__) || defined(__NVCC__)
    #define TURBORL_HAS_CUDA 1
    #include <cuda_runtime.h>
#endif

// ============================================================================
// Export macro
// ============================================================================
#if defined(_WIN32) && defined(TURBORL_EXPORTS)
    #define TURBORL_API __declspec(dllexport)
#elif defined(_WIN32)
    #define TURBORL_API __declspec(dllimport)
#elif defined(__GNUC__)
    #define TURBORL_API __attribute__((visibility("default")))
#else
    #define TURBORL_API
#endif

#define TURBORL_VERSION_MAJOR 1
#define TURBORL_VERSION_MINOR 0
#define TURBORL_VERSION_PATCH 0

namespace turborl {

// ============================================================================
// Status and error handling
// ============================================================================
enum class StatusCode {
    kOk = 0,
    kInvalidArgument = 1,
    kNotFound = 2,
    kAlreadyExists = 3,
    kResourceExhausted = 4,
    kUnimplemented = 5,
    kInternalError = 6,
    kUnavailable = 7,
    kTimeout = 8,
    kCudaError = 100,
};

class Status {
public:
    Status() : code_(StatusCode::kOk), message_() {}
    Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

    bool ok() const { return code_ == StatusCode::kOk; }
    StatusCode code() const { return code_; }
    const std::string& message() const { return message_; }

    std::string ToString() const {
        if (ok()) return "OK";
        std::ostringstream oss;
        oss << "Status(" << static_cast<int>(code_) << ", \"" << message_ << "\")";
        return oss.str();
    }

    static Status Ok() { return Status(); }
    static Status InvalidArgument(std::string msg) { return Status(StatusCode::kInvalidArgument, std::move(msg)); }
    static Status NotFound(std::string msg) { return Status(StatusCode::kNotFound, std::move(msg)); }
    static Status Unimplemented(std::string msg) { return Status(StatusCode::kUnimplemented, std::move(msg)); }
    static Status InternalError(std::string msg) { return Status(StatusCode::kInternalError, std::move(msg)); }
    static Status CudaError(std::string msg) { return Status(StatusCode::kCudaError, std::move(msg)); }

private:
    StatusCode code_;
    std::string message_;
};

// ============================================================================
// Shape
// ============================================================================
struct Shape {
    std::vector<int64_t> dims;

    Shape() = default;
    explicit Shape(std::vector<int64_t> d) : dims(std::move(d)) {}

    int64_t NumElements() const {
        if (dims.empty()) return 1;
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }

    int64_t operator[](size_t i) const { return dims[i]; }
    int64_t& operator[](size_t i) { return dims[i]; }
    size_t size() const { return dims.size(); }
    bool empty() const { return dims.empty(); }
};

// ============================================================================
// Device
// ============================================================================
enum class DeviceType { kCPU, kCUDA };

struct Device {
    DeviceType type = DeviceType::kCPU;
    int device_id = 0;

    Device() = default;
    Device(DeviceType t, int id = 0) : type(t), device_id(id) {}

    static Device CPU() { return Device(DeviceType::kCPU, 0); }
    static Device CUDA(int id = 0) { return Device(DeviceType::kCUDA, id); }

    std::string ToString() const {
        if (type == DeviceType::kCPU) return "CPU";
        return "CUDA:" + std::to_string(device_id);
    }

    bool operator==(const Device& other) const { return type == other.type && device_id == other.device_id; }
    bool operator!=(const Device& other) const { return !(*this == other); }
    bool is_cuda() const { return type == DeviceType::kCUDA; }
    bool is_cpu() const { return type == DeviceType::kCPU; }
};

// ============================================================================
// DataType
// ============================================================================
enum class DataType {
    kFloat32, kFloat16, kBFloat16, kInt32, kInt64, kInt16, kInt8, kUInt8, kBool,
};

inline size_t GetDataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::kFloat32: return 4;
        case DataType::kInt32:   return 4;
        case DataType::kFloat16:
        case DataType::kBFloat16:
        case DataType::kInt16:   return 2;
        case DataType::kInt64:   return 8;
        case DataType::kInt8:
        case DataType::kUInt8:
        case DataType::kBool:    return 1;
        default: return 4;
    }
}

// ============================================================================
// Tensor — device-agnostic contiguous buffer
// ============================================================================
class Tensor {
public:
    Tensor()
        : dtype_(DataType::kFloat32), device_(Device::CPU()),
          data_(nullptr), size_bytes_(0), owns_data_(false) {}
    ~Tensor() { Free(); }

    Tensor(const Shape& shape, DataType dtype, const Device& device)
        : shape_(shape), dtype_(dtype), device_(device), owns_data_(true) {
        size_bytes_ = static_cast<size_t>(shape_.NumElements()) * GetDataTypeSize(dtype_);
        Allocate();
    }

    // Copy — deep copy.
    Tensor(const Tensor& other)
        : shape_(other.shape_), dtype_(other.dtype_), device_(other.device_),
          size_bytes_(other.size_bytes_), owns_data_(true) {
        Allocate();
        CopyFromInternal(other);
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            Free();
            shape_ = other.shape_;
            dtype_ = other.dtype_;
            device_ = other.device_;
            size_bytes_ = other.size_bytes_;
            owns_data_ = true;
            Allocate();
            CopyFromInternal(other);
        }
        return *this;
    }

    Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_)), dtype_(other.dtype_), device_(other.device_),
          data_(other.data_), size_bytes_(other.size_bytes_), owns_data_(other.owns_data_) {
        other.data_ = nullptr;
        other.owns_data_ = false;
        other.size_bytes_ = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            Free();
            shape_ = std::move(other.shape_);
            dtype_ = other.dtype_;
            device_ = other.device_;
            data_ = other.data_;
            size_bytes_ = other.size_bytes_;
            owns_data_ = other.owns_data_;
            other.data_ = nullptr;
            other.owns_data_ = false;
            other.size_bytes_ = 0;
        }
        return *this;
    }

    // ---- Raw access (existing API) ----
    void* data() { return data_; }
    const void* data() const { return data_; }

    // ---- Typed access ----
    template <typename T>
    T* data() { return static_cast<T*>(data_); }
    template <typename T>
    const T* data() const { return static_cast<const T*>(data_); }
    template <typename T>
    T* mutable_data() { return static_cast<T*>(data_); }

    // ---- Metadata ----
    const Shape& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    Device device() const { return device_; }
    size_t size_bytes() const { return size_bytes_; }
    int64_t numel() const { return shape_.NumElements(); }
    int64_t dim(size_t i) const { return shape_[i]; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    bool is_cuda() const { return device_.is_cuda(); }
    bool is_cpu() const { return device_.is_cpu(); }
    bool owns_data() const { return owns_data_; }

    // ---- Clone (deep copy to the same device) ----
    Tensor Clone() const { return Tensor(*this); }

    // ---- Device transfer ----
    Status ToDevice(const Device& target) {
        if (device_ == target) return Status::Ok();
#ifdef TURBORL_HAS_CUDA
        if (target.is_cuda()) {
            void* gpu = nullptr;
            if (cudaMalloc(&gpu, size_bytes_) != cudaSuccess)
                return Status::CudaError("CUDA malloc failed in ToDevice");
            if (data_)
                cudaMemcpy(gpu, data_, size_bytes_, cudaMemcpyHostToDevice);
            Free();
            data_ = gpu;
        } else {
            if (!data_) {
                data_ = std::malloc(size_bytes_);
            } else {
                // data_ currently lives on GPU; copy down first.
                void* cpu = std::malloc(size_bytes_);
                cudaMemcpy(cpu, data_, size_bytes_, cudaMemcpyDeviceToHost);
                cudaFree(data_);
                data_ = cpu;
            }
        }
        device_ = target;
        owns_data_ = true;
#else
        (void)target;
        return Status::Unimplemented("CUDA support not compiled in");
#endif
        return Status::Ok();
    }

    Status Synchronize() const {
#ifdef TURBORL_HAS_CUDA
        if (device_.is_cuda()) cudaDeviceSynchronize();
#endif
        return Status::Ok();
    }

private:
    void Allocate() {
#ifdef TURBORL_HAS_CUDA
        if (device_.is_cuda()) {
            if (cudaMalloc(&data_, size_bytes_) != cudaSuccess)
                throw std::runtime_error("CUDA allocation failed");
        } else {
            data_ = std::malloc(size_bytes_);
        }
#else
        data_ = std::malloc(size_bytes_);
#endif
    }

    void Free() {
        if (!owns_data_ || !data_) return;
#ifdef TURBORL_HAS_CUDA
        if (device_.is_cuda()) cudaFree(data_);
        else
#endif
            std::free(data_);
        data_ = nullptr;
        owns_data_ = false;
    }

    void CopyFromInternal(const Tensor& other) {
        if (size_bytes_ == 0) return;
#ifdef TURBORL_HAS_CUDA
        if (device_.is_cuda() && other.device_.is_cuda())
            cudaMemcpy(data_, other.data_, size_bytes_, cudaMemcpyDeviceToDevice);
        else if (device_.is_cuda())
            cudaMemcpy(data_, other.data_, size_bytes_, cudaMemcpyHostToDevice);
        else if (other.device_.is_cuda())
            cudaMemcpy(data_, other.data_, size_bytes_, cudaMemcpyDeviceToHost);
        else
#endif
            std::memcpy(data_, other.data_, size_bytes_);
    }

    Shape shape_;
    DataType dtype_;
    Device device_;
    void* data_ = nullptr;
    size_t size_bytes_ = 0;
    bool owns_data_ = true;
};

// ============================================================================
// CUDA memory helpers
// ============================================================================
#ifdef TURBORL_HAS_CUDA
inline void* AllocateCUDA(size_t size, int device_id = 0) {
    void* ptr = nullptr;
    cudaSetDevice(device_id);
    cudaMalloc(&ptr, size);
    return ptr;
}

inline void FreeCUDA(void* ptr) { if (ptr) cudaFree(ptr); }

inline bool CheckCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

#define TURBORL_CUDA_CHECK(expr) ::turborl::CheckCudaError((expr), __FILE__, __LINE__)

inline void CudaSynchronize() { cudaDeviceSynchronize(); }
inline int GetCudaDeviceCount() { int count = 0; cudaGetDeviceCount(&count); return count; }
inline void SetCudaDevice(int device_id) { cudaSetDevice(device_id); }

struct CudaMemoryInfo {
    size_t total = 0;
    size_t free = 0;
    size_t used() const { return total - free; }
    double utilization() const { return total > 0 ? static_cast<double>(used()) / total : 0.0; }
};

inline CudaMemoryInfo GetCudaMemoryInfo(int device_id = 0) {
    CudaMemoryInfo info;
    cudaSetDevice(device_id);
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        info.free = free_mem;
        info.total = total_mem;
    }
    return info;
}
#endif

} // namespace turborl
