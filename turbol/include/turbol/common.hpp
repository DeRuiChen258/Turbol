#pragma once

// ============================================================================
// Standard Library Includes
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

// ============================================================================
// Platform Detection
// ============================================================================
#if defined(__linux__)
    #define TURBORL_PLATFORM_LINUX 1
#elif defined(_WIN32)
    #define TURBORL_PLATFORM_WINDOWS 1
#elif defined(__APPLE__)
    #define TURBORL_PLATFORM_MACOS 1
#endif

// Compiler Detection
#if defined(__GNUC__)
    #define TURBORL_COMPILER_GCC 1
#elif defined(__clang__)
    #define TURBORL_COMPILER_CLANG 1
#elif defined(_MSC_VER)
    #define TURBORL_COMPILER_MSVC 1
#endif

// C++ Standard Version
#if defined(__cplusplus) && __cplusplus >= 201703L
    #define TURBORL_CXX17 1
#endif

// CUDA Detection
#if defined(__CUDACC__) || defined(__NVCC__)
    #define TURBORL_CUDA 1
    #include <cuda_runtime.h>
#endif

// Export Macros
#if defined(TURBORL_EXPORTS)
    #define TURBORL_API __declspec(dllexport)
#else
    #define TURBORL_API __declspec(dllimport)
#endif

#define TURBORL_VERSION_MAJOR 1
#define TURBORL_VERSION_MINOR 0
#define TURBORL_VERSION_PATCH 0

namespace turborl {

// ============================================================================
// Status and Error Handling
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
};

class Status {
public:
    Status() : code_(StatusCode::kOk), message_() {}
    Status(StatusCode code, const std::string& message) : code_(code), message_(message) {}
    
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
    static Status InvalidArgument(const std::string& msg) { return Status(StatusCode::kInvalidArgument, msg); }
    static Status NotFound(const std::string& msg) { return Status(StatusCode::kNotFound, msg); }
    static Status Unimplemented(const std::string& msg) { return Status(StatusCode::kUnimplemented, msg); }
    static Status InternalError(const std::string& msg) { return Status(StatusCode::kInternalError, msg); }

private:
    StatusCode code_;
    std::string message_;
};

// ============================================================================
// Shape and Dimensions
// ============================================================================
struct Shape {
    std::vector<int64_t> dims;
    
    Shape() = default;
    explicit Shape(const std::vector<int64_t>& d) : dims(d) {}
    
    int64_t NumElements() const {
        if (dims.empty()) return 1;
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
    
    int64_t operator[](size_t i) const { return dims[i]; }
    int64_t& operator[](size_t i) { return dims[i]; }
    size_t size() const { return dims.size(); }
};

// ============================================================================
// Device Types
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
// Data Types
// ============================================================================
enum class DataType {
    kFloat32, kFloat16, kBFloat16, kInt32, kInt64, kInt16, kInt8, kUInt8, kBool,
};

inline size_t GetDataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::kFloat32: return 4;
        case DataType::kFloat16: case DataType::kBFloat16: case DataType::kInt16: return 2;
        case DataType::kInt32: case DataType::kInt64: return 8;
        case DataType::kInt8: case DataType::kUInt8: case DataType::kBool: return 1;
        default: return 4;
    }
}

// ============================================================================
// Tensor (CPU and CUDA Support)
// ============================================================================
class Tensor {
public:
    Tensor() : dtype_(DataType::kFloat32), device_(Device::CPU()), 
               data_(nullptr), size_bytes_(0), owns_data_(false) 
#ifdef TURBORL_CUDA
               , cuda_data_(nullptr)
#endif
    {}
    
    Tensor(const Shape& shape, DataType dtype, const Device& device)
        : shape_(shape), dtype_(dtype), device_(device), owns_data_(true) {
        size_bytes_ = static_cast<size_t>(shape_.NumElements()) * GetDataTypeSize(dtype_);
        Allocate();
    }
    
    ~Tensor() { Free(); }
    
    Tensor(const Tensor& other)
        : shape_(other.shape_), dtype_(other.dtype_), device_(other.device_),
          size_bytes_(other.size_bytes_), owns_data_(true) {
        Allocate();
        CopyFromInternal(other);
    }
    
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            Free();
            shape_ = other.shape_; dtype_ = other.dtype_; device_ = other.device_;
            size_bytes_ = other.size_bytes_; owns_data_ = true;
            Allocate();
            CopyFromInternal(other);
        }
        return *this;
    }
    
    Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_)), dtype_(other.dtype_), device_(other.device_),
          data_(other.data_), size_bytes_(other.size_bytes_), owns_data_(other.owns_data_) {
#ifdef TURBORL_CUDA
        cuda_data_ = other.cuda_data_;
        other.cuda_data_ = nullptr;
#endif
        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    
    void* data() { 
#ifdef TURBORL_CUDA
        if (device_.is_cuda() && cuda_data_) return cuda_data_;
#endif
        return data_; 
    }
    
    const void* data() const { 
#ifdef TURBORL_CUDA
        if (device_.is_cuda() && cuda_data_) return cuda_data_;
#endif
        return data_; 
    }
    
    const Shape& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    Device device() const { return device_; }
    size_t size_bytes() const { return size_bytes_; }
    bool is_cuda() const { return device_.is_cuda(); }
    bool owns_data() const { return owns_data_; }
    
#ifdef TURBORL_CUDA
    void* cuda_data() const { return cuda_data_; }
    
    Status ToDevice(const Device& target) {
        if (device_ == target) return Status::Ok();
        if (target.is_cuda()) {
            if (!cuda_data_) {
                if (cudaMalloc(&cuda_data_, size_bytes_) != cudaSuccess)
                    return Status::InternalError("CUDA malloc failed");
            }
            if (data_) cudaMemcpy(cuda_data_, data_, size_bytes_, cudaMemcpyHostToDevice);
        } else {
            if (cuda_data_ && data_) {
                cudaMemcpy(data_, cuda_data_, size_bytes_, cudaMemcpyDeviceToHost);
                cudaFree(cuda_data_);
                cuda_data_ = nullptr;
            }
        }
        device_ = target;
        return Status::Ok();
    }
    
    Status Synchronize() const {
        if (device_.is_cuda()) cudaDeviceSynchronize();
        return Status::Ok();
    }
#endif

private:
    void Allocate() {
#ifdef TURBORL_CUDA
        if (device_.is_cuda()) {
            if (cudaMalloc(&cuda_data_, size_bytes_) != cudaSuccess)
                throw std::runtime_error("CUDA allocation failed");
            data_ = nullptr;
        } else {
            data_ = malloc(size_bytes_);
            cuda_data_ = nullptr;
        }
#else
        data_ = malloc(size_bytes_);
#endif
    }
    
    void Free() {
        if (!owns_data_) return;
#ifdef TURBORL_CUDA
        if (device_.is_cuda() && cuda_data_) { cudaFree(cuda_data_); cuda_data_ = nullptr; }
        else
#endif
        if (data_) { free(data_); data_ = nullptr; }
    }
    
    void CopyFromInternal(const Tensor& other) {
#ifdef TURBORL_CUDA
        if (device_.is_cuda() && other.device_.is_cuda())
            cudaMemcpy(cuda_data_, other.cuda_data_, size_bytes_, cudaMemcpyDeviceToDevice);
        else if (device_.is_cuda()) cudaMemcpy(cuda_data_, other.data_, size_bytes_, cudaMemcpyHostToDevice);
        else if (other.device_.is_cuda()) cudaMemcpy(data_, other.cuda_data_, size_bytes_, cudaMemcpyDeviceToHost);
        else
#endif
            memcpy(data_, other.data_, size_bytes_);
    }
    
    Shape shape_;
    DataType dtype_;
    Device device_;
    void* data_ = nullptr;
    size_t size_bytes_ = 0;
    bool owns_data_ = true;
#ifdef TURBORL_CUDA
    void* cuda_data_ = nullptr;
#endif
};

// ============================================================================
// CUDA Memory Helpers
// ============================================================================
#ifdef TURBORL_CUDA
inline void* AllocateCUDA(size_t size, int device_id = 0) {
    void* ptr = nullptr;
    cudaSetDevice(device_id);
    cudaMalloc(&ptr, size);
    return ptr;
}

inline void FreeCUDA(void* ptr) { if (ptr) cudaFree(ptr); }

inline bool CheckCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

#define CHECK_CUDA(expr) ::turborl::CheckCudaError(expr, __FILE__, __LINE__)

inline void CudaSynchronize() { cudaDeviceSynchronize(); }
inline int GetCudaDeviceCount() { int count = 0; cudaGetDeviceCount(&count); return count; }
inline void SetCudaDevice(int device_id) { cudaSetDevice(device_id); }

struct CudaMemoryInfo {
    size_t total = 0;
    size_t free = 0;
    size_t used() const { return total - free; }
    double utilization() const { return total > 0 ? (double)used() / total : 0.0; }
};

inline CudaMemoryInfo GetCudaMemoryInfo(int device_id = 0) {
    CudaMemoryInfo info;
    cudaSetDevice(device_id);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    info.free = free_mem;
    info.total = total_mem;
    return info;
}
#endif

} // namespace turborl
