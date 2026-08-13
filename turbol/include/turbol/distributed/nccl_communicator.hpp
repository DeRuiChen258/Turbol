#pragma once
#include "../common.hpp"

#ifdef TURBORL_CUDA
#include <cuda_runtime.h>
#include <nccl.h>

namespace turborl {
namespace distributed {

enum class ReduceOp { kSum, kProd, kMin, kMax, kAvg };

class NcclCommunicator {
public:
    static Status Initialize(int rank, int world_size, int device_id = 0);
    static Status Shutdown();
    static NcclCommunicator& GetInstance();
    
    Status AllReduce(void* data, size_t count, void* count_type,
                     ReduceOp op = ReduceOp::kSum, cudaStream_t stream = nullptr);
    Status Broadcast(void* data, size_t count, int root, cudaStream_t stream = nullptr);
    Status Barrier();
    
    int Rank() const { return rank_; }
    int WorldSize() const { return world_size_; }
    bool IsInitialized() const { return initialized_; }

private:
    NcclCommunicator() : rank_(0), world_size_(1), nccl_comm_(nullptr), initialized_(false) {}
    ~NcclCommunicator() { Shutdown(); }
    
    static NcclCommunicator instance_;
    
    int rank_;
    int world_size_;
    ncclComm_t nccl_comm_;
    bool initialized_;
};

} // namespace distributed
} // namespace turborl
#endif
