#pragma once
#include "../common.hpp"

#ifdef TURBORL_NCCL_ENABLED
#include <nccl.h>
#endif

namespace turborl {
namespace distributed {

class NcclCommunicator {
public:
    static NcclCommunicator& Instance();
    
    Status Initialize(int rank, int world_size, int device_id = 0);
    Status Shutdown();
    
    Status AllReduce(const void* send_data, void* recv_data, size_t count, bool is_mean = false);
    Status AllGather(const void* send_data, void* recv_data, size_t count);
    Status Broadcast(const void* send_data, void* recv_data, size_t count, int root = 0);
    Status Barrier();
    
    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    bool is_initialized() const { return initialized_; }

private:
    NcclCommunicator() : rank_(0), world_size_(0), initialized_(false) {}
    ~NcclCommunicator();
    
    NcclCommunicator(const NcclCommunicator&) = delete;
    NcclCommunicator& operator=(const NcclCommunicator&) = delete;
    
    int rank_;
    int world_size_;
    bool initialized_;
    
#ifdef TURBORL_NCCL_ENABLED
    ncclComm_t nccl_comm_ = nullptr;
#endif
};

} // namespace distributed
} // namespace turbol
