#include "turbol/distributed/nccl_communicator.hpp"

#ifdef TURBORL_CUDA
#include <nccl.h>

namespace turborl {
namespace distributed {

NcclCommunicator NcclCommunicator::instance_;

NcclCommunicator& NcclCommunicator::GetInstance() {
    return instance_;
}

Status NcclCommunicator::Initialize(int rank, int world_size, int device_id) {
    if (initialized_) return Status::Ok();
    
    rank_ = rank;
    world_size_ = world_size;
    
    if (world_size == 1) {
        initialized_ = true;
        return Status::Ok();
    }
    
    cudaSetDevice(device_id);
    
    ncclUniqueId nccl_id;
    if (rank == 0) {
        if (ncclGetUniqueId(&nccl_id) != ncclSuccess) {
            return Status::InternalError("Failed to get NCCL unique ID");
        }
    }
    
    if (ncclCommInitRank(&nccl_comm_, world_size, nccl_id, rank) != ncclSuccess) {
        return Status::InternalError("Failed to initialize NCCL");
    }
    
    initialized_ = true;
    return Status::Ok();
}

Status NcclCommunicator::Shutdown() {
    if (!initialized_) return Status::Ok();
    
    if (nccl_comm_ && world_size_ > 1) {
        ncclCommDestroy(nccl_comm_);
        nccl_comm_ = nullptr;
    }
    
    initialized_ = false;
    return Status::Ok();
}

Status NcclCommunicator::AllReduce(void* data, size_t count, void* dtype_ptr,
                                  ReduceOp op, cudaStream_t stream) {
    if (!initialized_ || world_size_ == 1) return Status::Ok();
    
    ncclDataType_t dtype = *(static_cast<ncclDataType_t*>(dtype_ptr));
    ncclRedOp_t nccl_op = static_cast<ncclRedOp_t>(op);
    
    cudaSetDevice(0);
    ncclAllReduce(data, data, count, dtype, nccl_op, nccl_comm_, stream ? stream : 0);
    
    return Status::Ok();
}

Status NcclCommunicator::Broadcast(void* data, size_t count, int root, cudaStream_t stream) {
    if (!initialized_ || world_size_ == 1) return Status::Ok();
    
    cudaSetDevice(0);
    ncclBroadcast(data, data, count, ncclFloat32, root, nccl_comm_, stream ? stream : 0);
    
    return Status::Ok();
}

Status NcclCommunicator::Barrier() {
    if (!initialized_ || world_size_ == 1) return Status::Ok();
    
    ncclCommBarrier(nccl_comm_);
    return Status::Ok();
}

} // namespace distributed
} // namespace turborl
#endif
