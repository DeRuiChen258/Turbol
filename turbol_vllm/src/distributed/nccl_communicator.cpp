#include "turbol/distributed/nccl_communicator.hpp"

#include <fstream>

namespace turborl {
namespace distributed {

namespace {

// Bootstrap NCCL without an external MPI runtime: rank 0 writes the unique id
// to a shared file, remaining ranks poll until it appears and read it back.
// Sufficient for single-node multi-process launches; for multi-node setups a
// real process launcher (MPI/torchrun) should exchange the id instead.
std::string IdFilePath(int master_port) {
    return "/tmp/turborl_nccl_id_" + std::to_string(master_port) + ".bin";
}

} // namespace

NcclCommunicator& NcclCommunicator::Instance() {
    static NcclCommunicator instance;
    return instance;
}

NcclCommunicator::~NcclCommunicator() {
    Shutdown();
}

Status NcclCommunicator::Initialize(int rank, int world_size, int device_id) {
    if (initialized_) return Status::Ok();
    if (world_size <= 1) {
        // Degenerate single-rank group: still mark initialized so collectives
        // are no-ops rather than hard errors.
        rank_ = 0;
        world_size_ = 1;
        initialized_ = true;
        return Status::Ok();
    }

    rank_ = rank;
    world_size_ = world_size;

#ifdef TURBORL_NCCL_ENABLED
    cudaSetDevice(device_id);

    ncclUniqueId id;
    if (rank_ == 0) {
        if (ncclGetUniqueId(&id) != ncclSuccess)
            return Status::InternalError("ncclGetUniqueId failed");
        std::ofstream out(IdFilePath(29500), std::ios::binary | std::ios::trunc);
        if (!out.is_open())
            return Status::InternalError("cannot write NCCL id file");
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
        out.close();
    } else {
        // Poll for the id file written by rank 0.
        std::ifstream in;
        for (int attempt = 0; attempt < 500; ++attempt) {
            in.open(IdFilePath(29500), std::ios::binary);
            if (in.is_open()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!in.is_open())
            return Status::InternalError("timeout waiting for NCCL id file");
        in.read(reinterpret_cast<char*>(&id), sizeof(id));
        in.close();
    }

    if (ncclCommInitRank(&nccl_comm_, world_size_, id, rank_) != ncclSuccess)
        return Status::InternalError("ncclCommInitRank failed");
#endif

    initialized_ = true;
    return Status::Ok();
}

Status NcclCommunicator::Shutdown() {
    if (!initialized_) return Status::Ok();
#ifdef TURBORL_NCCL_ENABLED
    if (world_size_ > 1 && nccl_comm_ != nullptr) {
        ncclCommDestroy(nccl_comm_);
        nccl_comm_ = nullptr;
    }
#endif
    initialized_ = false;
    return Status::Ok();
}

Status NcclCommunicator::AllReduce(const void* send_data, void* recv_data,
                                   size_t count, bool is_mean) {
    if (!initialized_) return Status::InvalidArgument("NcclCommunicator not initialized");
    if (world_size_ <= 1) {
        // Single rank: result == input.
        std::memcpy(recv_data, send_data, count * sizeof(float));
        return Status::Ok();
    }
#ifdef TURBORL_NCCL_ENABLED
    if (ncclAllReduce(send_data, recv_data, count, ncclFloat, ncclSum,
                      nccl_comm_, nullptr) != ncclSuccess)
        return Status::InternalError("ncclAllReduce failed");
    if (is_mean) {
        // Scale by 1/world_size via the CUDA scale kernel (best effort).
        const float inv = 1.0f / static_cast<float>(world_size_);
        for (size_t i = 0; i < count; ++i)
            static_cast<float*>(recv_data)[i] *= inv;
    }
    return Status::Ok();
#else
    (void)send_data; (void)recv_data; (void)count; (void)is_mean;
    return Status::Unimplemented("NCCL not enabled");
#endif
}

Status NcclCommunicator::AllGather(const void* send_data, void* recv_data, size_t count) {
    if (!initialized_) return Status::InvalidArgument("NcclCommunicator not initialized");
    if (world_size_ <= 1) {
        std::memcpy(recv_data, send_data, count * sizeof(float));
        return Status::Ok();
    }
#ifdef TURBORL_NCCL_ENABLED
    if (ncclAllGather(send_data, recv_data, count, ncclFloat, nccl_comm_, nullptr) != ncclSuccess)
        return Status::InternalError("ncclAllGather failed");
    return Status::Ok();
#else
    (void)send_data; (void)recv_data; (void)count;
    return Status::Unimplemented("NCCL not enabled");
#endif
}

Status NcclCommunicator::Broadcast(const void* send_data, void* recv_data,
                                   size_t count, int root) {
    if (!initialized_) return Status::InvalidArgument("NcclCommunicator not initialized");
    if (world_size_ <= 1) {
        if (send_data && recv_data && send_data != recv_data)
            std::memcpy(recv_data, send_data, count * sizeof(float));
        return Status::Ok();
    }
#ifdef TURBORL_NCCL_ENABLED
    if (ncclBcast(recv_data, count, ncclFloat, root, nccl_comm_, nullptr) != ncclSuccess)
        return Status::InternalError("ncclBcast failed");
    return Status::Ok();
#else
    (void)send_data; (void)recv_data; (void)count; (void)root;
    return Status::Unimplemented("NCCL not enabled");
#endif
}

Status NcclCommunicator::Barrier() {
    if (!initialized_) return Status::InvalidArgument("NcclCommunicator not initialized");
    if (world_size_ <= 1) return Status::Ok();
#ifdef TURBORL_NCCL_ENABLED
    if (ncclCommBarrier(nccl_comm_) != ncclSuccess)
        return Status::InternalError("ncclCommBarrier failed");
    return Status::Ok();
#else
    return Status::Unimplemented("NCCL not enabled");
#endif
}

} // namespace distributed
} // namespace turborl
