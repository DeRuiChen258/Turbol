#include "turbol/distributed/distributed_trainer.hpp"
#include "turbol/distributed/nccl_communicator.hpp"

namespace turborl {
namespace distributed {

DistributedTrainer::DistributedTrainer(const DistributedConfig& config)
    : config_(config), initialized_(false) {}

DistributedTrainer::~DistributedTrainer() {
    if (initialized_) Shutdown();
}

Status DistributedTrainer::Initialize() {
    if (config_.use_nccl) {
        auto status = NcclCommunicator::Instance().Initialize(
            config_.rank, config_.world_size, /*device_id=*/0);
        if (!status.ok()) return status;
    }
    initialized_ = true;
    return Status::Ok();
}

Status DistributedTrainer::Shutdown() {
    if (config_.use_nccl) {
        auto status = NcclCommunicator::Instance().Shutdown();
        if (!status.ok()) return status;
    }
    initialized_ = false;
    return Status::Ok();
}

Status DistributedTrainer::AllReduce(Tensor* tensor, const std::string& op) {
    if (!tensor) return Status::InvalidArgument("tensor is null");
    if (tensor->numel() == 0) return Status::Ok();
    if (!initialized_ && config_.use_nccl)
        return Status::InvalidArgument("DistributedTrainer not initialized");

    if (config_.world_size <= 1) return Status::Ok();  // single rank: no-op
    if (!config_.use_nccl)
        return Status::Unimplemented("AllReduce requires NCCL backend");

    if (tensor->dtype() != DataType::kFloat32)
        return Status::Unimplemented("AllReduce only supports Float32 tensors");

    const size_t count = static_cast<size_t>(tensor->numel());
    const bool is_mean = (op == "mean");
    // In-place all-reduce (send == recv) is supported by NCCL.
    return NcclCommunicator::Instance().AllReduce(
        tensor->data(), tensor->data(), count, is_mean);
}

Status DistributedTrainer::Broadcast(Tensor* tensor, int root_rank) {
    if (!tensor) return Status::InvalidArgument("tensor is null");
    if (tensor->numel() == 0) return Status::Ok();
    if (!initialized_ && config_.use_nccl)
        return Status::InvalidArgument("DistributedTrainer not initialized");

    if (config_.world_size <= 1) return Status::Ok();
    if (!config_.use_nccl)
        return Status::Unimplemented("Broadcast requires NCCL backend");

    if (tensor->dtype() != DataType::kFloat32)
        return Status::Unimplemented("Broadcast only supports Float32 tensors");

    const size_t count = static_cast<size_t>(tensor->numel());
    return NcclCommunicator::Instance().Broadcast(
        tensor->data(), tensor->data(), count, root_rank);
}

} // namespace distributed
} // namespace turborl
