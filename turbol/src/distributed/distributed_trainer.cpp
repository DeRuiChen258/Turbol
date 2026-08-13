#include "turbol/distributed/distributed_trainer.hpp"

namespace turborl {
namespace distributed {

DistributedTrainer::DistributedTrainer(const DistributedConfig& config)
    : config_(config), initialized_(false) {}

DistributedTrainer::~DistributedTrainer() {
    if (initialized_) Shutdown();
}

Status DistributedTrainer::Initialize() {
    initialized_ = true;
    return Status::Ok();
}

Status DistributedTrainer::Shutdown() {
    initialized_ = false;
    return Status::Ok();
}

Status DistributedTrainer::AllReduce(Tensor* tensor, const std::string& op) {
    return Status::Ok();
}

} // namespace distributed
} // namespace turbol
