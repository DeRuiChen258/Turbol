#pragma once
#include "../common.hpp"

namespace turborl {
namespace distributed {

enum class ParallelStrategy { kDataParallel, kModelParallel, kPipelineParallel };

struct DistributedConfig {
    ParallelStrategy strategy = ParallelStrategy::kDataParallel;
    int world_size = 1;
    int rank = 0;
    std::string master_addr = "localhost";
    int master_port = 29500;
    bool use_nccl = false;
};

class DistributedTrainer {
public:
    DistributedTrainer(const DistributedConfig& config);
    ~DistributedTrainer();
    
    Status Initialize();
    Status Shutdown();
    Status AllReduce(Tensor* tensor, const std::string& op = "sum");
    Status Broadcast(Tensor* tensor, int root_rank = 0);
    
    int GetRank() const { return config_.rank; }
    int GetWorldSize() const { return config_.world_size; }

private:
    DistributedConfig config_;
    bool initialized_;
};

} // namespace distributed
} // namespace turbol
