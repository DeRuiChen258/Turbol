#pragma once
#include "../common.hpp"

namespace turborl {
namespace env {

class GPUEnvironment {
public:
    GPUEnvironment();
    ~GPUEnvironment();
    
    Status Reset();
    Status Step(const Tensor& action, Tensor* observation, Tensor* reward, Tensor* done);
    
    int NumEnvs() const { return num_envs_; }
    const Shape& ObsShape() const { return obs_shape_; }
    const Shape& ActionShape() const { return action_shape_; }

private:
    int num_envs_ = 1;
    Shape obs_shape_;
    Shape action_shape_;
};

} // namespace env
} // namespace turbol
