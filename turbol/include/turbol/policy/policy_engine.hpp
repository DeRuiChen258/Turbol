#pragma once
#include "../common.hpp"

namespace turborl {
namespace policy {

class PolicyEngine {
public:
    PolicyEngine();
    ~PolicyEngine();
    
    Status Initialize();
    Status Forward(const Tensor& observation, Tensor* action, Tensor* logprob);
    Status Update(const Tensor& batch);
    
private:
    bool initialized_ = false;
};

} // namespace policy
} // namespace turbol
