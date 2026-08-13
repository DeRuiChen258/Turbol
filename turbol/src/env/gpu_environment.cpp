#include "turbol/env/gpu_environment.hpp"

namespace turborl {
namespace env {

GPUEnvironment::GPUEnvironment() {}
GPUEnvironment::~GPUEnvironment() {}

Status GPUEnvironment::Reset() {
    return Status::Ok();
}

Status GPUEnvironment::Step(const Tensor& action, Tensor* observation, Tensor* reward, Tensor* done) {
    return Status::Ok();
}

} // namespace env
} // namespace turbol
