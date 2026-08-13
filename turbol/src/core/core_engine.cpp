#include "turbol/core/core_engine.hpp"

namespace turborl {

CoreEngine::CoreEngine() {}
CoreEngine::~CoreEngine() {}

Status CoreEngine::Initialize() {
    return Status::Ok();
}

Status CoreEngine::Shutdown() {
    return Status::Ok();
}

Status CoreEngine::Synchronize() {
#ifdef TURBORL_CUDA
    CudaSynchronize();
#endif
    return Status::Ok();
}

CoreEngine& CoreEngine::Instance() {
    static CoreEngine instance;
    return instance;
}

} // namespace turbol
