#include "turbol/distributed/pipeline_trainer.hpp"

#ifdef TURBORL_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace turborl {
namespace distributed {

PipelineTrainer::PipelineTrainer(const PipelineConfig& config)
    : config_(config) {}

PipelineTrainer::~PipelineTrainer() {
    Shutdown();
}

Status PipelineTrainer::Initialize() {
    if (initialized_) return Status::Ok();

    // Allocate micro-batch pool
    microbatches_.resize(config_.num_microbatches);

    // Create CUDA streams
#ifdef TURBORL_HAS_CUDA
    cudaSetDevice(config_.device.device_id);
    cudaStreamCreate(&forward_stream_);
    cudaStreamCreate(&backward_stream_);
    cudaStreamCreate(&comm_stream_);

    // Check peer access
    int can_access = 0;
    for (int i = 0; i < config_.num_pipeline_stages; ++i) {
        if (i == config_.stage_id) continue;
        cudaDeviceCanAccessPeer(&can_access, config_.device.device_id, i);
        peer_accessible_[i] = (can_access == 1);
    }
#else
    return Status::Unimplemented("CUDA required for PipelineTrainer");
#endif

    initialized_ = true;
    return Status::Ok();
}

Status PipelineTrainer::Shutdown() {
#ifdef TURBORL_HAS_CUDA
    if (forward_stream_)  { cudaStreamDestroy(forward_stream_);  forward_stream_  = 0; }
    if (backward_stream_) { cudaStreamDestroy(backward_stream_); backward_stream_ = 0; }
    if (comm_stream_)     { cudaStreamDestroy(comm_stream_);     comm_stream_     = 0; }
#endif
    initialized_ = false;
    return Status::Ok();
}

Status PipelineTrainer::TrainingStep(float* out_avg_loss) {
    if (!initialized_) return Status::InvalidArgument("PipelineTrainer not initialized");
    if (!out_avg_loss) return Status::InvalidArgument("out_avg_loss is null");

    float total_loss = 0.0f;
    int processed = 0;

    // Pipeline schedule:
    // 1. Warmup: first num_warmup_microbatches run forward asynchronously
    // 2. Steady: alternate forward/backward as microbatches finish
    // 3. Cooldown: drain remaining backward passes

    // Simplified: run all microbatches serially (real impl uses async streams)
    for (int m = 0; m < config_.num_microbatches; ++m) {
        MicroBatch& mb = microbatches_[m];

        // Allocate micro-batch tensors
        mb.input_ids    = Tensor(Shape({config_.microbatch_batch_sz,
                                        config_.microbatch_seq_len}),
                                 DataType::kFloat32, config_.device);
        mb.labels       = Tensor(Shape({config_.microbatch_batch_sz,
                                        config_.microbatch_seq_len}),
                                 DataType::kInt32, config_.device);
        mb.attention_mask = Tensor(Shape({config_.microbatch_batch_sz,
                                          config_.microbatch_seq_len}),
                                   DataType::kInt32, config_.device);

        float step_loss = 0.0f;
        auto s = Forward(mb.input_ids, mb.labels, nullptr, &step_loss);
        if (!s.ok()) return s;

        total_loss += step_loss;
        ++processed;
    }

    *out_avg_loss = processed > 0 ? (total_loss / processed) : 0.0f;
    ++current_step_;
    return Status::Ok();
}

Status PipelineTrainer::Forward(const Tensor& input_ids,
                                const Tensor& labels,
                                Tensor* /*logits*/,
                                float* loss) {
    // Stub: real impl would execute model forward pass
    // For now: return a dummy loss (random scalar)
    if (loss) {
#ifdef TURBORL_HAS_CUDA
        float dummy = 1.0f;
        if (config_.device.is_cuda()) {
            // Dummy compute to keep the GPU busy
        }
        (void)dummy;
#endif
        *loss = 0.0f;
    }
    (void)input_ids; (void)labels;
    return Status::Ok();
}

Status PipelineTrainer::SendForward(Tensor* tensor, int dst_stage) {
#ifdef TURBORL_HAS_CUDA
    if (!peer_accessible_[dst_stage]) {
        // Fall back to CPU copy via host
        Tensor host = tensor->Clone();
        host.ToDevice(Device::CPU());
        tensor->ToDevice(config_.device);
        (void)host;
    }
    // P2P send via CUDA memcpy peer
    cudaSetDevice(config_.device.device_id);
    // cudaMemcpyPeerAsync would be used here in a real implementation
#endif
    (void)tensor; (void)dst_stage;
    return Status::Ok();
}

Status PipelineTrainer::RecvForward(Tensor* tensor, int src_stage) {
    (void)tensor; (void)src_stage;
    return Status::Ok();
}

Status PipelineTrainer::SendBackward(Tensor* tensor, int dst_stage) {
    (void)tensor; (void)dst_stage;
    return Status::Ok();
}

Status PipelineTrainer::RecvBackward(Tensor* tensor, int src_stage) {
    (void)tensor; (void)src_stage;
    return Status::Ok();
}

Status PipelineTrainer::WaitAll() {
#ifdef TURBORL_HAS_CUDA
    if (config_.device.is_cuda()) {
        cudaStreamSynchronize(forward_stream_);
        cudaStreamSynchronize(backward_stream_);
        cudaStreamSynchronize(comm_stream_);
    }
#endif
    return Status::Ok();
}

} // namespace distributed
} // namespace turbol
