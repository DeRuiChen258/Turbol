#pragma once
#include "../common.hpp"

namespace turborl {
namespace distributed {

// ============================================================================
// PipelineTrainer — pipeline parallelism for LLM RLHF training.
//
//   Splits the model across N GPUs (pipeline stages), using micro-batches
//   and async communication to overlap compute with communication.
//
//   Reference: GPipe (Huang et al., 2019) + PipeDream (Harlap et al., 2018).
//
//   Each pipeline stage owns a contiguous set of model layers and communicates
//   activations (forward) / gradients (backward) with its neighbors via
//   CUDA streams and NCCL point-to-point sends.
//
//   Configuration:
//     num_stages       : number of pipeline stages (one per GPU in PipelineParallel)
//     num_microbatches : micro-batches per gradient accumulation step
//     stage_id         : which stage this rank owns
//     num_warmup_microbatches : async warmup bubbles before pipeline fill
//
//   Example (4 GPUs, 8 microbatches, 3 layers each):
//     GPU 0: layers  0-8   (stage 0, first  75%)
//     GPU 1: layers  9-17  (stage 1)
//     GPU 2: layers 18-26  (stage 2)
//     GPU 3: layers 27-35  (stage 3, last  25%)
// ============================================================================

struct PipelineConfig {
    // Parallelism
    int  num_pipeline_stages    = 1;     // number of pipeline stages
    int  num_microbatches       = 8;      // micro-batches per optimizer step
    int  stage_id               = 0;     // which stage this rank owns
    int  num_warmup_microbatches = 4;    // pipeline fill depth

    // Micro-batch sizes
    int  microbatch_seq_len  = 512;
    int  microbatch_batch_sz = 1;

    // Memory
    int  max_cache_seq_len   = 8192;     // KV cache max sequence length

    // Checkpointing
    bool activation_checkpointing = true;  // recompute activations in backward
    int  checkpoint_every_n_layers = 2;

    // Device
    Device device = Device::CUDA(0);
};

struct MicroBatch {
    Tensor input_ids;     // [batch, seq_len]
    Tensor labels;       // [batch, seq_len] (-100 = ignore)
    Tensor attention_mask;
    Tensor output;       // filled during forward pass
    Tensor loss;         // filled after loss computation
    bool   forward_done  = false;
    bool   backward_done = false;
};

class PipelineTrainer {
public:
    explicit PipelineTrainer(const PipelineConfig& config);
    ~PipelineTrainer();

    Status Initialize();
    Status Shutdown();

    // Run one full training step (all microbatches, forward + backward + optimizer)
    // Returns the total loss averaged over all microbatches this stage processed.
    Status TrainingStep(float* out_avg_loss);

    // Standalone forward pass for inference / evaluation
    Status Forward(const Tensor& input_ids, const Tensor& labels,
                   Tensor* logits, float* loss);

    // For inter-stage communication
    Status SendForward(Tensor* tensor, int dst_stage);
    Status RecvForward(Tensor* tensor, int src_stage);
    Status SendBackward(Tensor* tensor, int dst_stage);
    Status RecvBackward(Tensor* tensor, int src_stage);

    const PipelineConfig& config() const { return config_; }
    int CurrentStep() const { return current_step_; }

private:
    PipelineConfig config_;
    bool           initialized_ = false;
    int            current_step_ = 0;

    // Micro-batch pool
    std::vector<MicroBatch> microbatches_;

    // CUDA streams for overlap
#ifdef TURBORL_HAS_CUDA
    cudaStream_t forward_stream_;
    cudaStream_t backward_stream_;
    cudaStream_t comm_stream_;
#endif

    // Peer access (for direct GPU-to-GPU P2P)
    bool peer_accessible_[8] = {false};  // up to 8 GPUs

    // Helpers
    Status ScheduleForward();
    Status ScheduleBackward();
    Status RunOptimizerStep();
    Status WaitAll();
};

} // namespace distributed
} // namespace turbol
