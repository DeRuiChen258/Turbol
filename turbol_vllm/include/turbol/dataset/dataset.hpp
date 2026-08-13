#pragma once
#include "../common.hpp"

#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <fstream>

namespace turborl {
namespace dataset {

// ============================================================================
// Dataset — data loading and preprocessing for RLHF training.
//
//   Supports:
//   - JSONL / CSV prompt / preference datasets (RLHF formats)
//   - Streaming (memory-efficient — never loads the full file)
//   - GPU-prefetched batches (decoupled from CPU tokenization)
//   - Dataset mixture (weighted sampling across multiple sources)
//
//   Memory layout for a batch:
//     input_ids    : [batch_size, max_seq_len]  int32
//     attention_msk : [batch_size, max_seq_len]  int32
//     labels        : [batch_size, max_seq_len]  int32  (-100 = ignore)
//     prompt_lens   : [batch_size]               int32
// ============================================================================

struct DatasetConfig {
    // Source
    std::vector<std::string> data_paths;   // JSONL or CSV files
    std::string format = "jsonl";          // "jsonl" | "csv" | "arrow"

    // Tokenization
    std::string tokenizer_path = "";
    int  max_seq_length = 2048;
    bool truncate = true;

    // Loading
    int   num_workers = 4;              // data-loading threads
    int   prefetch_batches = 2;         // async prefetch depth
    bool  shuffle = true;
    int   shuffle_buffer_size = 10000;
    float train_split = 0.98f;          // train / val split

    // Mixture
    struct MixEntry {
        std::string name;
        float       weight = 1.0f;
        std::vector<std::string> paths;
    };
    std::vector<MixEntry> mixture;

    // Device for prefetch
    Device prefetch_device = Device::CPU();
};

struct DataSample {
    std::string prompt;
    std::string chosen;       // preferred response (RLHF / DPO)
    std::string rejected;     // dispreferred response (DPO only)
    float       reward = 0.0f; // optional pre-computed reward
    std::string source;
    int64_t     index = 0;
};

struct PreprocessedBatch {
    Tensor input_ids;       // [batch, seq_len]  int32 token ids
    Tensor attention_mask; // [batch, seq_len]  int32 (1 = valid)
    Tensor labels;         // [batch, seq_len]  int32 (-100 = ignore)
    Tensor prompt_lens;     // [batch]            int32
    std::vector<int64_t> sample_indices;
};

class Dataset {
public:
    explicit Dataset(const DatasetConfig& config);
    ~Dataset();

    // Lifecycle
    Status Initialize();
    Status Shutdown();

    // Iteration
    bool   HasNext() const;
    Status NextBatch(PreprocessedBatch* batch);
    void   Reset();

    // Accessors
    size_t     Size() const { return total_samples_; }
    DatasetConfig config() const { return config_; }

    // Seek to an exact index (for reproducible evaluation)
    Status Seek(size_t index);

private:
    DatasetConfig       config_;
    bool                initialized_ = false;

    // File handles / streams (for streaming)
    struct FileHandle {
        std::ifstream         stream;
        std::string           path;
        size_t                line_count = 0;
        std::vector<size_t>   line_offsets;  // byte offsets for fast seek
    };
    std::vector<FileHandle> files_;

    size_t      total_samples_ = 0;
    size_t      current_index_ = 0;

    // Shuffle buffer (ring buffer of sample indices)
    std::vector<size_t> shuffle_indices_;
    std::vector<size_t> shuffle_buffer_;
    size_t      shuffle_head_ = 0;

    // Internal helpers
    Status LoadFile(const std::string& path);
    Status ParseLine(const std::string& line, DataSample* out);
    size_t NextShuffledIndex();

    // Tokenizer stub (simplified — real impl would call HuggingFace tokenizers)
    Status Tokenize(const DataSample& sample, PreprocessedBatch* batch,
                    int batch_size);
};

} // namespace dataset
} // namespace turbol
