#include "turbol/dataset/dataset.hpp"

#include <algorithm>
#include <random>
#include <sstream>

#define RETURN_IF_ERROR(expr) do { auto _s = (expr); if (!_s.ok()) return _s; } while(0)

namespace turborl {
namespace dataset {

Dataset::Dataset(const DatasetConfig& config) : config_(config) {}
Dataset::~Dataset() = default;

Status Dataset::Initialize() {
    if (initialized_) return Status::Ok();
    for (const auto& path : config_.data_paths) {
        RETURN_IF_ERROR(LoadFile(path));
    }
    if (shuffle_buffer_.size() < static_cast<size_t>(config_.shuffle_buffer_size))
        shuffle_buffer_.resize(config_.shuffle_buffer_size);
    initialized_ = true;
    return Status::Ok();
}

Status Dataset::Shutdown() {
    for (auto& fh : files_) fh.stream.close();
    initialized_ = false;
    return Status::Ok();
}

bool Dataset::HasNext() const {
    return current_index_ < total_samples_;
}

Status Dataset::NextBatch(PreprocessedBatch* batch) {
    if (!initialized_) return Status::InvalidArgument("Dataset not initialized");
    if (!batch) return Status::InvalidArgument("batch pointer is null");

    const int B = config_.num_workers > 0 ? config_.num_workers : 1;
    batch->input_ids     = Tensor(Shape({B, config_.max_seq_length}), DataType::kInt32,
                                  config_.prefetch_device);
    batch->attention_mask = Tensor(Shape({B, config_.max_seq_length}), DataType::kInt32,
                                    config_.prefetch_device);
    batch->labels       = Tensor(Shape({B, config_.max_seq_length}), DataType::kInt32,
                                  config_.prefetch_device);
    batch->prompt_lens  = Tensor(Shape({B}), DataType::kInt32, config_.prefetch_device);
    batch->sample_indices.resize(B);

    for (int i = 0; i < B && HasNext(); ++i) {
        size_t idx = NextShuffledIndex();
        DataSample sample;
        sample.index = idx;
        RETURN_IF_ERROR(Tokenize(sample, batch, i));
        batch->sample_indices[i] = idx;
        ++current_index_;
    }

    return Status::Ok();
}

void Dataset::Reset() {
    current_index_ = 0;
    if (config_.shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(shuffle_indices_.begin(), shuffle_indices_.end(), g);
    }
}

Status Dataset::Seek(size_t index) {
    if (index >= total_samples_) return Status::InvalidArgument("index out of range");
    current_index_ = index;
    return Status::Ok();
}

Status Dataset::LoadFile(const std::string& path) {
    FileHandle fh;
    fh.path = path;
    fh.stream.open(path, std::ios::in);
    if (!fh.stream) return Status::NotFound("Cannot open dataset file: " + path);

    // Count lines
    std::string line;
    size_t byte_pos = 0;
    while (std::getline(fh.stream, line)) {
        fh.line_offsets.push_back(byte_pos);
        byte_pos += line.size() + 1;
        ++fh.line_count;
        ++total_samples_;
    }
    fh.stream.close();
    fh.stream.open(path, std::ios::in);
    files_.push_back(std::move(fh));

    // Initialize shuffle indices
    for (size_t i = 0; i < total_samples_; ++i)
        shuffle_indices_.push_back(i);
    if (config_.shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(shuffle_indices_.begin(), shuffle_indices_.end(), g);
    }

    return Status::Ok();
}

Status Dataset::ParseLine(const std::string& line, DataSample* out) {
    // Minimal JSONL parser: {"prompt": "...", "chosen": "...", "rejected": "...", "reward": 0.0}
    // Accepts both single-line JSON and simple key=value
    if (line.empty()) return Status::InvalidArgument("Empty line");

    // Try JSON first
    size_t prompt_start = line.find("\"prompt\"");
    if (prompt_start != std::string::npos) {
        size_t q1 = line.find('"', prompt_start + 8);
        size_t q2 = line.find('"', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos)
            out->prompt = line.substr(q1 + 1, q2 - q1 - 1);
    } else {
        // Fallback: treat the whole line as the prompt
        out->prompt = line;
    }

    // chosen
    size_t ch_start = line.find("\"chosen\"");
    if (ch_start != std::string::npos) {
        size_t q1 = line.find('"', ch_start + 8);
        size_t q2 = line.find('"', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos)
            out->chosen = line.substr(q1 + 1, q2 - q1 - 1);
    }

    // rejected
    size_t rj_start = line.find("\"rejected\"");
    if (rj_start != std::string::npos) {
        size_t q1 = line.find('"', rj_start + 10);
        size_t q2 = line.find('"', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos)
            out->rejected = line.substr(q1 + 1, q2 - q1 - 1);
    }

    return Status::Ok();
}

size_t Dataset::NextShuffledIndex() {
    if (!config_.shuffle) {
        size_t idx = current_index_;
        return idx;
    }
    if (shuffle_head_ >= shuffle_indices_.size()) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(shuffle_indices_.begin(), shuffle_indices_.end(), g);
        shuffle_head_ = 0;
    }
    return shuffle_indices_[shuffle_head_++];
}

Status Dataset::Tokenize(const DataSample& /*sample*/, PreprocessedBatch* /*batch*/,
                         int /*batch_idx*/) {
    // Stub: real implementation would call HuggingFace tokenizers C++ bindings.
    // For now: fill with zeros (placeholder token ids).
    return Status::Ok();
}

} // namespace dataset
} // namespace turbol
