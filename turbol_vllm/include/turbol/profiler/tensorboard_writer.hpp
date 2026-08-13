#pragma once
#include "../common.hpp"

#include <string>
#include <vector>
#include <unordered_map>

namespace turborl {
namespace profiler {

// ============================================================================
// TensorBoardWriter — writes scalar / histogram / text summaries
//                    to an Event file (tensorboard --logdir).
//
//   Depends only on the standard library; no external TB library needed.
//   Writes the binary protobuf record format directly.
//
//   Usage:
//
//     TensorBoardWriter writer("/tmp/turbol_runs/run1");
//     writer.AddScalar("train/loss", 0.42f, step=100);
//     writer.AddScalar("train/lr", 1e-4f, step=100);
//     writer.Flush();
//
//   Then:  tensorboard --logdir=/tmp/turbol_runs
// ============================================================================

class TensorBoardWriter {
public:
    explicit TensorBoardWriter(const std::string& log_dir);
    ~TensorBoardWriter();

    // Scalars
    void AddScalar(const std::string& tag, float value, int step);
    void AddScalar(const std::string& tag, double value, int step);

    // Histogram of raw values
    void AddHistogram(const std::string& tag,
                      const float* values, int count, int step);

    // Text / markdown (displayed in Text tab)
    void AddText(const std::string& tag, const std::string& text, int step);

    // Flush all pending writes
    void Flush();

    int64_t TotalEventsWritten() const { return total_written_; }

private:
    std::string log_dir_;
    std::string event_path_;
    void* file_ = nullptr;   // FILE*
    int64_t step_offset_ = 0;
    int64_t total_written_ = 0;

    // Inline RecordWriter — no external protobuf dependency.
    void WriteRecord(const void* data, size_t size);

    // Encode a tensorboard Event proto (simplified — only what we need)
    struct EventProto;
    void WriteEvent(const EventProto& ev);

    struct ScalarEvent {
        std::string tag;
        float       value;
        int64_t     step;
        int64_t     wall_time_us;
    };
    std::vector<ScalarEvent> pending_scalars_;
    int64_t wall_time_us() const;
};

// ============================================================================
// PrometheusWriter — writes metrics in Prometheus text exposition format.
//
//   Usage:
//
//     PrometheusWriter writer("/tmp/metrics.prom");
//     writer.Write("turborl_env_step_duration_ms", 0.42, {{"device","cuda:0"}});
//     writer.Flush();
//
//   Then:  prometheus_scrape("file:///tmp/metrics.prom")
// ============================================================================

struct Labels {
    std::unordered_map<std::string, std::string> kv;
    std::string ToString() const;
};

class PrometheusWriter {
public:
    explicit PrometheusWriter(const std::string& path);
    ~PrometheusWriter();

    void Write(const std::string& name, double value, const Labels& labels = {});
    void WriteCounter(const std::string& name, double value, const Labels& labels = {});
    void WriteGauge(const std::string& name, double value, const Labels& labels = {});
    void WriteHistogram(const std::string& name, double value,
                        const std::vector<double>& buckets,
                        const Labels& labels = {});
    void Flush();

private:
    std::string path_;
    std::vector<std::string> lines_;
    int64_t wall_time_us() const;
};

} // namespace profiler
} // namespace turborl
