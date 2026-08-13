#pragma once
#include "../common.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <fstream>

#ifdef TURBORL_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace turborl {
namespace profiler {

// ============================================================================
// Profiler — low-overhead instrumentation for CPU/CUDA workloads.
//
//   Provides:
//   - CPU wall-clock / user / sys timers via std::chrono
//   - CUDA Event pairs (kernel-level GPU timing)
//   - NVTX ranges (Nsight Systems / nvvp visibility)
//   - Span tree (hierarchical call profiles)
//   - Export to JSON (Chrome Trace) and TensorBoard summary format
//
//   Usage:
//
//     Profiler prof;
//     prof.Enable();
//     {
//         auto span = prof.BeginSpan("env_step");
//         env.Step(...);
//     }
//     prof.Disable();
//     prof.ExportChromeTrace("/tmp/trace.json");
//
//   Guard RAII wrapper for automatic Begin/End:
//
//     {
//         ProfilerGuard guard(&prof, "forward_pass");
//         model.Forward(input);
//     }   // span ends here
// ============================================================================

struct SpanID { uint64_t id = 0; };
struct EventID { uint64_t id = 0; };

struct ProfilerConfig {
    bool enable_nvtx        = true;   // NVTX push/pop markers
    bool enable_cuda_timing = true;  // CUDA Event pairs
    bool enable_memory      = false;  // memory tracking
    bool enable_cpu         = true;   // wall-clock timing
    int  cuda_device_id     = 0;
    std::string export_dir  = "/tmp/turbol_profiler";
};

// Single timed event / span
struct Span {
    uint64_t       id       = 0;
    uint64_t       parent   = 0;
    std::string    name;
    std::string    category;          // "compute" | "communication" | "io"
    int            device_id = -1;     // -1 = CPU
    int64_t        start_ns = 0;
    int64_t        end_ns   = 0;
    int64_t        duration_ns() const { return end_ns - start_ns; }

    std::unordered_map<std::string, std::string> metadata;
};

// Aggregated stats for one named event
struct AggregatedStats {
    std::string name;
    int64_t  count       = 0;
    int64_t  total_ns    = 0;
    int64_t  min_ns      = INT64_MAX;
    int64_t  max_ns      = 0;
    double   avg_ns() const { return count > 0 ? static_cast<double>(total_ns) / count : 0.0; }
};

class Profiler {
public:
    explicit Profiler(const ProfilerConfig& config = {});
    ~Profiler();

    // Start / stop
    void Enable();
    void Disable();
    bool IsEnabled() const { return enabled_; }

    // CPU span — RAII guard calls Begin/End automatically
    SpanID BeginSpan(const std::string& name,
                     const std::string& category = "compute",
                     int device_id = -1);
    void   EndSpan(SpanID id);

    // RAII guard helper
    class Guard {
    public:
        Guard(Profiler* prof, const std::string& name,
              const std::string& category = "compute", int device_id = -1)
            : prof_(prof), id_(prof_->BeginSpan(name, category, device_id)) {}
        ~Guard() { prof_->EndSpan(id_); }
    private:
        Profiler* prof_;
        SpanID    id_;
    };

    // CUDA Event pair (GPU-level timing)
    EventID BeginEvent(const std::string& name, int device_id = 0);
    void    EndEvent(EventID id);

    // Named mark (zero-duration point annotation)
    void    Mark(const std::string& name);

    // Aggregate and export
    std::vector<Span>              GetAllSpans() const;
    std::vector<AggregatedStats>   GetStats() const;       // per-name aggregate
    void ExportChromeTrace(const std::string& path) const;   // JSON for chrome://tracing
    void ExportJSON(const std::string& path) const;
    void PrintSummary() const;

private:
    ProfilerConfig    config_;
    bool              enabled_ = false;
    uint64_t          next_id_ = 1;

    std::chrono::steady_clock::time_point start_time_;

    std::vector<Span>    spans_;          // flat list of completed spans
    std::vector<Span>    open_spans_;     // stack for nesting
    std::vector<uint64_t> span_stack_;     // parent chain

    struct CudaEventPair {
        std::string name;
        int         device_id;
        cudaEvent_t start;
        cudaEvent_t end;
        int64_t     start_ns = 0;
        int64_t     end_ns   = 0;
    };
    std::vector<CudaEventPair> cuda_events_;

    mutable std::mutex mutex_;
};

} // namespace profiler
} // namespace turborl
