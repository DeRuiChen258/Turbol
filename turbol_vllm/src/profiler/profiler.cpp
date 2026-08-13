#include "turbol/profiler/profiler.hpp"

namespace turborl {
namespace profiler {

Profiler::Profiler(const ProfilerConfig& config) : config_(config) {
    start_time_ = std::chrono::steady_clock::now();
}

Profiler::~Profiler() {
    if (enabled_) Disable();
    for (auto& e : cuda_events_) {
#ifdef TURBORL_HAS_CUDA
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(e.device_id);
        if (e.start) cudaEventDestroy(e.start);
        if (e.end)   cudaEventDestroy(e.end);
        cudaSetDevice(prev);
#endif
    }
}

void Profiler::Enable() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (enabled_) return;
    enabled_ = true;
    start_time_ = std::chrono::steady_clock::now();
}

void Profiler::Disable() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) return;
    // Close any still-open spans
    while (!open_spans_.empty()) {
        auto& span = open_spans_.back();
        span.end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_time_).count();
        spans_.push_back(std::move(span));
        open_spans_.pop_back();
        if (!span_stack_.empty()) span_stack_.pop_back();
    }
    enabled_ = false;
}

SpanID Profiler::BeginSpan(const std::string& name,
                            const std::string& category,
                            int device_id) {
    if (!enabled_) return {};
    std::lock_guard<std::mutex> lock(mutex_);
    Span span;
    span.id       = next_id_++;
    span.parent   = span_stack_.empty() ? 0 : span_stack_.back();
    span.name     = name;
    span.category = category;
    span.device_id = device_id;
    span.start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start_time_).count();
    open_spans_.push_back(span);
    span_stack_.push_back(span.id);

#ifdef TURBORL_HAS_CUDA
    if (config_.enable_nvtx) {
        static constexpr char kNvtxLabel[] = "turborl";
        // nvtxMark is a lightweight zero-duration annotation
        // Full NVTX range push/pop would use nvtxRangePushA / nvtxRangePopA
        (void)kNvtxLabel;
    }
#endif
    return {span.id};
}

void Profiler::EndSpan(SpanID id) {
    if (!enabled_ || id.id == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (open_spans_.empty()) return;
    Span span = std::move(open_spans_.back());
    open_spans_.pop_back();
    if (!span_stack_.empty()) span_stack_.pop_back();
    span.end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start_time_).count();
    spans_.push_back(std::move(span));
}

EventID Profiler::BeginEvent(const std::string& name, int device_id) {
    if (!enabled_) return {};
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef TURBORL_HAS_CUDA
    if (config_.enable_cuda_timing) {
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(device_id);
        CudaEventPair p;
        p.name = name;
        p.device_id = device_id;
        cudaEventCreate(&p.start);
        cudaEventCreate(&p.end);
        cudaEventRecord(p.start, 0);
        cuda_events_.push_back(std::move(p));
        cudaSetDevice(prev);
        return {static_cast<uint64_t>(cuda_events_.size())};
    }
#endif
    (void)name; (void)device_id;
    return {};
}

void Profiler::EndEvent(EventID id) {
    if (!enabled_ || id.id == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef TURBORL_HAS_CUDA
    if (!config_.enable_cuda_timing) return;
    size_t idx = static_cast<size_t>(id.id - 1);
    if (idx >= cuda_events_.size()) return;
    auto& p = cuda_events_[idx];
    cudaEventRecord(p.end, 0);
    cudaEventSynchronize(p.end);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, p.start, p.end);
    p.end_ns = static_cast<int64_t>(ms * 1e6f); // µs → ns
    // Start time for cuda events: approximate from the last CPU span start
    // (CUDA events are pinned to the CPU timeline via stream 0)
    int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start_time_).count();
    p.start_ns = now - p.end_ns;
    (void)now;
#endif
}

void Profiler::Mark(const std::string& /*name*/) {
    // Zero-duration annotation — useful for chrome://tracing
}

std::vector<Span> Profiler::GetAllSpans() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return spans_;
}

std::vector<AggregatedStats> Profiler::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_map<std::string, AggregatedStats> map;
    for (const auto& s : spans_) {
        auto& st = map[s.name];
        st.name = s.name;
        ++st.count;
        st.total_ns += s.duration_ns();
        st.min_ns = std::min(st.min_ns, s.duration_ns());
        st.max_ns = std::max(st.max_ns, s.duration_ns());
    }
    std::vector<AggregatedStats> result;
    result.reserve(map.size());
    for (auto& kv : map) result.push_back(std::move(kv.second));
    return result;
}

void Profiler::ExportChromeTrace(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream f(path);
    if (!f) return;
    f << "[\n";
    bool first = true;

    // CPU spans
    for (const auto& s : spans_) {
        (void)first; // always false after first
        f << (first ? "  " : "  ,");
        first = false;
        f << "{\"name\":\"" << s.name << "\","
          << "\"cat\":\"" << s.category << "\","
          << "\"ph\":\"X\","
          << "\"pid\":0,"
          << "\"tid\":" << s.device_id << ","
          << "\"ts\":" << s.start_ns / 1000 << ","
          << "\"dur\":" << s.duration_ns() / 1000 << "}\n";
    }

    // CUDA events
    for (const auto& e : cuda_events_) {
        f << "  ,{\"name\":\"" << e.name << "\","
          << "\"cat\":\"cuda\","
          << "\"ph\":\"X\","
          << "\"pid\":0,"
          << "\"tid\":" << e.device_id << ","
          << "\"ts\":" << e.start_ns / 1000 << ","
          << "\"dur\":" << e.end_ns / 1000 << "}\n";
    }

    f << "]\n";
}

void Profiler::ExportJSON(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream f(path);
    if (!f) return;
    f << "{\n";
    f << "  \"spans\": " << spans_.size() << ",\n";
    f << "  \"cuda_events\": " << cuda_events_.size() << ",\n";
    f << "  \"stats\": [\n";
    bool first = true;
    for (const auto& st : GetStats()) {
        f << (first ? "    {" : "    ,{");
        first = false;
        f << "\"name\":\"" << st.name << "\","
          << "\"count\":" << st.count << ","
          << "\"total_us\":" << st.total_ns / 1000 << ","
          << "\"avg_us\":" << static_cast<int64_t>(st.avg_ns() / 1000) << ","
          << "\"min_us\":" << st.min_ns / 1000 << ","
          << "\"max_us\":" << st.max_ns / 1000 << "}\n";
    }
    f << "  ]\n}\n";
}

void Profiler::PrintSummary() const {
    auto stats = GetStats();
    int64_t total_ns = 0;
    for (const auto& s : spans_) total_ns += s.duration_ns();

    printf("\n=== TurboRL Profiler Summary ===\n");
    printf("%-32s %10s %10s %10s %10s\n",
           "Name", "Count", "Total(µs)", "Avg(µs)", "Max(µs)");
    printf("%-32s %10s %10s %10s %10s\n",
           "----", "-----", "---------", "--------", "--------");
    for (const auto& s : stats) {
        printf("%-32s %10lld %10lld %10lld %10lld\n",
               s.name.c_str(),
               static_cast<long long>(s.count),
               static_cast<long long>(s.total_ns / 1000),
               static_cast<long long>(s.avg_ns() / 1000),
               static_cast<long long>(s.max_ns / 1000));
    }
    printf("\nTotal time: %.3f ms\n", total_ns / 1e6);
    printf("=================================\n");
}

} // namespace profiler
} // namespace turbol
