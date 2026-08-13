#include "turbol/profiler/tensorboard_writer.hpp"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/stat.h>
#endif

namespace turborl {
namespace profiler {

// ============================================================================
// TensorBoardWriter implementation
// ============================================================================

TensorBoardWriter::TensorBoardWriter(const std::string& log_dir)
    : log_dir_(log_dir) {
    mkdir(log_dir_.c_str(), 0755);
    event_path_ = log_dir_ + "/events.out.tfevents."
                + std::to_string(wall_time_us());
    file_ = fopen(event_path_.c_str(), "wb");
}

TensorBoardWriter::~TensorBoardWriter() { Flush(); if (file_) fclose(static_cast<FILE*>(file_)); }

int64_t TensorBoardWriter::wall_time_us() const {
    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(epoch).count();
}

// Minimal binary record writer (no protobuf dependency):
//   each record is: [4-byte little-endian size][size bytes data]
void TensorBoardWriter::WriteRecord(const void* data, size_t size) {
    if (!file_) return;
    uint32_t len = static_cast<uint32_t>(size);
    fwrite(&len, 4, 1, static_cast<FILE*>(file_));
    fwrite(data, 1, size, static_cast<FILE*>(file_));
    ++total_written_;
}

void TensorBoardWriter::Flush() {
    if (!file_) return;
    for (const auto& s : pending_scalars_) {
        // Build a minimal Event proto (tagged as "event" message type = 2 in Summary).
        // We write a raw Summary binary blob — TensorBoard is lenient.
        std::string ev;
        // Wire format: field 1 (1) = value, field 2 (2) = SimpleValue
        // SimpleValue: field 1 = float value
        char buf[64];
        int64_t off = 0;
        // tag field (1 << 3 | 2 = 10)
        buf[off++] = 10;
        // length of tag string
        buf[off++] = static_cast<char>(s.tag.size());
        memcpy(buf + off, s.tag.data(), s.tag.size());
        off += static_cast<int>(s.tag.size());
        // value field (2 << 3 | 5 = 18)
        buf[off++] = 18;
        // length = 8 (one float64)
        buf[off++] = 8;
        float f = s.value;
        memcpy(buf + off, &f, 4);
        off += 4;
        // (no more fields)

        // Wrap in Summary (field 1, repeated Summary.Value)
        std::string summary;
        summary.reserve(128);
        char tmp[16];
        // summary field 1 = repeated value
        tmp[0] = 10;  // field 1 << 3 | 2 (length-delimited)
        tmp[1] = static_cast<char>(off);
        summary.append(tmp, 2);
        summary.append(buf, off);

        // Write Event proto: field 1 (wall_time, 9), field 2 (step, 16), field 3 (summary, 26)
        char event[256];
        int64_t eoff = 0;
        // wall_time (9 = field 1, wire 9 = fixed64)
        event[eoff++] = 9;
        memcpy(event + eoff, &s.wall_time_us, 8);
        eoff += 8;
        // step (16 = field 2, wire 16 = fixed64)
        event[eoff++] = 16;
        memcpy(event + eoff, &s.step, 8);
        eoff += 8;
        // summary (26 = field 3, wire 26 = length-delimited)
        event[eoff++] = 26;
        event[eoff++] = static_cast<char>(summary.size());
        memcpy(event + eoff, summary.data(), summary.size());
        eoff += static_cast<int64_t>(summary.size());

        WriteRecord(event, static_cast<size_t>(eoff));
    }
    pending_scalars_.clear();
    fflush(static_cast<FILE*>(file_));
}

void TensorBoardWriter::AddScalar(const std::string& tag, float value, int step) {
    ScalarEvent ev;
    ev.tag = tag;
    ev.value = value;
    ev.step = step;
    ev.wall_time_us = wall_time_us();
    pending_scalars_.push_back(ev);
}

void TensorBoardWriter::AddScalar(const std::string& tag, double value, int step) {
    AddScalar(tag, static_cast<float>(value), step);
}

void TensorBoardWriter::AddHistogram(const std::string& /*tag*/,
                                    const float* /*values*/, int /*count*/, int /*step*/) {
    // Histogram encoding is more involved; stubbed for now.
    // Full implementation would emit Summary with HistogramProto.
}

void TensorBoardWriter::AddText(const std::string& /*tag*/,
                                const std::string& /*text*/, int /*step*/) {
    // Text summary stub.
}

// ============================================================================
// PrometheusWriter implementation
// ============================================================================

std::string Labels::ToString() const {
    if (kv.empty()) return {};
    std::string s = "{";
    bool first = true;
    for (const auto& p : kv) {
        if (!first) s += ",";
        first = false;
        s += p.first + "=\"" + p.second + "\"";
    }
    s += "}";
    return s;
}

PrometheusWriter::PrometheusWriter(const std::string& path) : path_(path) {}

PrometheusWriter::~PrometheusWriter() { Flush(); }

int64_t PrometheusWriter::wall_time_us() const {
    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(epoch).count();
}

void PrometheusWriter::Write(const std::string& name, double value, const Labels& labels) {
    lines_.push_back(name + labels.ToString() + " " + std::to_string(value));
}

void PrometheusWriter::WriteCounter(const std::string& name, double value, const Labels& labels) {
    lines_.push_back("# TYPE " + name + " counter");
    Write(name, value, labels);
}

void PrometheusWriter::WriteGauge(const std::string& name, double value, const Labels& labels) {
    lines_.push_back("# TYPE " + name + " gauge");
    Write(name, value, labels);
}

void PrometheusWriter::WriteHistogram(const std::string& name, double value,
                                     const std::vector<double>& buckets,
                                     const Labels& labels) {
    lines_.push_back("# TYPE " + name + " histogram");
    for (double b : buckets) {
        std::string bucket_name = name + "_bucket{le=\"" + std::to_string(b) + "\"}";
        Write(bucket_name, 0.0, labels);  // placeholder; caller should accumulate
    }
    Write(name + "_sum", value, labels);
    Write(name + "_count", 1.0, labels);
}

void PrometheusWriter::Flush() {
    FILE* f = fopen(path_.c_str(), "w");
    if (!f) return;
    for (const auto& line : lines_) {
        fprintf(f, "%s\n", line.c_str());
    }
    fclose(f);
    lines_.clear();
}

} // namespace profiler
} // namespace turbol
