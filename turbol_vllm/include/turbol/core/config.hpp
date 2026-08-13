#pragma once
#include "../common.hpp"

namespace turborl {

// ============================================================================
// Config — lightweight typed key/value store with file persistence.
//   On-disk format is a simple "key=value" text file (one entry per line,
//   '#' starts a comment). This keeps the module dependency-free while
//   remaining trivially parseable and human-readable.
// ============================================================================
class Config {
public:
    Config() = default;
    ~Config() = default;

    // Load entries from `path` (existing entries are merged/overridden).
    Status Load(const std::string& path);

    // Persist entries to `path` in "key=value" form.
    Status Save(const std::string& path) const;

    // Typed accessors. All return `default_value` when the key is absent or
    // cannot be parsed as T.
    template <typename T>
    T Get(const std::string& key, const T& default_value) const {
        auto it = data_.find(key);
        if (it == data_.end()) return default_value;
        T parsed{};
        if (ParseValue(it->second, parsed)) return parsed;
        return default_value;
    }

    // Set a value (converted to its canonical string form).
    template <typename T>
    void Set(const std::string& key, const T& value) {
        data_[key] = ToString(value);
    }

    bool Has(const std::string& key) const {
        return data_.find(key) != data_.end();
    }

    void Erase(const std::string& key) { data_.erase(key); }

    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }
    void Clear() { data_.clear(); }

private:
    static std::string ToString(const std::string& v) { return v; }
    static std::string ToString(const char* v) { return std::string(v); }
    static std::string ToString(bool v) { return v ? "true" : "false"; }
    static std::string ToString(int v) { return std::to_string(v); }
    static std::string ToString(long v) { return std::to_string(v); }
    static std::string ToString(long long v) { return std::to_string(v); }
    static std::string ToString(unsigned v) { return std::to_string(v); }
    static std::string ToString(float v) {
        // max_digits10 guarantees a lossless round-trip.
        std::ostringstream oss;
        oss << std::setprecision(9) << v;
        return oss.str();
    }
    static std::string ToString(double v) {
        std::ostringstream oss;
        oss << std::setprecision(17) << v;
        return oss.str();
    }

    static bool ParseValue(const std::string& raw, std::string& out) { out = raw; return true; }
    static bool ParseValue(const std::string& raw, bool& out);
    static bool ParseValue(const std::string& raw, int& out);
    static bool ParseValue(const std::string& raw, long& out);
    static bool ParseValue(const std::string& raw, long long& out);
    static bool ParseValue(const std::string& raw, float& out);
    static bool ParseValue(const std::string& raw, double& out);

    std::unordered_map<std::string, std::string> data_;
};

} // namespace turborl
