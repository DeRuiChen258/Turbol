#include "turbol/core/config.hpp"

#include <cctype>

namespace turborl {

namespace {

std::string Trim(const std::string& s) {
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) ++begin;
    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(begin, end - begin);
}

} // namespace

Status Config::Load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return Status::NotFound("Cannot open config file: " + path);

    std::string line;
    int line_no = 0;
    while (std::getline(file, line)) {
        ++line_no;
        std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;

        size_t eq = trimmed.find('=');
        if (eq == std::string::npos) {
            return Status::InvalidArgument(
                "Malformed config line " + std::to_string(line_no) + ": " + trimmed);
        }
        std::string key = Trim(trimmed.substr(0, eq));
        std::string value = Trim(trimmed.substr(eq + 1));
        if (key.empty()) {
            return Status::InvalidArgument(
                "Empty key on config line " + std::to_string(line_no));
        }
        data_[key] = value;
    }
    return Status::Ok();
}

Status Config::Save(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) return Status::InvalidArgument("Cannot write config file: " + path);

    // Deterministic output order.
    std::vector<std::pair<std::string, std::string>> entries(data_.begin(), data_.end());
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [key, value] : entries) {
        file << key << "=" << value << "\n";
    }
    file.flush();
    return file.good() ? Status::Ok() : Status::InternalError("Failed writing config: " + path);
}

// ---- Scalar parsers ----
bool Config::ParseValue(const std::string& raw, bool& out) {
    std::string v = Trim(raw);
    for (auto& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (v == "true" || v == "1" || v == "yes" || v == "on") { out = true; return true; }
    if (v == "false" || v == "0" || v == "no" || v == "off") { out = false; return true; }
    return false;
}

bool Config::ParseValue(const std::string& raw, int& out) {
    try {
        size_t pos = 0;
        out = std::stoi(Trim(raw), &pos);
        return pos == Trim(raw).size();
    } catch (...) { return false; }
}

bool Config::ParseValue(const std::string& raw, long& out) {
    try {
        size_t pos = 0;
        out = std::stol(Trim(raw), &pos);
        return pos == Trim(raw).size();
    } catch (...) { return false; }
}

bool Config::ParseValue(const std::string& raw, long long& out) {
    try {
        size_t pos = 0;
        out = std::stoll(Trim(raw), &pos);
        return pos == Trim(raw).size();
    } catch (...) { return false; }
}

bool Config::ParseValue(const std::string& raw, float& out) {
    try {
        size_t pos = 0;
        out = std::stof(Trim(raw), &pos);
        return pos == Trim(raw).size();
    } catch (...) { return false; }
}

bool Config::ParseValue(const std::string& raw, double& out) {
    try {
        size_t pos = 0;
        out = std::stod(Trim(raw), &pos);
        return pos == Trim(raw).size();
    } catch (...) { return false; }
}

} // namespace turborl
