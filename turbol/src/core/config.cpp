#include "turbol/core/config.hpp"

namespace turborl {

Config::Config() {}
Config::~Config() {}

Status Config::Load(const std::string& path) {
    return Status::Ok();
}

Status Config::Save(const std::string& path) const {
    return Status::Ok();
}

} // namespace turbol
