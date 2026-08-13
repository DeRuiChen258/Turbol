#pragma once
#include "../common.hpp"

namespace turborl {

class Config {
public:
    Config();
    ~Config();
    
    Status Load(const std::string& path);
    Status Save(const std::string& path) const;
    
    template<typename T>
    T Get(const std::string& key, const T& default_value) const;
    
private:
    std::unordered_map<std::string, std::string> data_;
};

} // namespace turbol
