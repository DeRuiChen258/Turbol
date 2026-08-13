#pragma once
#include "../common.hpp"

namespace turborl {

class CoreEngine {
public:
    static CoreEngine& Instance();
    
    Status Initialize();
    Status Run();
    Status Shutdown();
    Status Synchronize();
    
    template<typename T>
    T* GetModule(const std::string& name);
    
private:
    CoreEngine();
    ~CoreEngine();
};

} // namespace turbol
