#pragma once
#include "../common.hpp"

namespace turborl {
namespace replay_buffer {

class RingBuffer {
public:
    RingBuffer(int capacity, int obs_dim, int act_dim);
    ~RingBuffer();
    
    Status Push(const Tensor& obs, const Tensor& act, const Tensor& reward, bool done);
    Status Sample(int batch_size, Tensor* obs, Tensor* act, Tensor* reward, Tensor* done);
    int Size() const;
    int Capacity() const { return capacity_; }
    void Clear();

private:
    int capacity_;
    int obs_dim_;
    int act_dim_;
    int head_ = 0;
    int size_ = 0;
    
    Tensor observations_;
    Tensor actions_;
    Tensor rewards_;
    Tensor dones_;
};

} // namespace replay_buffer
} // namespace turbol
