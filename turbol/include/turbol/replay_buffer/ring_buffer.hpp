#pragma once
#include "../common.hpp"

namespace turborl {
namespace replay_buffer {

class RingBuffer {
public:
    RingBuffer(int capacity, int obs_dim, int act_dim);
    ~RingBuffer();

    // Push a single transition into the next slot (O(1)).
    Status Push(const Tensor& obs, const Tensor& act, const Tensor& reward, bool done);

    // Uniformly sample `batch_size` transitions (with replacement).
    // Outputs are freshly allocated on the CPU:
    //   obs    [batch_size, obs_dim]
    //   act    [batch_size, act_dim]
    //   reward [batch_size]
    //   done   [batch_size] bool
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

    // Row-major storage, one slot per transition.
    Tensor observations_;  // [capacity, obs_dim]
    Tensor actions_;       // [capacity, act_dim]
    Tensor rewards_;       // [capacity]
    Tensor dones_;         // [capacity] bool

    std::mt19937 rng_;
};

} // namespace replay_buffer
} // namespace turbol
