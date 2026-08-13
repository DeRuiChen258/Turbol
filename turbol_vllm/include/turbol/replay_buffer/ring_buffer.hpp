#pragma once
#include "../common.hpp"

namespace turborl {
namespace replay_buffer {

// ============================================================================
// RingBuffer — fixed-capacity GPU/CPU-native experience replay store.
//
//   Storage layout (row-major, one slot per transition):
//     observations_ : [capacity, obs_dim]  float32
//     actions_      : [capacity, act_dim]  float32
//     rewards_      : [capacity]           float32
//     dones_        : [capacity]           bool
//
//   Push is O(1) (single slot write + head advance). Sample performs uniform
//   sampling with replacement over the valid [0, size_) range. All data stays
//   on the configured device (CPU or CUDA); no implicit host transfers occur.
// ============================================================================
class RingBuffer {
public:
    RingBuffer(int capacity, int obs_dim, int act_dim,
               Device device = Device::CPU());
    ~RingBuffer();

    // Push a single transition. `reward` must hold at least one element.
    Status Push(const Tensor& obs, const Tensor& act, const Tensor& reward, bool done);

    // Push a batch of transitions (obs: [B, obs_dim], act: [B, act_dim],
    // reward: [B], done: [B] bool).
    Status PushBatch(const Tensor& obs, const Tensor& act,
                     const Tensor& reward, const Tensor& done);

    // Uniformly sample `batch_size` transitions (with replacement).
    // Outputs are allocated on this buffer's device:
    //   obs    [batch_size, obs_dim]
    //   act    [batch_size, act_dim]
    //   reward [batch_size]
    //   done   [batch_size] bool
    Status Sample(int batch_size, Tensor* obs, Tensor* act,
                  Tensor* reward, Tensor* done);

    int Size() const { return size_; }
    int Capacity() const { return capacity_; }
    Device device() const { return device_; }
    void Clear();

private:
    // Fill `indices` with `batch_size` uniform samples from [0, size_).
    void SampleIndices(int batch_size, std::vector<int>* indices);

    int capacity_;
    int obs_dim_;
    int act_dim_;
    int head_ = 0;
    int size_ = 0;
    Device device_;

    Tensor observations_;  // [capacity, obs_dim]
    Tensor actions_;       // [capacity, act_dim]
    Tensor rewards_;       // [capacity]
    Tensor dones_;         // [capacity] bool

    std::mt19937 rng_;
};

} // namespace replay_buffer
} // namespace turborl
