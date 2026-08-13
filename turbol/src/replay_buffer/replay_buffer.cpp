#include "turbol/replay_buffer/ring_buffer.hpp"

namespace turborl {
namespace replay_buffer {

RingBuffer::RingBuffer(int capacity, int obs_dim, int act_dim)
    : capacity_(capacity), obs_dim_(obs_dim), act_dim_(act_dim) {
    
    Shape obs_shape({obs_dim_});
    Shape act_shape({act_dim_});
    Shape reward_shape({1});
    Shape done_shape({1});
    
    observations_ = Tensor(obs_shape, DataType::kFloat32, Device::CPU());
    actions_ = Tensor(act_shape, DataType::kFloat32, Device::CPU());
    rewards_ = Tensor(reward_shape, DataType::kFloat32, Device::CPU());
    dones_ = Tensor(done_shape, DataType::kBool, Device::CPU());
    
    head_ = 0;
    size_ = 0;
}

RingBuffer::~RingBuffer() {}

Status RingBuffer::Push(const Tensor& obs, const Tensor& act, const Tensor& reward, bool done) {
    return Status::Ok();
}

Status RingBuffer::Sample(int batch_size, Tensor* obs, Tensor* act, Tensor* reward, Tensor* done) {
    return Status::Ok();
}

int RingBuffer::Size() const { return size_; }

void RingBuffer::Clear() {
    head_ = 0;
    size_ = 0;
}

} // namespace replay_buffer
} // namespace turbol
