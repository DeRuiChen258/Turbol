#include "turbol/replay_buffer/ring_buffer.hpp"

namespace turborl {
namespace replay_buffer {

RingBuffer::RingBuffer(int capacity, int obs_dim, int act_dim)
    : capacity_(capacity), obs_dim_(obs_dim), act_dim_(act_dim),
      rng_(std::random_device{}()) {
    if (capacity <= 0 || obs_dim <= 0 || act_dim <= 0) {
        throw std::invalid_argument("RingBuffer dims must be positive");
    }

    observations_ = Tensor(Shape({capacity_, obs_dim_}), DataType::kFloat32, Device::CPU());
    actions_ = Tensor(Shape({capacity_, act_dim_}), DataType::kFloat32, Device::CPU());
    rewards_ = Tensor(Shape({capacity_}), DataType::kFloat32, Device::CPU());
    dones_ = Tensor(Shape({capacity_}), DataType::kBool, Device::CPU());
}

RingBuffer::~RingBuffer() = default;

Status RingBuffer::Push(const Tensor& obs, const Tensor& act, const Tensor& reward, bool done) {
    if (obs.shape().NumElements() < obs_dim_)
        return Status::InvalidArgument("obs too small");
    if (act.shape().NumElements() < act_dim_)
        return Status::InvalidArgument("act too small");
    if (reward.shape().NumElements() < 1)
        return Status::InvalidArgument("reward empty");

    const int slot = head_;
    std::memcpy(static_cast<float*>(observations_.data()) + static_cast<size_t>(slot) * obs_dim_,
                obs.data(), static_cast<size_t>(obs_dim_) * sizeof(float));
    std::memcpy(static_cast<float*>(actions_.data()) + static_cast<size_t>(slot) * act_dim_,
                act.data(), static_cast<size_t>(act_dim_) * sizeof(float));
    static_cast<float*>(rewards_.data())[slot] = static_cast<const float*>(reward.data())[0];
    static_cast<bool*>(dones_.data())[slot] = done;

    head_ = (head_ + 1) % capacity_;
    if (size_ < capacity_) ++size_;
    return Status::Ok();
}

Status RingBuffer::Sample(int batch_size, Tensor* obs, Tensor* act, Tensor* reward, Tensor* done) {
    if (!obs || !act || !reward || !done)
        return Status::InvalidArgument("null output pointer");
    if (size_ == 0)
        return Status::InvalidArgument("buffer is empty");
    if (batch_size <= 0)
        return Status::InvalidArgument("batch_size must be > 0");

    *obs = Tensor(Shape({batch_size, obs_dim_}), DataType::kFloat32, Device::CPU());
    *act = Tensor(Shape({batch_size, act_dim_}), DataType::kFloat32, Device::CPU());
    *reward = Tensor(Shape({batch_size}), DataType::kFloat32, Device::CPU());
    *done = Tensor(Shape({batch_size}), DataType::kBool, Device::CPU());

    std::uniform_int_distribution<int> dist(0, size_ - 1);
    for (int i = 0; i < batch_size; ++i) {
        const int src = dist(rng_);
        std::memcpy(static_cast<float*>(obs->data()) + static_cast<size_t>(i) * obs_dim_,
                    static_cast<const float*>(observations_.data()) + static_cast<size_t>(src) * obs_dim_,
                    static_cast<size_t>(obs_dim_) * sizeof(float));
        std::memcpy(static_cast<float*>(act->data()) + static_cast<size_t>(i) * act_dim_,
                    static_cast<const float*>(actions_.data()) + static_cast<size_t>(src) * act_dim_,
                    static_cast<size_t>(act_dim_) * sizeof(float));
        static_cast<float*>(reward->data())[i] = static_cast<const float*>(rewards_.data())[src];
        static_cast<bool*>(done->data())[i] = static_cast<const bool*>(dones_.data())[src];
    }
    return Status::Ok();
}

int RingBuffer::Size() const { return size_; }

void RingBuffer::Clear() {
    head_ = 0;
    size_ = 0;
}

} // namespace replay_buffer
} // namespace turbol
