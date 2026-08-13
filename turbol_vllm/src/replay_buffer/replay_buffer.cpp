#include "turbol/replay_buffer/ring_buffer.hpp"

namespace turborl {
namespace replay_buffer {

RingBuffer::RingBuffer(int capacity, int obs_dim, int act_dim, Device device)
    : capacity_(capacity), obs_dim_(obs_dim), act_dim_(act_dim),
      device_(device), rng_(std::random_device{}()) {
    if (capacity <= 0 || obs_dim <= 0 || act_dim <= 0) {
        throw std::invalid_argument("RingBuffer dims must be positive");
    }
    observations_ = Tensor(Shape({capacity, obs_dim}), DataType::kFloat32, device_);
    actions_      = Tensor(Shape({capacity, act_dim}), DataType::kFloat32, device_);
    rewards_      = Tensor(Shape({capacity}), DataType::kFloat32, device_);
    dones_        = Tensor(Shape({capacity}), DataType::kBool, device_);
}

RingBuffer::~RingBuffer() = default;

Status RingBuffer::Push(const Tensor& obs, const Tensor& act,
                        const Tensor& reward, bool done) {
    if (obs.numel() < obs_dim_) return Status::InvalidArgument("obs too small");
    if (act.numel() < act_dim_) return Status::InvalidArgument("act too small");
    if (reward.numel() < 1)     return Status::InvalidArgument("reward empty");

    const int slot = head_;
#ifdef TURBORL_HAS_CUDA
    if (device_.is_cuda()) {
        cudaMemcpy(observations_.data<float>() + static_cast<size_t>(slot) * obs_dim_,
                   obs.data(), static_cast<size_t>(obs_dim_) * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(actions_.data<float>() + static_cast<size_t>(slot) * act_dim_,
                   act.data(), static_cast<size_t>(act_dim_) * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(rewards_.data<float>() + slot, reward.data(), sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemsetAsync(dones_.data<bool>() + slot, done ? 1 : 0, sizeof(bool));
        cudaStreamSynchronize(nullptr);
    } else
#endif
    {
        std::memcpy(observations_.data<float>() + static_cast<size_t>(slot) * obs_dim_,
                    obs.data(), static_cast<size_t>(obs_dim_) * sizeof(float));
        std::memcpy(actions_.data<float>() + static_cast<size_t>(slot) * act_dim_,
                    act.data(), static_cast<size_t>(act_dim_) * sizeof(float));
        rewards_.data<float>()[slot] = reward.data<float>()[0];
        dones_.data<bool>()[slot] = done;
    }

    head_ = (head_ + 1) % capacity_;
    if (size_ < capacity_) ++size_;
    return Status::Ok();
}

Status RingBuffer::PushBatch(const Tensor& obs, const Tensor& act,
                             const Tensor& reward, const Tensor& done) {
    const int batch = static_cast<int>(obs.shape()[0]);
    if (batch <= 0) return Status::InvalidArgument("empty batch");
    if (obs.numel() < static_cast<int64_t>(batch) * obs_dim_)
        return Status::InvalidArgument("obs batch dims mismatch");
    if (act.numel() < static_cast<int64_t>(batch) * act_dim_)
        return Status::InvalidArgument("act batch dims mismatch");

    for (int i = 0; i < batch; ++i) {
        // Wrap-around writes are handled slot-by-slot; simple and correct.
        const int slot = head_;
#ifdef TURBORL_HAS_CUDA
        if (device_.is_cuda()) {
            cudaMemcpy(observations_.data<float>() + static_cast<size_t>(slot) * obs_dim_,
                       obs.data<float>() + static_cast<size_t>(i) * obs_dim_,
                       static_cast<size_t>(obs_dim_) * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(actions_.data<float>() + static_cast<size_t>(slot) * act_dim_,
                       act.data<float>() + static_cast<size_t>(i) * act_dim_,
                       static_cast<size_t>(act_dim_) * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(rewards_.data<float>() + slot, reward.data<float>() + i,
                       sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemsetAsync(dones_.data<bool>() + slot, done.data<bool>()[i] ? 1 : 0, sizeof(bool));
        } else
#endif
        {
            std::memcpy(observations_.data<float>() + static_cast<size_t>(slot) * obs_dim_,
                        obs.data<float>() + static_cast<size_t>(i) * obs_dim_,
                        static_cast<size_t>(obs_dim_) * sizeof(float));
            std::memcpy(actions_.data<float>() + static_cast<size_t>(slot) * act_dim_,
                        act.data<float>() + static_cast<size_t>(i) * act_dim_,
                        static_cast<size_t>(act_dim_) * sizeof(float));
            rewards_.data<float>()[slot] = reward.data<float>()[i];
            dones_.data<bool>()[slot] = done.data<bool>()[i];
        }
        head_ = (head_ + 1) % capacity_;
        if (size_ < capacity_) ++size_;
    }
#ifdef TURBORL_HAS_CUDA
    if (device_.is_cuda()) cudaStreamSynchronize(nullptr);
#endif
    return Status::Ok();
}

void RingBuffer::SampleIndices(int batch_size, std::vector<int>* indices) {
    indices->resize(batch_size);
    std::uniform_int_distribution<int> dist(0, size_ - 1);
    for (int i = 0; i < batch_size; ++i) (*indices)[i] = dist(rng_);
}

Status RingBuffer::Sample(int batch_size, Tensor* obs, Tensor* act,
                          Tensor* reward, Tensor* done) {
    if (!obs || !act || !reward || !done)
        return Status::InvalidArgument("null output pointer");
    if (size_ == 0) return Status::InvalidArgument("Buffer is empty");
    if (batch_size <= 0) return Status::InvalidArgument("batch_size must be > 0");

    *obs    = Tensor(Shape({batch_size, obs_dim_}), DataType::kFloat32, device_);
    *act    = Tensor(Shape({batch_size, act_dim_}), DataType::kFloat32, device_);
    *reward = Tensor(Shape({batch_size}), DataType::kFloat32, device_);
    *done   = Tensor(Shape({batch_size}), DataType::kBool, device_);

    std::vector<int> indices;
    SampleIndices(batch_size, &indices);

    for (int i = 0; i < batch_size; ++i) {
        const int src = indices[i];
#ifdef TURBORL_HAS_CUDA
        if (device_.is_cuda()) {
            cudaMemcpy(obs->data<float>() + static_cast<size_t>(i) * obs_dim_,
                       observations_.data<float>() + static_cast<size_t>(src) * obs_dim_,
                       static_cast<size_t>(obs_dim_) * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(act->data<float>() + static_cast<size_t>(i) * act_dim_,
                       actions_.data<float>() + static_cast<size_t>(src) * act_dim_,
                       static_cast<size_t>(act_dim_) * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(reward->data<float>() + i, rewards_.data<float>() + src,
                       sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(done->data<bool>() + i, dones_.data<bool>() + src,
                       sizeof(bool), cudaMemcpyDeviceToDevice);
        } else
#endif
        {
            std::memcpy(obs->data<float>() + static_cast<size_t>(i) * obs_dim_,
                        observations_.data<float>() + static_cast<size_t>(src) * obs_dim_,
                        static_cast<size_t>(obs_dim_) * sizeof(float));
            std::memcpy(act->data<float>() + static_cast<size_t>(i) * act_dim_,
                        actions_.data<float>() + static_cast<size_t>(src) * act_dim_,
                        static_cast<size_t>(act_dim_) * sizeof(float));
            reward->data<float>()[i] = rewards_.data<float>()[src];
            done->data<bool>()[i] = dones_.data<bool>()[src];
        }
    }
    return Status::Ok();
}

void RingBuffer::Clear() {
    head_ = 0;
    size_ = 0;
}

} // namespace replay_buffer
} // namespace turborl
