#include <gtest/gtest.h>
#include "turbol/replay_buffer/ring_buffer.hpp"

using namespace turborl;
using namespace turborl::replay_buffer;

class RingBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        buffer = std::make_unique<RingBuffer>(1000, 128, 8);
    }
    
    std::unique_ptr<RingBuffer> buffer;
};

TEST_F(RingBufferTest, Construction) {
    EXPECT_EQ(buffer->Capacity(), 1000);
    EXPECT_EQ(buffer->Size(), 0);
}

TEST_F(RingBufferTest, PushAndSample) {
    Shape obs_shape({128});
    Shape act_shape({8});
    Shape reward_shape({1});
    
    Tensor obs(obs_shape, DataType::kFloat32, Device::CPU());
    Tensor act(act_shape, DataType::kFloat32, Device::CPU());
    Tensor reward(reward_shape, DataType::kFloat32, Device::CPU());
    
    // Fill with test data
    float* obs_data = static_cast<float*>(obs.data());
    float* act_data = static_cast<float*>(act.data());
    for (int i = 0; i < 128; i++) obs_data[i] = 1.0f;
    for (int i = 0; i < 8; i++) act_data[i] = 0.5f;
    
    Status status = buffer->Push(obs, act, reward, false);
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(buffer->Size(), 1);
    
    // Test sample
    Tensor obs_out, act_out, reward_out, done_out;
    status = buffer->Sample(1, &obs_out, &act_out, &reward_out, &done_out);
    EXPECT_TRUE(status.ok());
}

TEST_F(RingBufferTest, Clear) {
    buffer->Clear();
    EXPECT_EQ(buffer->Size(), 0);
    EXPECT_EQ(buffer->Capacity(), 1000);
}
