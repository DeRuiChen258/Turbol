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

TEST_F(RingBufferTest, SampleRoundTrip) {
    RingBuffer small(/*capacity=*/4, /*obs_dim=*/2, /*act_dim=*/1);

    for (int i = 0; i < 6; ++i) {          // 6 pushes into a capacity-4 buffer
        Tensor obs(Shape({2}), DataType::kFloat32, Device::CPU());
        Tensor act(Shape({1}), DataType::kFloat32, Device::CPU());
        Tensor reward(Shape({1}), DataType::kFloat32, Device::CPU());
        static_cast<float*>(obs.data())[0] = static_cast<float>(i);
        static_cast<float*>(obs.data())[1] = static_cast<float>(i) * 10.0f;
        static_cast<float*>(act.data())[0] = static_cast<float>(i) + 0.5f;
        static_cast<float*>(reward.data())[0] = static_cast<float>(i) * 100.0f;

        EXPECT_TRUE(small.Push(obs, act, reward, i % 2 == 1).ok());
    }
    EXPECT_EQ(small.Size(), 4);            // capacity caps the size

    Tensor obs, act, reward, done;
    ASSERT_TRUE(small.Sample(8, &obs, &act, &reward, &done).ok());
    ASSERT_EQ(obs.shape().dims, std::vector<int64_t>({8, 2}));

    const float* obs_data = static_cast<const float*>(obs.data());
    const float* act_data = static_cast<const float*>(act.data());
    const float* reward_data = static_cast<const float*>(reward.data());
    for (int i = 0; i < 8; ++i) {
        // Only the 4 most recent transitions survive in the buffer.
        EXPECT_GE(obs_data[i * 2], 2.0f);
        EXPECT_LE(obs_data[i * 2], 5.0f);
        EXPECT_FLOAT_EQ(obs_data[i * 2 + 1], obs_data[i * 2] * 10.0f);
        EXPECT_FLOAT_EQ(act_data[i], obs_data[i * 2] + 0.5f);
        EXPECT_FLOAT_EQ(reward_data[i], obs_data[i * 2] * 100.0f);
    }
}

TEST_F(RingBufferTest, SampleRejectsEmptyBuffer) {
    Tensor obs, act, reward, done;
    EXPECT_FALSE(buffer->Sample(1, &obs, &act, &reward, &done).ok());
}

TEST_F(RingBufferTest, Clear) {
    buffer->Clear();
    EXPECT_EQ(buffer->Size(), 0);
    EXPECT_EQ(buffer->Capacity(), 1000);
}
