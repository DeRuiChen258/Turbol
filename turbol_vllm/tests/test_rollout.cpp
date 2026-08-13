#include <gtest/gtest.h>
#include "turbol/rollout/rollout_engine.hpp"

using namespace turborl;
using namespace turborl::rollout;

class RolloutTest : public ::testing::Test {
protected:
    void SetUp() override {
        config.inference_backend = "vllm";
        config.max_new_tokens = 256;
        config.temperature = 1.0f;
        engine = std::make_unique<RolloutEngine>(config);
    }
    
    RolloutConfig config;
    std::unique_ptr<RolloutEngine> engine;
};

TEST_F(RolloutTest, Construction) {
    EXPECT_NE(engine, nullptr);
}

TEST_F(RolloutTest, InitializeAndShutdown) {
    Status status = engine->Initialize();
    EXPECT_TRUE(status.ok());
    
    status = engine->Shutdown();
    EXPECT_TRUE(status.ok());
}

TEST_F(RolloutTest, Generate) {
    Status status = engine->Initialize();
    EXPECT_TRUE(status.ok());
    
    std::string output;
    status = engine->Generate("Hello, world!", &output);
    EXPECT_TRUE(status.ok());
    EXPECT_FALSE(output.empty());
    EXPECT_NE(output.find("Hello"), std::string::npos);
}

TEST_F(RolloutTest, GenerateBatch) {
    Status status = engine->Initialize();
    EXPECT_TRUE(status.ok());
    
    std::vector<std::string> prompts = {"Hello", "World", "Test"};
    std::vector<std::string> outputs;
    
    status = engine->GenerateBatch(prompts, &outputs);
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(outputs.size(), prompts.size());
}

TEST_F(RolloutTest, Config) {
    EXPECT_EQ(config.inference_backend, "vllm");
    EXPECT_EQ(config.max_new_tokens, 256);
    EXPECT_EQ(config.temperature, 1.0f);
}
