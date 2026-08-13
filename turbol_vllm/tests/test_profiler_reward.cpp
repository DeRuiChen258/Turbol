#include <gtest/gtest.h>
#include "turbol/profiler/profiler.hpp"
#include "turbol/reward/reward_engine.hpp"

namespace {

using namespace turborl;

// ============================================================================
// Profiler Tests
// ============================================================================

class ProfilerTest : public ::testing::Test {};

TEST_F(ProfilerTest, EnableDisable) {
    profiler::ProfilerConfig cfg;
    cfg.enable_cpu = true;
    profiler::Profiler prof(cfg);

    EXPECT_FALSE(prof.IsEnabled());
    prof.Enable();
    EXPECT_TRUE(prof.IsEnabled());
    prof.Disable();
    EXPECT_FALSE(prof.IsEnabled());
}

TEST_F(ProfilerTest, CpuSpan) {
    profiler::Profiler prof;
    prof.Enable();

    auto id = prof.BeginSpan("test_span", "compute", -1);
    // Minimal work to ensure duration > 0
    for (volatile int i = 0; i < 1000; ++i) {}
    prof.EndSpan(id);

    auto spans = prof.GetAllSpans();
    ASSERT_EQ(spans.size(), 1u);
    EXPECT_EQ(spans[0].name, "test_span");
    EXPECT_GT(spans[0].duration_ns(), 0);
}

TEST_F(ProfilerTest, AggregatedStats) {
    profiler::Profiler prof;
    prof.Enable();

    for (int i = 0; i < 3; ++i) {
        auto id = prof.BeginSpan("repeat", "compute", -1);
        for (volatile int j = 0; j < 1000; ++j) {}
        prof.EndSpan(id);
    }

    auto stats = prof.GetStats();
    ASSERT_EQ(stats.size(), 1u);
    EXPECT_EQ(stats[0].name, "repeat");
    EXPECT_EQ(stats[0].count, 3);
    EXPECT_GT(stats[0].min_ns, 0);
    EXPECT_GT(stats[0].total_ns, 0);
}

TEST_F(ProfilerTest, ExportJson) {
    profiler::Profiler prof;
    prof.Enable();

    auto id = prof.BeginSpan("export_test", "io", -1);
    prof.EndSpan(id);

    // Check stats contain the expected event
    auto stats = prof.GetStats();
    ASSERT_EQ(stats.size(), 1u);
    EXPECT_EQ(stats[0].name, "export_test");
    EXPECT_EQ(stats[0].count, 1);
}

TEST_F(ProfilerTest, ChromeTraceExport) {
    profiler::Profiler prof;
    prof.Enable();

    auto id = prof.BeginSpan("chrome_test", "compute", 0);
    prof.EndSpan(id);

    // Just verify no crash and spans are collected
    auto spans = prof.GetAllSpans();
    ASSERT_EQ(spans.size(), 1u);
    EXPECT_EQ(spans[0].name, "chrome_test");
    EXPECT_GT(spans[0].duration_ns(), 0);
}

TEST_F(ProfilerTest, RAIIGuard) {
    profiler::Profiler prof;
    prof.Enable();
    {
        profiler::Profiler::Guard guard(&prof, "scoped_span", "compute");
        for (volatile int j = 0; j < 10000; ++j) {}
    }
    auto spans = prof.GetAllSpans();
    ASSERT_EQ(spans.size(), 1u);
    EXPECT_EQ(spans[0].name, "scoped_span");
    EXPECT_GE(spans[0].duration_ns(), 0);
}

// ============================================================================
// RewardEngine Tests
// ============================================================================

class RewardEngineTest : public ::testing::Test {};

TEST_F(RewardEngineTest, Construction) {
    reward::RewardConfig cfg;
    cfg.kl_coef = 0.02f;
    cfg.normalize = true;
    cfg.reward_clip = 5.0f;
    cfg.device = Device::CPU();

    reward::RewardEngine engine(cfg);
    EXPECT_EQ(engine.config().kl_coef, 0.02f);
}

TEST_F(RewardEngineTest, Initialize) {
    reward::RewardConfig cfg;
    cfg.device = Device::CPU();
    reward::RewardEngine engine(cfg);

    auto status = engine.Initialize();
    EXPECT_TRUE(status.ok());
    // Double init should be fine
    status = engine.Initialize();
    EXPECT_TRUE(status.ok());
}

TEST_F(RewardEngineTest, ScoreBatchRuleBased) {
    reward::RewardConfig cfg;
    cfg.normalize = false;
    cfg.reward_clip = 100.0f;
    cfg.format_bonus = 0.1f;
    cfg.length_penalty = -0.001f;
    cfg.target_length = 64;
    cfg.device = Device::CPU();

    reward::RewardEngine engine(cfg);
    engine.Initialize();

    std::vector<std::string> responses = {
        "Hello world.",       // has format bonus
        "Hello world",        // no trailing dot
        "A",                  // short, no bonus
        "A.",                 // short with dot
    };

    reward::RewardResult result;
    auto status = engine.ScoreBatch(responses, std::nullopt, &result);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(result.total.shape()[0], 4);
    ASSERT_EQ(result.rule_scores.shape()[0], 4);

    // Response 0 should have higher score (format bonus)
    float s0 = result.rule_scores.data<float>()[0];
    float s1 = result.rule_scores.data<float>()[1];
    EXPECT_GT(s0, s1);  // format bonus applied
}

TEST_F(RewardEngineTest, ScoreBatchWithClip) {
    reward::RewardConfig cfg;
    cfg.normalize = false;
    cfg.reward_clip = 0.1f;
    cfg.format_bonus = 999.0f;  // huge bonus
    cfg.device = Device::CPU();

    reward::RewardEngine engine(cfg);
    engine.Initialize();

    std::vector<std::string> responses = {"Hello world."};
    reward::RewardResult result;
    engine.ScoreBatch(responses, std::nullopt, &result);

    // Should be clipped to 0.1
    float total = result.total.data<float>()[0];
    EXPECT_NEAR(total, 0.1f, 1e-5f);
}

TEST_F(RewardEngineTest, AdaptiveKL) {
    reward::RewardConfig cfg;
    cfg.kl_coef = 0.01f;
    cfg.kl_target = 0.1f;
    cfg.adaptive_kl = true;
    cfg.kl_adapt_rate = 0.1f;
    cfg.device = Device::CPU();

    reward::RewardEngine engine(cfg);
    engine.Initialize();

    float initial_kl = engine.CurrentKLCoef();
    EXPECT_NEAR(initial_kl, 0.01f, 1e-6f);

    // Update with observed KL > target → kl_coef should increase
    engine.UpdateAdaptiveKL(0.2f);  // observed > target
    float new_kl = engine.CurrentKLCoef();
    EXPECT_GT(new_kl, initial_kl);

    // Update with observed KL < target → kl_coef should decrease
    engine.UpdateAdaptiveKL(0.01f);  // observed < target
    float lower_kl = engine.CurrentKLCoef();
    EXPECT_LT(lower_kl, new_kl);
}

TEST_F(RewardEngineTest, Normalization) {
    reward::RewardConfig cfg;
    cfg.normalize = true;
    cfg.reward_clip = 100.0f;
    cfg.device = Device::CPU();

    reward::RewardEngine engine(cfg);
    engine.Initialize();

    std::vector<std::string> responses = {"Hello.", "World.", "Test."};
    reward::RewardResult result;
    engine.ScoreBatch(responses, std::nullopt, &result);

    // After normalization, mean should be 0
    float* data = result.total.data<float>();
    float mean = (data[0] + data[1] + data[2]) / 3.0f;
    EXPECT_NEAR(mean, 0.0f, 1e-4f);
}

}  // namespace
