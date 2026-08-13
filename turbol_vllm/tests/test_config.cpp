#include <gtest/gtest.h>
#include <fstream>

#include "turbol/core/config.hpp"

using namespace turborl;

class ConfigTest : public ::testing::Test {
protected:
    void TearDown() override {
        std::remove(tmp_path_.c_str());
    }

    std::string tmp_path_ = "/tmp/turborl_config_test.ini";
};

TEST_F(ConfigTest, SetAndGetTyped) {
    Config config;
    config.Set("int_val", 42);
    config.Set("float_val", 3.14f);
    config.Set("double_val", 2.718281828);
    config.Set("bool_val", true);
    config.Set("str_val", std::string("hello"));

    EXPECT_EQ(config.Get<int>("int_val", 0), 42);
    EXPECT_FLOAT_EQ(config.Get<float>("float_val", 0.0f), 3.14f);
    EXPECT_DOUBLE_EQ(config.Get<double>("double_val", 0.0), 2.718281828);
    EXPECT_TRUE(config.Get<bool>("bool_val", false));
    EXPECT_EQ(config.Get<std::string>("str_val", ""), "hello");

    // Missing keys fall back to the default.
    EXPECT_EQ(config.Get<int>("missing", 7), 7);
    EXPECT_FALSE(config.Has("missing"));
}

TEST_F(ConfigTest, SaveAndLoadRoundTrip) {
    Config config;
    config.Set("learning_rate", 1e-4f);
    config.Set("num_layers", 24);
    config.Set("model_name", std::string("turborl-v1"));

    ASSERT_TRUE(config.Save(tmp_path_).ok());

    Config loaded;
    ASSERT_TRUE(loaded.Load(tmp_path_).ok());
    EXPECT_FLOAT_EQ(loaded.Get<float>("learning_rate", 0.0f), 1e-4f);
    EXPECT_EQ(loaded.Get<int>("num_layers", 0), 24);
    EXPECT_EQ(loaded.Get<std::string>("model_name", ""), "turborl-v1");
}

TEST_F(ConfigTest, LoadMissingFile) {
    Config config;
    Status s = config.Load("/tmp/does_not_exist_12345.ini");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), StatusCode::kNotFound);
}

TEST_F(ConfigTest, HasEraseClear) {
    Config config;
    config.Set("a", 1);
    config.Set("b", 2);
    EXPECT_EQ(config.size(), 2u);
    EXPECT_TRUE(config.Has("a"));

    config.Erase("a");
    EXPECT_FALSE(config.Has("a"));
    EXPECT_EQ(config.size(), 1u);

    config.Clear();
    EXPECT_TRUE(config.empty());
}
