#include <gtest/gtest.h>

#include "turbol/utils/tensor_utils.hpp"

using namespace turborl;
using namespace turborl::utils;

class TensorUtilsTest : public ::testing::Test {
protected:
    Tensor MakeFloat(const Shape& shape, const std::vector<float>& vals) {
        Tensor t(shape, DataType::kFloat32, Device::CPU());
        std::copy(vals.begin(), vals.end(), t.data<float>());
        return t;
    }
};

TEST_F(TensorUtilsTest, SameLayout) {
    Tensor a(Shape({2, 3}), DataType::kFloat32, Device::CPU());
    Tensor b(Shape({2, 3}), DataType::kFloat32, Device::CPU());
    Tensor c(Shape({2, 4}), DataType::kFloat32, Device::CPU());
    Tensor d(Shape({2, 3}), DataType::kInt32, Device::CPU());

    EXPECT_TRUE(TensorUtils::SameLayout(a, b));
    EXPECT_FALSE(TensorUtils::SameLayout(a, c));
    EXPECT_FALSE(TensorUtils::SameLayout(a, d));
}

TEST_F(TensorUtilsTest, Fill) {
    Tensor t(Shape({8}), DataType::kFloat32, Device::CPU());
    ASSERT_TRUE(TensorUtils::Fill(&t, 3.5f).ok());
    for (int i = 0; i < 8; ++i) EXPECT_FLOAT_EQ(t.data<float>()[i], 3.5f);
}

TEST_F(TensorUtilsTest, AddScalar) {
    Tensor t = MakeFloat(Shape({4}), {1.0f, 2.0f, 3.0f, 4.0f});
    ASSERT_TRUE(TensorUtils::AddScalar(&t, 10.0f).ok());
    EXPECT_FLOAT_EQ(t.data<float>()[0], 11.0f);
    EXPECT_FLOAT_EQ(t.data<float>()[3], 14.0f);
}

TEST_F(TensorUtilsTest, Scale) {
    Tensor t = MakeFloat(Shape({4}), {1.0f, 2.0f, 3.0f, 4.0f});
    ASSERT_TRUE(TensorUtils::Scale(&t, 2.0f).ok());
    EXPECT_FLOAT_EQ(t.data<float>()[0], 2.0f);
    EXPECT_FLOAT_EQ(t.data<float>()[3], 8.0f);
}

TEST_F(TensorUtilsTest, Add) {
    Tensor a = MakeFloat(Shape({3}), {1.0f, 2.0f, 3.0f});
    Tensor b = MakeFloat(Shape({3}), {4.0f, 5.0f, 6.0f});
    Tensor out;
    ASSERT_TRUE(TensorUtils::Add(a, b, &out).ok());
    EXPECT_FLOAT_EQ(out.data<float>()[0], 5.0f);
    EXPECT_FLOAT_EQ(out.data<float>()[1], 7.0f);
    EXPECT_FLOAT_EQ(out.data<float>()[2], 9.0f);
}

TEST_F(TensorUtilsTest, Copy) {
    Tensor src = MakeFloat(Shape({3}), {7.0f, 8.0f, 9.0f});
    Tensor dst;
    ASSERT_TRUE(TensorUtils::Copy(src, &dst).ok());
    EXPECT_EQ(dst.numel(), 3);
    EXPECT_FLOAT_EQ(dst.data<float>()[2], 9.0f);
}
