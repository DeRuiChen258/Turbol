#include <gtest/gtest.h>
#include "turbol/common.hpp"

using namespace turborl;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(TensorTest, DefaultConstruction) {
    Tensor tensor;
    EXPECT_EQ(tensor.size_bytes(), 0);
    EXPECT_FALSE(tensor.is_cuda());
    EXPECT_EQ(tensor.dtype(), DataType::kFloat32);
}

TEST_F(TensorTest, ShapeConstruction) {
    Shape shape({2, 3, 4});
    Tensor tensor(shape, DataType::kFloat32, Device::CPU());
    
    EXPECT_EQ(tensor.shape().size(), 3);
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 3);
    EXPECT_EQ(tensor.shape()[2], 4);
    EXPECT_EQ(tensor.size_bytes(), 2 * 3 * 4 * 4); // 96 bytes
}

TEST_F(TensorTest, CopyConstruction) {
    Shape shape({10, 20});
    Tensor tensor1(shape, DataType::kFloat32, Device::CPU());
    
    // Fill with data
    float* data = static_cast<float*>(tensor1.data());
    for (int i = 0; i < 10 * 20; i++) {
        data[i] = static_cast<float>(i);
    }
    
    Tensor tensor2(tensor1);
    EXPECT_EQ(tensor2.size_bytes(), tensor1.size_bytes());
    
    float* data2 = static_cast<float*>(tensor2.data());
    EXPECT_EQ(data2[0], 0.0f);
    EXPECT_EQ(data2[199], 199.0f);
}

TEST_F(TensorTest, MoveConstruction) {
    Shape shape({5, 5});
    Tensor tensor1(shape, DataType::kFloat32, Device::CPU());
    size_t size = tensor1.size_bytes();
    
    Tensor tensor2(std::move(tensor1));
    EXPECT_EQ(tensor2.size_bytes(), size);
}

TEST_F(TensorTest, DataTypes) {
    EXPECT_EQ(GetDataTypeSize(DataType::kFloat32), 4);
    EXPECT_EQ(GetDataTypeSize(DataType::kFloat16), 2);
    EXPECT_EQ(GetDataTypeSize(DataType::kInt32), 4);
    EXPECT_EQ(GetDataTypeSize(DataType::kInt64), 8);
    EXPECT_EQ(GetDataTypeSize(DataType::kInt8), 1);
}

TEST_F(TensorTest, DeviceOperations) {
    Shape shape({100});
    Tensor tensor(shape, DataType::kFloat32, Device::CPU());
    
    Device cpu = Device::CPU();
    Device cuda = Device::CUDA(0);
    
    EXPECT_TRUE(cpu.is_cpu());
    EXPECT_FALSE(cpu.is_cuda());
    EXPECT_TRUE(cuda.is_cuda());
    EXPECT_FALSE(cuda.is_cpu());
    EXPECT_EQ(cpu.ToString(), "CPU");
    EXPECT_EQ(cuda.ToString(), "CUDA:0");
}

TEST_F(TensorTest, ShapeOperations) {
    Shape shape({2, 3, 4, 5});
    EXPECT_EQ(shape.NumElements(), 2 * 3 * 4 * 5);
    EXPECT_EQ(shape.size(), 4);
}
