#pragma once
#include <torch/torch.h>
#include <random>

// phi(x, t) = tanh(A x + b t + c)
// A: (H, d), b: (H,), c: (H,)
class RandomFeatureFunction {
public:
    RandomFeatureFunction(int64_t dim, int64_t hidden_dim, uint64_t seed = 42);

    // 重新随机采样内层参数（A,b,c）
    void resample_params(uint64_t seed);

    // 计算 phi(x, t)，返回形状 (H) 的 1D Tensor（float）
    // x: (d) 1D Tensor；t: 标量 float
    torch::Tensor phi(const torch::Tensor& x, float t) const;

    int64_t dim() const { return dim_; }
    int64_t hidden_dim() const { return H_; }

    // 直接获取参数（如需调试）
    const torch::Tensor& A() const { return A_; }
    const torch::Tensor& b() const { return b_; }
    const torch::Tensor& c() const { return c_; }

private:
    int64_t dim_;
    int64_t H_;
    torch::Tensor A_; // (H, d)
    torch::Tensor b_; // (H)
    torch::Tensor c_; // (H)
};
