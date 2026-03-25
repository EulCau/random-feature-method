#include "rff.h"
#include <cmath>
#include <ATen/cuda/CUDAGeneratorImpl.h>

// TODO: to GPU
[[maybe_unused]] static torch::Tensor randn_like_shape(
    const std::vector<int64_t>& shape, const torch::Device& device, const uint64_t seed)
{

    auto gen = device.is_cuda()?
        torch::make_generator<torch::CUDAGeneratorImpl>(seed):
        torch::make_generator<torch::CPUGeneratorImpl>(seed);

    return torch::randn(shape, gen, torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

RandomFeatureFunction::RandomFeatureFunction(
    const int64_t dim, const int64_t hidden_dim, const torch::Device device, const uint64_t seed)
        : dim_(dim), hidden_(hidden_dim), seed_(seed),
          device_(device)
{
    resample_params(seed);
}

void RandomFeatureFunction::resample_params(const uint64_t seed)
{
    // 可试验其他分布, 此处暂用 N(0,1)
    A_ = randn_like_shape({dim_, hidden_}, device_, seed ^ 0x9e3779b97f4a7c15ULL);
    b_ = randn_like_shape({   1, hidden_}, device_, seed ^ 0x243f6a8885a308d3ULL);
    c_ = randn_like_shape({   1, hidden_}, device_, seed ^ 0xb7e151628aed2a6bULL);
    seed_ = seed;
}

torch::Tensor RandomFeatureFunction::phi(
    const torch::Tensor& t,  // (1, T, 1, 1) 或 (B, T, 1, 1)
    const torch::Tensor& x   // (B, T, 1, dim)
) const
{
    TORCH_CHECK(t.dtype() == torch::kFloat32, "t must be float32");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    TORCH_CHECK(t.dim() == 4, "t must be a 4D tensor");
    TORCH_CHECK(x.dim() == 4, "x must be a 4D tensor");

    TORCH_CHECK(
        t.size(1) == x.size(1) &&
        t.size(2) == 1 &&
        t.size(3) == 1,
        "t must have shape (1 or B, T, 1, 1), but got ", t.sizes()
    );

    TORCH_CHECK(
        x.size(2) == 1 && x.size(3) == dim_,
        "x must have shape (B, T, 1, ", dim_, "), but got ", x.sizes()
    );

    const auto B = x.size(0);
    const auto T = x.size(1);
    const auto N = B * T;

    TORCH_CHECK(
        t.size(0) == 1 || t.size(0) == B,
        "t.size(0) must be 1 or match x.size(0). Got t.size(0)=",
        t.size(0), ", x.size(0)=", B
    );

    // 若 t 是 (1, T, 1, 1), 则扩展成 (B, T, 1, 1)
    const auto t_batched = t.size(0) == B ? t : t.expand({B, T, 1, 1});

    // x_flat: (B*T, dim)
    const auto x_flat = x.squeeze(2).contiguous().view({N, dim_});

    // t_flat: (B*T, 1)
    const auto t_flat = t_batched.reshape({N, 1});

    const auto xA = torch::mm(x_flat, A_);                 // (N, hidden)
    const auto tb = torch::mm(t_flat, b_);                 // (N, hidden)
    const auto c_flat = c_.expand({N, c_.size(1)});      // (N, hidden)

    // h: (B*T, hidden)
    const auto h = xA + tb + c_flat;

    const auto out = torch::tanh(h).view({B, T, hidden_, 1});

    TORCH_CHECK(
        out.size(0) == B &&
        out.size(1) == T &&
        out.size(2) == hidden_ &&
        out.size(3) == 1,
        "phi output has wrong shape, expected (",
        B, ", ", T, ", ", hidden_, ", 1), but got ", out.sizes()
    );

    return out;
}
