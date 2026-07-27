#include "rff.h"
#include <cmath>
#include <ATen/cuda/CUDAGeneratorImpl.h>

namespace
{

torch::Generator make_generator(const torch::Device& device, const uint64_t seed)
{
    return device.is_cuda()
        ? torch::make_generator<torch::CUDAGeneratorImpl>(seed)
        : torch::make_generator<torch::CPUGeneratorImpl>(seed);
}

torch::Tensor randn_like_shape(
    const std::vector<int64_t>& shape, const torch::Device& device, const uint64_t seed)
{
    auto gen = make_generator(device, seed);
    return torch::randn(
        shape,
        gen,
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );
}

torch::Tensor rand_like_shape(
    const std::vector<int64_t>& shape, const torch::Device& device, const uint64_t seed)
{
    auto gen = make_generator(device, seed);
    return torch::rand(
        shape,
        gen,
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );
}

void check_inputs(
    const torch::Tensor& t,
    const torch::Tensor& x,
    const int64_t dim)
{
    TORCH_CHECK(t.dtype() == torch::kFloat32, "t must be float32");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(t.dim() == 4, "t must be a 4D tensor");
    TORCH_CHECK(x.dim() == 4, "x must be a 4D tensor");
    TORCH_CHECK(
        t.size(1) == x.size(1) && t.size(2) == 1 && t.size(3) == 1,
        "t must have shape (1 or B, T, 1, 1), but got ", t.sizes()
    );
    TORCH_CHECK(
        x.size(2) == 1 && x.size(3) == dim,
        "x must have shape (B, T, 1, ", dim, "), but got ", x.sizes()
    );
    TORCH_CHECK(
        t.size(0) == 1 || t.size(0) == x.size(0),
        "t.size(0) must be 1 or match x.size(0), got t=",
        t.sizes(), ", x=", x.sizes()
    );
}

} // namespace

RandomFeatureFunction::RandomFeatureFunction(
    const int64_t dim,
    const int64_t hidden_dim,
    const float total_time,
    const torch::Device device,
    const uint64_t seed,
    const float scale_min,
    const float scale_max,
    const float space_scale,
    const float time_scale,
    const float bias_scale)
        : dim_(dim),
          hidden_(hidden_dim),
          total_time_(total_time),
          scale_min_(scale_min),
          scale_max_(scale_max),
          space_scale_(space_scale),
          time_scale_(time_scale),
          bias_scale_(bias_scale),
          seed_(seed),
          device_(device)
{
    TORCH_CHECK(dim_ > 0, "dim must be positive");
    TORCH_CHECK(hidden_ > 0, "hidden_dim must be positive");
    TORCH_CHECK(total_time_ > 0.0f, "total_time must be positive");
    TORCH_CHECK(scale_min_ > 0.0f, "scale_min must be positive");
    TORCH_CHECK(scale_max_ >= scale_min_, "scale_max must be at least scale_min");
    TORCH_CHECK(space_scale_ > 0.0f, "space_scale must be positive");
    TORCH_CHECK(time_scale_ >= 0.0f, "time_scale must be nonnegative");
    TORCH_CHECK(bias_scale_ >= 0.0f, "bias_scale must be nonnegative");
    resample_params(seed);
}

void RandomFeatureFunction::resample_params(const uint64_t seed)
{
    q_ = randn_like_shape(
        {dim_, hidden_},
        device_,
        seed ^ 0x9e3779b97f4a7c15ULL
    );
    q_ = q_ / q_.norm(2, 0, true).clamp_min(1.0e-12f);

    const auto log_scale = rand_like_shape(
        {1, hidden_},
        device_,
        seed ^ 0x13198a2e03707344ULL
    );
    scales_ = torch::exp(
        std::log(scale_min_) +
        (std::log(scale_max_) - std::log(scale_min_)) * log_scale
    );
    gamma_ = time_scale_ * randn_like_shape(
        {1, hidden_},
        device_,
        seed ^ 0x243f6a8885a308d3ULL
    );
    c_ = bias_scale_ * randn_like_shape(
        {1, hidden_},
        device_,
        seed ^ 0xb7e151628aed2a6bULL
    );
    seed_ = seed;
}

torch::Tensor RandomFeatureFunction::phi(
    const torch::Tensor& t,
    const torch::Tensor& x
) const
{
    check_inputs(t, x, dim_);

    const auto B = x.size(0);
    const auto T = x.size(1);
    const auto N = B * T;
    const auto t_batched = t.size(0) == B ? t : t.expand({B, T, 1, 1});
    const auto x_flat =
        x.squeeze(2).contiguous().view({N, dim_}) / space_scale_;
    const auto t_flat = t_batched.reshape({N, 1}) / total_time_;
    const auto projection = torch::mm(x_flat, q_) +
        torch::mm(t_flat, gamma_);
    const auto preactivation = projection * scales_ + c_;
    const auto out = torch::tanh(preactivation).view({B, T, hidden_, 1});

    TORCH_CHECK(
        out.sizes() == torch::IntArrayRef({B, T, hidden_, 1}),
        "phi output has wrong shape, expected (",
        B, ", ", T, ", ", hidden_, ", 1), but got ", out.sizes()
    );

    return out;
}

torch::Tensor RandomFeatureFunction::value(
    const torch::Tensor& t,
    const torch::Tensor& x,
    const torch::Tensor& beta0,
    const torch::Tensor& beta
) const
{
    TORCH_CHECK(beta0.numel() == 1, "beta0 must have one element");
    TORCH_CHECK(
        beta.numel() == hidden_,
        "beta must have ", hidden_, " elements, but got ", beta.numel()
    );

    const auto features = phi(t, x).squeeze(-1); // (B, T, H)
    return (
        beta0.reshape({1}) +
        torch::matmul(features, beta.reshape({hidden_}))
    ).unsqueeze(2).unsqueeze(3).contiguous();
}

torch::Tensor RandomFeatureFunction::spatial_gradient_features(
    const torch::Tensor& t,
    const torch::Tensor& x
) const
{
    const auto features = phi(t, x).squeeze(-1); // (B, T, H)
    const auto derivative = (1.0f - features.square()) * scales_;
    return derivative.unsqueeze(-1) *
        q_.transpose(0, 1).reshape({1, 1, hidden_, dim_}) /
        space_scale_;
}

torch::Tensor RandomFeatureFunction::spatial_gradient(
    const torch::Tensor& t,
    const torch::Tensor& x,
    const torch::Tensor& beta
) const
{
    TORCH_CHECK(
        beta.numel() == hidden_,
        "beta must have ", hidden_, " elements, but got ", beta.numel()
    );

    const auto features = phi(t, x).squeeze(-1); // (B, T, H)
    const auto weighted_derivative =
        (1.0f - features.square()) *
        scales_ *
        beta.reshape({1, 1, hidden_});
    return (torch::matmul(weighted_derivative, q_.transpose(0, 1)) /
        space_scale_)
        .unsqueeze(2)
        .contiguous();
}
