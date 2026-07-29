#include "rff.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
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
    const std::vector<int64_t>& shape,
    const torch::Device& device,
    const torch::ScalarType dtype,
    const uint64_t seed)
{
    auto gen = make_generator(device, seed);
    return torch::randn(
        shape,
        gen,
        torch::TensorOptions().dtype(dtype).device(device)
    );
}

torch::Tensor rand_like_shape(
    const std::vector<int64_t>& shape,
    const torch::Device& device,
    const torch::ScalarType dtype,
    const uint64_t seed)
{
    auto gen = make_generator(device, seed);
    return torch::rand(
        shape,
        gen,
        torch::TensorOptions().dtype(dtype).device(device)
    );
}

void check_inputs(
    const torch::Tensor& t,
    const torch::Tensor& x,
    const int64_t dim,
    const torch::ScalarType dtype)
{
    TORCH_CHECK(t.dtype() == dtype, "t must have dtype ", dtype);
    TORCH_CHECK(x.dtype() == dtype, "x must have dtype ", dtype);
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

std::vector<int64_t> allocate_band_counts(
    const int64_t hidden_dim,
    const std::vector<RandomFeatureScaleBand>& bands)
{
    const double total_weight = std::accumulate(
        bands.begin(),
        bands.end(),
        0.0,
        [](const double sum, const RandomFeatureScaleBand& band)
        {
            return sum + band.weight;
        }
    );

    std::vector<int64_t> counts(bands.size(), 0);
    std::vector<std::pair<double, size_t>> remainders;
    remainders.reserve(bands.size());
    int64_t assigned = 0;
    for (size_t i = 0; i < bands.size(); ++i)
    {
        const double exact =
            static_cast<double>(hidden_dim) * bands[i].weight / total_weight;
        counts[i] = static_cast<int64_t>(std::floor(exact));
        assigned += counts[i];
        remainders.emplace_back(exact - static_cast<double>(counts[i]), i);
    }

    std::stable_sort(
        remainders.begin(),
        remainders.end(),
        [](const auto& lhs, const auto& rhs)
        {
            return lhs.first > rhs.first;
        }
    );
    for (int64_t i = assigned; i < hidden_dim; ++i)
    {
        ++counts[remainders[static_cast<size_t>(i - assigned)].second];
    }
    return counts;
}

} // namespace

RandomFeatureFunction::RandomFeatureFunction(
    const int64_t dim,
    const int64_t hidden_dim,
    const double total_time,
    const torch::Device device,
    const uint64_t seed,
    const NumericDType dtype,
    const RandomFeatureOptions& options)
        : dim_(dim),
          hidden_(hidden_dim),
          total_time_(total_time),
          space_scale_(options.space_scale),
          time_scale_(options.time_scale),
          bias_scale_(options.bias_scale),
          scale_bands_(options.scale_bands.empty()
              ? std::vector<RandomFeatureScaleBand>{{
                    options.scale_min,
                    options.scale_max,
                    1.0
                }}
              : options.scale_bands),
          seed_(seed),
          device_(device),
          dtype_(dtype == NumericDType::Float64
              ? torch::kFloat64
              : torch::kFloat32)
{
    TORCH_CHECK(dim_ > 0, "dim must be positive");
    TORCH_CHECK(hidden_ > 0, "hidden_dim must be positive");
    TORCH_CHECK(total_time_ > 0.0, "total_time must be positive");
    TORCH_CHECK(space_scale_ > 0.0, "space_scale must be positive");
    TORCH_CHECK(time_scale_ >= 0.0, "time_scale must be nonnegative");
    TORCH_CHECK(bias_scale_ >= 0.0, "bias_scale must be nonnegative");
    TORCH_CHECK(!scale_bands_.empty(), "at least one scale band is required");
    for (const auto& band : scale_bands_)
    {
        TORCH_CHECK(band.scale_min > 0.0, "scale band min must be positive");
        TORCH_CHECK(
            band.scale_max >= band.scale_min,
            "scale band max must be at least min"
        );
        TORCH_CHECK(band.weight > 0.0, "scale band weight must be positive");
    }
    resample_params(seed);
}

void RandomFeatureFunction::resample_params(const uint64_t seed)
{
    q_ = randn_like_shape(
        {dim_, hidden_},
        device_,
        dtype_,
        seed ^ 0x9e3779b97f4a7c15ULL
    );
    q_ = q_ / q_.norm(2, 0, true).clamp_min(1.0e-12);

    const auto uniform_scale = rand_like_shape(
        {1, hidden_},
        device_,
        dtype_,
        seed ^ 0x13198a2e03707344ULL);
    const auto band_counts = allocate_band_counts(hidden_, scale_bands_);
    std::vector<torch::Tensor> scale_chunks;
    scale_chunks.reserve(scale_bands_.size());
    int64_t offset = 0;
    for (size_t i = 0; i < scale_bands_.size(); ++i)
    {
        const int64_t count = band_counts[i];
        if (count == 0)
        {
            continue;
        }
        const auto& band = scale_bands_[i];
        const auto uniform_chunk = uniform_scale.slice(1, offset, offset + count);
        scale_chunks.push_back(torch::exp(
            std::log(band.scale_min) +
            (std::log(band.scale_max) - std::log(band.scale_min)) *
                uniform_chunk
        ));
        offset += count;
    }
    scales_ = torch::cat(scale_chunks, 1).contiguous();
    gamma_ = time_scale_ * randn_like_shape(
        {1, hidden_},
        device_,
        dtype_,
        seed ^ 0x243f6a8885a308d3ULL
    );
    c_ = bias_scale_ * randn_like_shape(
        {1, hidden_},
        device_,
        dtype_,
        seed ^ 0xb7e151628aed2a6bULL
    );
    seed_ = seed;
}

torch::Tensor RandomFeatureFunction::phi(
    const torch::Tensor& t,
    const torch::Tensor& x
) const
{
    check_inputs(t, x, dim_, dtype_);

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
    const auto derivative = (1.0 - features.square()) * scales_;
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
        (1.0 - features.square()) *
        scales_ *
        beta.reshape({1, 1, hidden_});
    return (torch::matmul(weighted_derivative, q_.transpose(0, 1)) /
        space_scale_)
        .unsqueeze(2)
        .contiguous();
}
