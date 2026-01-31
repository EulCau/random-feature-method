#include "rff.h"
#include <cmath>

// TODO: to GPU
[[maybe_unused]] static torch::Tensor randn_like_shape(const std::vector<int64_t>& shape, const uint64_t seed) {
    auto gen = torch::make_generator<torch::CPUGeneratorImpl>(seed);
    return torch::randn(shape, gen, torch::TensorOptions().dtype(torch::kFloat32));
}

RandomFeatureFunction::RandomFeatureFunction(const int64_t dim, const int64_t hidden_dim, const uint64_t seed)
        : dim_(dim), hidden_(hidden_dim), seed_(seed) {
    resample_params(seed);
}

void RandomFeatureFunction::resample_params(const uint64_t seed) {
    // 可试验其他分布，此处暂用 N(0,1)
    A_ = randn_like_shape({dim_, hidden_}, seed ^ 0x9e3779b97f4a7c15ULL);
    b_ = randn_like_shape({1, hidden_}, seed ^ 0x243f6a8885a308d3ULL);
    c_ = randn_like_shape({1, hidden_}, seed ^ 0xb7e151628aed2a6bULL);
    seed_ = seed;
}

torch::Tensor RandomFeatureFunction::phi(
    const torch::Tensor& t,  // (*, *, 1, 1)
    const torch::Tensor& x   // (*, *, dim, 1)
) const
{
    TORCH_CHECK(t.dtype() == torch::kFloat32, "t must be float32");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    TORCH_CHECK(t.dim() == 4, "t must be a 4D tensor");
    TORCH_CHECK(x.dim() == 4, "x must be a 4D tensor");

    TORCH_CHECK(
        x.size(2) == dim_ && x.size(3) == 1,
        "x must have shape (*, *, ", dim_, ", 1), but got ", x.sizes()
    );

    TORCH_CHECK(
        t.size(2) == 1 && t.size(3) == 1,
        "t must have shape (*, *, 1, 1), but got ", t.sizes()
    );

    // ---- flatten the first two dims ----
    const auto B = x.size(0);
    const auto T = x.size(1);
    const auto N = B * T;

    // x_flat: (N, dim)
    const torch::Tensor x_flat = x.view({N, dim_});

    // t_flat: (N, 1)
    const torch::Tensor t_flat = t.view({N, 1});

    // h: (N, hidden_)
    const torch::Tensor h =
        torch::mm(x_flat, A_) +
        t_flat * b_ +
        c_;

    // output: (B, T, hidden_, 1)
    torch::Tensor out =
        torch::tanh(h)
            .view({B, T, hidden_, 1});

    // ---- final shape check ----
    TORCH_CHECK(
        out.size(0) == x.size(0) &&
        out.size(1) == x.size(1) &&
        out.size(2) == hidden_ &&
        out.size(3) == 1,
        "phi output has wrong shape, expected (",
        x.size(0), ", ",
        x.size(1), ", ",
        hidden_, ", 1), but got ",
        out.sizes()
    );

    return out;
}
