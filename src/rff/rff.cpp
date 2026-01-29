#include "rff.h"
#include <cmath>

// TODO: to GPU
[[maybe_unused]] static inline torch::Tensor randn_like_shape(const std::vector<int64_t>& shape, uint64_t seed) {
    auto gen = torch::make_generator<torch::CPUGeneratorImpl>(seed);
    return torch::randn(shape, gen, torch::TensorOptions().dtype(torch::kFloat32));
}

RandomFeatureFunction::RandomFeatureFunction(int64_t dim, int64_t hidden_dim, uint64_t seed)
        : dim_(dim), H_(hidden_dim), seed_(seed) {
    resample_params(seed);
}

void RandomFeatureFunction::resample_params(uint64_t seed) {
    // 可试验其他分布，此处暂用 N(0,1)
    A_ = randn_like_shape({dim_, H_}, seed ^ 0x9e3779b97f4a7c15ULL);
    b_ = randn_like_shape({1, H_}, seed ^ 0x243f6a8885a308d3ULL);
    c_ = randn_like_shape({1, H_}, seed ^ 0xb7e151628aed2a6bULL);
    seed_ = seed;
}

torch::Tensor RandomFeatureFunction::phi(
        const torch::Tensor& x,  // (N, d, 1)
        const torch::Tensor& t   // (N, 1, 1)
) const {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(t.dtype() == torch::kFloat32, "t must be float32");

    TORCH_CHECK(x.dim() == 3, "x must have shape (N, d, 1)");
    TORCH_CHECK(t.dim() == 3, "t must have shape (N, 1, 1)");

    const auto N = x.size(0);

    TORCH_CHECK(x.size(1) == dim_ && x.size(2) == 1,
                "x must have shape (N, dim, 1)");
    TORCH_CHECK(t.size(0) == N && t.size(1) == 1 && t.size(2) == 1,
                "t must have shape (N, 1, 1)");

    torch::Tensor h = torch::mm(x.squeeze(-1), A_) + t.view({-1, 1}) * b_ + c_;

    return torch::tanh(h).unsqueeze(-1);  // (N, H, 1)
}
