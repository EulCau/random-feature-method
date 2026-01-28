#include "rff.h"
#include <cmath>

[[maybe_unused]] static inline torch::Tensor randn_like_shape(const std::vector<int64_t>& shape, uint64_t seed) {
    auto gen = torch::make_generator<torch::CPUGeneratorImpl>(seed);
    return torch::randn(shape, gen, torch::TensorOptions().dtype(torch::kFloat32));
}

RandomFeatureFunction::RandomFeatureFunction(int64_t dim, int64_t hidden_dim, uint64_t seed)
        : dim_(dim), H_(hidden_dim) {
    resample_params(seed);
}

void RandomFeatureFunction::resample_params(uint64_t seed) {
    // 可试验其他分布，此处暂用 N(0,1)
    auto gen_A = torch::make_generator<torch::CPUGeneratorImpl>(seed ^ 0x9e3779b97f4a7c15ULL);
    auto gen_b = torch::make_generator<torch::CPUGeneratorImpl>(seed ^ 0x243f6a8885a308d3ULL);
    auto gen_c = torch::make_generator<torch::CPUGeneratorImpl>(seed ^ 0xb7e151628aed2a6bULL);

    A_ = torch::randn({H_, dim_}, gen_A, torch::TensorOptions().dtype(torch::kFloat32));
    b_ = torch::randn({H_}, gen_b, torch::TensorOptions().dtype(torch::kFloat32));
    c_ = torch::randn({H_}, gen_c, torch::TensorOptions().dtype(torch::kFloat32));
}

torch::Tensor RandomFeatureFunction::phi(const torch::Tensor& x, float t) const {
    // x: (d) float32
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 1 && x.size(0) == dim_, "x shape must be (dim,)");

    // Ax : (H), b t: (H), c: (H)
    torch::Tensor Ax = torch::matmul(A_, x);               // (H)
    torch::Tensor bt = b_ * t;                             // (H)
    torch::Tensor h  = Ax + bt + c_;                       // (H)
    return torch::tanh(h);                                 // (H)
}
