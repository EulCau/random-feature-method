#pragma once

#include <torch/torch.h>

inline std::tuple<torch::Tensor, torch::Tensor, float> solve_y0_alpha_ridge_dual(
    const torch::Tensor& A,          // (n, 1 + dim * hidden_dim)
    const torch::Tensor& B,          // (n, 1)
    int64_t dim,
    int64_t hidden_dim,
    const double lambda = 1e-6
) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D, got ", A.sizes());
    TORCH_CHECK(B.dim() == 2, "B must be 2D, got ", B.sizes());
    TORCH_CHECK(B.size(1) == 1, "B must have shape (n, 1), got ", B.sizes());
    TORCH_CHECK(
        A.size(0) == B.size(0),
        "A and B row mismatch: A=", A.sizes(), ", B=", B.sizes()
    );
    TORCH_CHECK(
        A.size(1) == 1 + dim * hidden_dim,
        "A must have shape (n, 1 + dim * hidden_dim), got A=",
        A.sizes(), ", dim=", dim, ", hidden_dim=", hidden_dim
    );
    TORCH_CHECK(lambda > 0.0, "lambda must be positive");

    auto device = A.device();
    auto dtype = A.dtype();

    // 对偶形式: x = A^T (A A^T + lambda I)^(-1) B
    const auto At = A.transpose(0, 1).contiguous();                 // (p, n)
    const auto G = torch::matmul(A, At);                   // (n, n)
    const auto I = torch::eye(G.size(0), torch::TensorOptions().dtype(dtype).device(device));
    const auto Y = torch::linalg_solve(G + lambda * I, B);    // (n, 1)
    const auto X = torch::matmul(At, Y).contiguous();      // (p, 1)

    // 拆分参数
    const auto y0 = X.index({0, 0}).clone();                  // scalar
    const auto alpha = X.index({
        torch::indexing::Slice(1, torch::indexing::None),
        0
    }).reshape({dim, hidden_dim}).contiguous();

    // 计算 MSE loss
    const auto residual = torch::matmul(A, X) - B;         // (n, 1)
    const auto mse_loss = std::sqrt(residual.pow(2).mean().item<float>());

    return {y0, alpha, mse_loss};
}
