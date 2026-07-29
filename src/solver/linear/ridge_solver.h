#pragma once

#include <torch/torch.h>

#include <cmath>
#include <tuple>

inline std::tuple<torch::Tensor, torch::Tensor, double> solve_y0_beta_ridge_dual(
    const torch::Tensor& A,          // (n, 1 + hidden_dim)
    const torch::Tensor& B,          // (n, 1)
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
        A.size(1) == 1 + hidden_dim,
        "A must have shape (n, 1 + hidden_dim), got A=",
        A.sizes(), ", hidden_dim=", hidden_dim
    );
    TORCH_CHECK(lambda > 0.0, "lambda must be positive");

    const auto device = A.device();
    const auto dtype = A.dtype();
    const int64_t n = A.size(0);
    const int64_t p = A.size(1);

    const auto opts = torch::TensorOptions().dtype(dtype).device(device);
    const auto A_work = A.contiguous();
    const auto B_work = B.contiguous();

    torch::Tensor X;
    if (n >= p)
    {
        const auto At = A_work.transpose(0, 1).contiguous(); // (p, n)
        const auto normal = torch::matmul(At, A_work);       // (p, p)
        const auto rhs = torch::matmul(At, B_work);          // (p, 1)
        auto penalty = torch::eye(p, opts);

        // y0 is not a random-feature coefficient, so only regularize beta.
        penalty.index_put_({0, 0}, 0.0);
        X = torch::linalg_solve(normal + lambda * penalty, rhs).contiguous();
    }
    else
    {
        // Dual ridge form for underdetermined systems.
        const auto At = A_work.transpose(0, 1).contiguous(); // (p, n)
        const auto G = torch::matmul(A_work, At);            // (n, n)
        const auto I = torch::eye(n, opts);
        const auto Y = torch::linalg_solve(G + lambda * I, B_work);
        X = torch::matmul(At, Y).contiguous();               // (p, 1)
    }

    const auto y0 = X.index({0, 0}).clone();
    const auto beta = X.index({
        torch::indexing::Slice(1, torch::indexing::None),
        0
    }).reshape({hidden_dim}).contiguous();

    const auto residual = torch::matmul(A_work, X) - B_work; // (n, 1)
    const auto mse_loss = std::sqrt(residual.pow(2).mean().item<double>());

    return {y0, beta, mse_loss};
}
