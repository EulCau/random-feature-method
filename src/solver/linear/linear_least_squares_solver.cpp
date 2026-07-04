#include "linear_least_squares_solver.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#include "ridge_solver.h"

namespace
{

[[nodiscard]] LinearSolveResult split_solution(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& X,
    const int64_t dim,
    const int64_t hidden_dim)
{
    const auto X_matrix = X.reshape({-1, 1}).contiguous();

    const auto y0 = X_matrix.index({0, 0}).clone();
    const auto alpha = X_matrix.index({
        torch::indexing::Slice(1, torch::indexing::None),
        0
    }).reshape({dim, hidden_dim}).contiguous();

    const auto residual = torch::matmul(A.contiguous(), X_matrix) - B.contiguous();
    const float rmse = std::sqrt(residual.pow(2).mean().template item<float>());

    return {y0, alpha, rmse};
}

[[nodiscard]] torch::Tensor solve_qr_libtorch(
    const torch::Tensor& A,
    const torch::Tensor& B)
{
    TORCH_CHECK(A.dim() == 2, "A must be 2D, got ", A.sizes());
    TORCH_CHECK(B.dim() == 2, "B must be 2D, got ", B.sizes());
    TORCH_CHECK(A.size(0) == B.size(0), "A and B row mismatch: A=", A.sizes(), ", B=", B.sizes());
    TORCH_CHECK(A.size(0) >= A.size(1), "QR least squares requires rows >= cols, got ", A.sizes());

    const auto [Q, R] = torch::linalg_qr(A.contiguous(), "reduced");
    const auto rhs = torch::matmul(Q.transpose(0, 1).contiguous(), B.contiguous());
    return torch::linalg_solve_triangular(R, rhs, true).contiguous();
}

[[nodiscard]] std::pair<torch::Tensor, torch::Tensor> reduce_qr_libtorch(
    const torch::Tensor& A,
    const torch::Tensor& B)
{
    TORCH_CHECK(A.dim() == 2, "A must be 2D, got ", A.sizes());
    TORCH_CHECK(B.dim() == 2, "B must be 2D, got ", B.sizes());
    TORCH_CHECK(A.size(0) == B.size(0), "A and B row mismatch: A=", A.sizes(), ", B=", B.sizes());
    TORCH_CHECK(A.size(0) >= A.size(1), "QR reduction requires rows >= cols, got ", A.sizes());

    const auto [Q, R] = torch::linalg_qr(A.contiguous(), "reduced");
    const auto rhs = torch::matmul(Q.transpose(0, 1).contiguous(), B.contiguous());
    return {R.contiguous(), rhs.contiguous()};
}

} // namespace

LinearSolveResult solve_linear_least_squares(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const int64_t dim,
    const int64_t hidden_dim,
    const LinearSolverOptions& options)
{
    if (options.solver_type == LinearSolverType::RidgeDual)
    {
        const auto [y0, alpha, rmse] = solve_y0_alpha_ridge_dual(
            A,
            B,
            dim,
            hidden_dim,
            options.ridge_lambda
        );
        return {y0, alpha, rmse};
    }

    torch::Tensor X;
    if (options.solver_type == LinearSolverType::QR)
    {
        X = solve_qr_libtorch(A, B);
    }
    else
    {
        TORCH_CHECK(options.solver_type == LinearSolverType::BatchedQR, "unknown linear solver type");
        TORCH_CHECK(options.qr_batch_size > 0, "qr_batch_size must be positive");

        std::vector<torch::Tensor> pending_A;
        std::vector<torch::Tensor> pending_B;
        torch::Tensor R;
        torch::Tensor rhs;
        bool initialized = false;
        int64_t pending_rows = 0;
        const int64_t parameter_count = A.size(1);

        pending_A.reserve((parameter_count + options.qr_batch_size - 1) / options.qr_batch_size);
        pending_B.reserve(pending_A.capacity());

        for (int64_t row_begin = 0; row_begin < A.size(0); row_begin += options.qr_batch_size)
        {
            const int64_t row_end = std::min(row_begin + options.qr_batch_size, A.size(0));
            const auto A_batch = A.index({
                torch::indexing::Slice(row_begin, row_end),
                torch::indexing::Slice()
            }).contiguous();
            const auto B_batch = B.index({
                torch::indexing::Slice(row_begin, row_end),
                torch::indexing::Slice()
            }).contiguous();

            if (!initialized)
            {
                pending_A.push_back(A_batch);
                pending_B.push_back(B_batch);
                pending_rows += A_batch.size(0);

                if (pending_rows < parameter_count)
                {
                    continue;
                }

                std::tie(R, rhs) = reduce_qr_libtorch(
                    torch::cat(pending_A, 0).contiguous(),
                    torch::cat(pending_B, 0).contiguous()
                );
                initialized = true;
                pending_A.clear();
                pending_B.clear();
                continue;
            }

            std::tie(R, rhs) = reduce_qr_libtorch(
                torch::cat({R, A_batch}, 0).contiguous(),
                torch::cat({rhs, B_batch}, 0).contiguous()
            );
        }

        TORCH_CHECK(initialized, "A must have at least as many rows as columns for batched QR");
        X = torch::linalg_solve_triangular(R, rhs, true).contiguous();
    }

    return split_solution(A, B, X, dim, hidden_dim);
}
