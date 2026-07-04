#include "linear_least_squares_solver.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "qr_decomposition.h"
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
        X = solver_utils::solve_least_squares_qr(A, B, options.qr_method);
    }
    else
    {
        TORCH_CHECK(options.solver_type == LinearSolverType::BatchedQR, "unknown linear solver type");
        TORCH_CHECK(options.qr_batch_size > 0, "qr_batch_size must be positive");

        std::vector<solver_utils::LeastSquaresBatch> batches;
        batches.reserve((A.size(0) + options.qr_batch_size - 1) / options.qr_batch_size);
        for (int64_t row_begin = 0; row_begin < A.size(0); row_begin += options.qr_batch_size)
        {
            const int64_t row_end = std::min(row_begin + options.qr_batch_size, A.size(0));
            batches.push_back({
                A.index({
                    torch::indexing::Slice(row_begin, row_end),
                    torch::indexing::Slice()
                }).contiguous(),
                B.index({
                    torch::indexing::Slice(row_begin, row_end),
                    torch::indexing::Slice()
                }).contiguous()
            });
        }
        X = solver_utils::solve_batched_least_squares_qr(batches, options.qr_method);
    }

    return split_solution(A, B, X, dim, hidden_dim);
}
