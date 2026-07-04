#pragma once

#include <torch/torch.h>

#include <vector>

namespace solver_utils
{

enum class QRMethod
{
    Householder,
    Givens
};

struct QRDecomposition
{
    torch::Tensor Q; // (m, m)
    torch::Tensor R; // (m, n)
};

struct LeastSquaresBatch
{
    torch::Tensor A; // (batch_size, n)
    torch::Tensor b; // (batch_size) or (batch_size, rhs_dim)
};

struct QRLeastSquaresReduction
{
    torch::Tensor R;   // (n, n)
    torch::Tensor rhs; // (n) or (n, rhs_dim), matching the input rhs rank
};

[[nodiscard]] QRDecomposition householder_qr(const torch::Tensor& A);

[[nodiscard]] QRDecomposition givens_qr(const torch::Tensor& A);

[[nodiscard]] QRDecomposition qr_decompose(
    const torch::Tensor& A,
    QRMethod method);

[[nodiscard]] QRLeastSquaresReduction reduce_least_squares_qr(
    const torch::Tensor& A,
    const torch::Tensor& b,
    QRMethod method);

[[nodiscard]] QRLeastSquaresReduction batched_reduce_least_squares_qr(
    const std::vector<LeastSquaresBatch>& batches,
    QRMethod method);

[[nodiscard]] torch::Tensor solve_reduced_least_squares(
    const QRLeastSquaresReduction& reduction);

[[nodiscard]] torch::Tensor solve_least_squares_qr(
    const torch::Tensor& A,
    const torch::Tensor& b,
    QRMethod method);

[[nodiscard]] torch::Tensor solve_batched_least_squares_qr(
    const std::vector<LeastSquaresBatch>& batches,
    QRMethod method);

} // namespace solver_utils
