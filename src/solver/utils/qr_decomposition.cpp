#include "qr_decomposition.h"

#include <cmath>

namespace solver_utils
{
namespace
{

void check_qr_input(const torch::Tensor& A)
{
    TORCH_CHECK(A.dim() == 2, "A must be 2D, got ", A.sizes());
    TORCH_CHECK(A.is_floating_point(), "A must be floating point, got ", A.dtype());
}

torch::Tensor rhs_as_matrix(const torch::Tensor& b)
{
    TORCH_CHECK(b.dim() == 1 || b.dim() == 2, "b must be 1D or 2D, got ", b.sizes());
    if (b.dim() == 1)
    {
        return b.reshape({b.size(0), 1}).contiguous();
    }
    return b.contiguous();
}

torch::Tensor restore_rhs_rank(const torch::Tensor& rhs, const bool input_was_vector)
{
    if (input_was_vector)
    {
        return rhs.squeeze(1).contiguous();
    }
    return rhs.contiguous();
}

void check_least_squares_input(const torch::Tensor& A, const torch::Tensor& b)
{
    check_qr_input(A);
    TORCH_CHECK(
        A.size(0) >= A.size(1),
        "A must have at least as many rows as columns, got ", A.sizes()
    );
    TORCH_CHECK(b.dim() == 1 || b.dim() == 2, "b must be 1D or 2D, got ", b.sizes());
    TORCH_CHECK(
        b.size(0) == A.size(0),
        "A and b row mismatch: A=", A.sizes(), ", b=", b.sizes()
    );
    TORCH_CHECK(
        b.device() == A.device(),
        "A and b must be on the same device, got ", A.device(), " and ", b.device()
    );
    TORCH_CHECK(
        b.dtype() == A.dtype(),
        "A and b must have the same dtype, got ", A.dtype(), " and ", b.dtype()
    );
}

} // namespace

QRDecomposition householder_qr(const torch::Tensor& A)
{
    check_qr_input(A);

    const int64_t m = A.size(0);
    const int64_t n = A.size(1);
    const int64_t steps = std::min(m, n);
    auto R = A.contiguous().clone();
    auto Q = torch::eye(m, torch::TensorOptions().dtype(A.dtype()).device(A.device()));

    for (int64_t k = 0; k < steps; ++k)
    {
        auto x = R.index({
            torch::indexing::Slice(k, torch::indexing::None),
            k
        }).contiguous();
        const auto norm_x_tensor = torch::linalg_norm(x);
        const double norm_x = norm_x_tensor.item<double>();
        if (norm_x == 0.0)
        {
            continue;
        }

        auto e1 = torch::zeros_like(x);
        e1.index_put_({0}, 1.0);
        const double sign = x.index({0}).item<double>() >= 0.0 ? 1.0 : -1.0;
        auto v = x + sign * norm_x_tensor * e1;
        const double norm_v = torch::linalg_norm(v).item<double>();
        if (norm_v == 0.0)
        {
            continue;
        }
        v = (v / norm_v).contiguous();

        auto R_sub = R.index({
            torch::indexing::Slice(k, torch::indexing::None),
            torch::indexing::Slice(k, torch::indexing::None)
        });
        const auto R_update = 2.0 * torch::matmul(
            v.reshape({-1, 1}),
            torch::matmul(v.reshape({1, -1}), R_sub)
        );
        R_sub.sub_(R_update);

        auto Q_sub = Q.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(k, torch::indexing::None)
        });
        const auto Q_update = 2.0 * torch::matmul(
            torch::matmul(Q_sub, v.reshape({-1, 1})),
            v.reshape({1, -1})
        );
        Q_sub.sub_(Q_update);
    }

    return {Q.contiguous(), R.contiguous()};
}

QRDecomposition givens_qr(const torch::Tensor& A)
{
    check_qr_input(A);

    const int64_t m = A.size(0);
    const int64_t n = A.size(1);
    auto R = A.contiguous().clone();
    auto Q = torch::eye(m, torch::TensorOptions().dtype(A.dtype()).device(A.device()));

    for (int64_t j = 0; j < n; ++j)
    {
        for (int64_t i = m - 1; i > j; --i)
        {
            const double a = R.index({i - 1, j}).item<double>();
            const double b = R.index({i, j}).item<double>();
            if (b == 0.0)
            {
                continue;
            }

            const double r = std::hypot(a, b);
            const double c = a / r;
            const double s = -b / r;

            const auto row1 = R.index({i - 1, torch::indexing::Slice(j, torch::indexing::None)}).clone();
            const auto row2 = R.index({i, torch::indexing::Slice(j, torch::indexing::None)}).clone();
            R.index_put_(
                {i - 1, torch::indexing::Slice(j, torch::indexing::None)},
                c * row1 - s * row2
            );
            R.index_put_(
                {i, torch::indexing::Slice(j, torch::indexing::None)},
                s * row1 + c * row2
            );

            const auto col1 = Q.index({torch::indexing::Slice(), i - 1}).clone();
            const auto col2 = Q.index({torch::indexing::Slice(), i}).clone();
            Q.index_put_({torch::indexing::Slice(), i - 1}, c * col1 - s * col2);
            Q.index_put_({torch::indexing::Slice(), i}, s * col1 + c * col2);
        }
    }

    return {Q.contiguous(), R.contiguous()};
}

QRDecomposition qr_decompose(
    const torch::Tensor& A,
    const QRMethod method)
{
    switch (method)
    {
    case QRMethod::Householder:
        return householder_qr(A);
    case QRMethod::Givens:
        return givens_qr(A);
    }

    TORCH_CHECK(false, "unknown QR method");
}

QRLeastSquaresReduction reduce_least_squares_qr(
    const torch::Tensor& A,
    const torch::Tensor& b,
    const QRMethod method)
{
    check_least_squares_input(A, b);

    const int64_t n = A.size(1);
    const bool input_was_vector = b.dim() == 1;
    const auto b_matrix = rhs_as_matrix(b);
    const auto [Q, R_full] = qr_decompose(A, method);
    const auto transformed_rhs = torch::matmul(Q.transpose(0, 1).contiguous(), b_matrix);

    auto R = R_full.index({
        torch::indexing::Slice(0, n),
        torch::indexing::Slice()
    }).contiguous();
    auto rhs = transformed_rhs.index({
        torch::indexing::Slice(0, n),
        torch::indexing::Slice()
    }).contiguous();

    return {R, restore_rhs_rank(rhs, input_was_vector)};
}

QRLeastSquaresReduction batched_reduce_least_squares_qr(
    const std::vector<LeastSquaresBatch>& batches,
    const QRMethod method)
{
    TORCH_CHECK(!batches.empty(), "batches must not be empty");

    const auto& first_A = batches.front().A;
    const auto& first_b = batches.front().b;
    check_qr_input(first_A);
    TORCH_CHECK(first_b.dim() == 1 || first_b.dim() == 2, "b must be 1D or 2D, got ", first_b.sizes());

    const int64_t n = first_A.size(1);
    const bool input_was_vector = first_b.dim() == 1;
    const int64_t rhs_cols = input_was_vector ? 1 : first_b.size(1);
    bool initialized = false;
    torch::Tensor R;
    torch::Tensor rhs;
    int64_t pending_rows = 0;
    std::vector<torch::Tensor> pending_A;
    std::vector<torch::Tensor> pending_b;

    for (const auto& batch : batches)
    {
        check_qr_input(batch.A);
        TORCH_CHECK(
            batch.A.size(1) == n,
            "all A batches must have ", n, " columns, got ", batch.A.sizes()
        );
        TORCH_CHECK(
            batch.b.dim() == first_b.dim(),
            "all b batches must have the same rank as the first batch"
        );
        TORCH_CHECK(
            batch.b.size(0) == batch.A.size(0),
            "A and b row mismatch: A=", batch.A.sizes(), ", b=", batch.b.sizes()
        );
        TORCH_CHECK(
            batch.A.device() == first_A.device() && batch.b.device() == first_A.device(),
            "all batches must be on the same device"
        );
        TORCH_CHECK(
            batch.A.dtype() == first_A.dtype() && batch.b.dtype() == first_A.dtype(),
            "all batches must have the same dtype"
        );
        if (!input_was_vector)
        {
            TORCH_CHECK(batch.b.size(1) == rhs_cols, "all b batches must have ", rhs_cols, " columns");
        }

        const auto batch_A = batch.A.contiguous();
        const auto batch_b = rhs_as_matrix(batch.b);

        if (!initialized)
        {
            pending_A.push_back(batch_A);
            pending_b.push_back(batch_b);
            pending_rows += batch_A.size(0);

            if (pending_rows < n)
            {
                continue;
            }

            const auto A_init = torch::cat(pending_A, 0).contiguous();
            const auto b_init = torch::cat(pending_b, 0).contiguous();
            auto reduction = reduce_least_squares_qr(A_init, b_init, method);
            R = reduction.R;
            rhs = rhs_as_matrix(reduction.rhs);
            initialized = true;
            pending_A.clear();
            pending_b.clear();
            continue;
        }

        const auto stacked_A = torch::cat({R, batch_A}, 0).contiguous();
        const auto stacked_b = torch::cat({rhs, batch_b}, 0).contiguous();
        auto reduction = reduce_least_squares_qr(stacked_A, stacked_b, method);
        R = reduction.R;
        rhs = rhs_as_matrix(reduction.rhs);
    }

    TORCH_CHECK(
        initialized,
        "total number of rows across batches must be at least the number of columns"
    );

    return {R.contiguous(), restore_rhs_rank(rhs, input_was_vector)};
}

torch::Tensor solve_reduced_least_squares(const QRLeastSquaresReduction& reduction)
{
    TORCH_CHECK(reduction.R.dim() == 2, "R must be 2D, got ", reduction.R.sizes());
    TORCH_CHECK(
        reduction.R.size(0) == reduction.R.size(1),
        "R must be square, got ", reduction.R.sizes()
    );

    const bool rhs_was_vector = reduction.rhs.dim() == 1;
    const auto rhs = rhs_as_matrix(reduction.rhs);
    TORCH_CHECK(
        rhs.size(0) == reduction.R.size(0),
        "R and rhs row mismatch: R=", reduction.R.sizes(), ", rhs=", reduction.rhs.sizes()
    );

    const auto solution = torch::linalg_solve(reduction.R, rhs);
    if (rhs_was_vector)
    {
        return solution.squeeze(1).contiguous();
    }
    return solution.contiguous();
}

torch::Tensor solve_least_squares_qr(
    const torch::Tensor& A,
    const torch::Tensor& b,
    const QRMethod method)
{
    return solve_reduced_least_squares(reduce_least_squares_qr(A, b, method));
}

torch::Tensor solve_batched_least_squares_qr(
    const std::vector<LeastSquaresBatch>& batches,
    const QRMethod method)
{
    return solve_reduced_least_squares(batched_reduce_least_squares_qr(batches, method));
}

} // namespace solver_utils
