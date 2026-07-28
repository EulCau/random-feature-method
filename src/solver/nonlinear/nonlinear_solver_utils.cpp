#include "nonlinear_solver_utils.h"

#include <cmath>

namespace solver_utils
{

torch::Tensor pack_nonlinear_parameters(
    const torch::Tensor& y0,
    const torch::Tensor& beta)
{
    return torch::cat({
        y0.reshape({1}),
        beta.reshape({-1})
    }).contiguous();
}

torch::Tensor jacobian_column_scales(
    const torch::Tensor& jacobian,
    const float epsilon)
{
    TORCH_CHECK(jacobian.dim() == 2, "jacobian must be 2D");
    TORCH_CHECK(jacobian.size(0) > 0, "jacobian must have at least one row");
    TORCH_CHECK(epsilon > 0.0f, "column scale epsilon must be positive");
    return jacobian.square()
        .mean(0)
        .sqrt()
        .clamp_min(epsilon)
        .contiguous();
}

torch::Tensor solve_lm_step(
    const torch::Tensor& jacobian,
    const torch::Tensor& residual,
    const float lambda)
{
    TORCH_CHECK(lambda > 0.0f, "lambda must be positive");

    const auto j_t = jacobian.transpose(0, 1).contiguous();
    const auto system = torch::matmul(j_t, jacobian);
    const auto rhs = -torch::matmul(j_t, residual.reshape({-1, 1}));
    const auto identity = torch::eye(
        system.size(0),
        torch::TensorOptions().dtype(system.dtype()).device(system.device())
    );

    return torch::linalg_solve(system + lambda * identity, rhs).reshape({-1}).contiguous();
}

torch::Tensor solve_lm_step_qr(
    const torch::Tensor& jacobian,
    const torch::Tensor& residual,
    const float lambda)
{
    TORCH_CHECK(lambda > 0.0f, "lambda must be positive");

    const int64_t num_param = jacobian.size(1);
    const auto opts = torch::TensorOptions().dtype(jacobian.dtype()).device(jacobian.device());
    const auto regularization = std::sqrt(lambda) * torch::eye(num_param, opts);
    const auto augmented_jacobian = torch::cat({jacobian.contiguous(), regularization}, 0).contiguous();
    const auto augmented_rhs = torch::cat({
        -residual.reshape({-1, 1}).contiguous(),
        torch::zeros({num_param, 1}, opts)
    }, 0).contiguous();

    const auto [Q, R] = torch::linalg_qr(augmented_jacobian, "reduced");
    const auto rhs = torch::matmul(Q.transpose(0, 1).contiguous(), augmented_rhs);
    return torch::linalg_solve_triangular(R, rhs, true).reshape({-1}).contiguous();
}

} // namespace solver_utils
