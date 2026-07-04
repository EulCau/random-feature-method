#include "nonlinear_solver_utils.h"

namespace solver_utils
{

torch::Tensor pack_nonlinear_parameters(
    const torch::Tensor& y0,
    const torch::Tensor& alpha)
{
    return torch::cat({
        y0.reshape({1}),
        alpha.reshape({-1})
    }).contiguous();
}

torch::Tensor compute_nonlinear_jacobian(
    const torch::Tensor& residual,
    const torch::Tensor& theta)
{
    const int64_t num_residual = residual.numel();
    const int64_t num_param = theta.numel();
    auto jacobian = torch::zeros(
        {num_residual, num_param},
        theta.options().dtype(theta.dtype())
    );

    for (int64_t i = 0; i < num_residual; ++i)
    {
        auto grad_output = torch::zeros_like(residual);
        grad_output.index_put_({i}, 1.0f);

        auto grads = torch::autograd::grad(
            {residual},
            {theta},
            {grad_output},
            true,
            false,
            false
        );

        jacobian.index_put_({i}, grads[0].reshape({num_param}));
    }

    return jacobian.contiguous();
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

} // namespace solver_utils
