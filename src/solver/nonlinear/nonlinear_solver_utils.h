#pragma once

#include <torch/torch.h>

namespace solver_utils
{

[[nodiscard]] torch::Tensor pack_nonlinear_parameters(
    const torch::Tensor& y0,
    const torch::Tensor& alpha);

[[nodiscard]] torch::Tensor compute_nonlinear_jacobian(
    const torch::Tensor& residual,
    const torch::Tensor& theta);

[[nodiscard]] torch::Tensor solve_lm_step(
    const torch::Tensor& jacobian,
    const torch::Tensor& residual,
    float lambda);

} // namespace solver_utils
