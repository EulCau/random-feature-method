#pragma once

#include <torch/torch.h>

namespace solver_utils
{

[[nodiscard]] torch::Tensor pack_nonlinear_parameters(
    const torch::Tensor& y0,
    const torch::Tensor& beta);

[[nodiscard]] torch::Tensor jacobian_column_scales(
    const torch::Tensor& jacobian,
    float epsilon);

[[nodiscard]] torch::Tensor solve_lm_step(
    const torch::Tensor& jacobian,
    const torch::Tensor& residual,
    float lambda);

[[nodiscard]] torch::Tensor solve_lm_step_qr(
    const torch::Tensor& jacobian,
    const torch::Tensor& residual,
    float lambda);

} // namespace solver_utils
