#pragma once

#include <torch/torch.h>

#include "linear_solver_options.h"

struct LinearSolveResult
{
    torch::Tensor y0;
    torch::Tensor beta;
    float rmse{};
};

[[nodiscard]] LinearSolveResult solve_linear_least_squares(
    const torch::Tensor& A,
    const torch::Tensor& B,
    int64_t hidden_dim,
    const LinearSolverOptions& options);
