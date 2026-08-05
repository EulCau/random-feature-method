#pragma once

#include "config.h"
#include "equation.h"
#include "rfm_solver.h"

#include <cstdint>
#include <vector>

struct InternalSolutionError
{
    double time;
    int64_t sample_count;
    double reference_mean;
    double reference_std;
    double direct_rmse;
    double direct_normalized_rmse;
    double direct_bias;
    double propagated_rmse;
    double propagated_normalized_rmse;
    double propagated_bias;
    double consistency_rmse;
};

[[nodiscard]] std::vector<InternalSolutionError> validate_internal_solution(
    const RFMSolver& solver,
    const Equation& equation,
    const torch::Tensor& y0,
    const torch::Tensor& beta,
    const ReferenceEvaluationOptions& options,
    uint64_t seed);

void print_internal_solution_errors(
    const std::vector<InternalSolutionError>& errors);
