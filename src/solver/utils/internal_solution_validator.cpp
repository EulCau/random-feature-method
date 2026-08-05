#include "internal_solution_validator.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace
{
uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ x >> 30) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ x >> 27) * 0x94D049BB133111EBULL;
    return x ^ x >> 31;
}

std::vector<int64_t> resolve_reference_time_indices(
    const ReferenceEvaluationOptions& options,
    const int64_t time_interval_count)
{
    TORCH_CHECK(
        time_interval_count >= 2,
        "internal solution validation requires at least two time intervals"
    );
    TORCH_CHECK(
        !options.time_fractions.empty(),
        "reference time_fractions must not be empty"
    );

    std::vector<int64_t> indices;
    indices.reserve(options.time_fractions.size());
    for (const double fraction : options.time_fractions)
    {
        TORCH_CHECK(
            std::isfinite(fraction) && 0.0 < fraction && fraction < 1.0,
            "reference time fraction must satisfy 0 < fraction < 1, got ",
            fraction
        );
        indices.push_back(std::clamp(
            static_cast<int64_t>(std::llround(
                fraction * static_cast<double>(time_interval_count)
            )),
            int64_t{1},
            time_interval_count - 1
        ));
    }
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    return indices;
}
}

std::vector<InternalSolutionError> validate_internal_solution(
    const RFMSolver& solver,
    const Equation& equation,
    const torch::Tensor& y0,
    const torch::Tensor& beta,
    const ReferenceEvaluationOptions& options,
    const uint64_t seed)
{
    TORCH_CHECK(
        equation.has_reference_solution(),
        "internal solution validation requires Equation::reference_solution, "
        "but this equation does not provide it"
    );
    TORCH_CHECK(options.sample_size > 0,
        "reference sample_size must be positive");
    TORCH_CHECK(options.batch_size >= 0,
        "reference batch_size must be nonnegative");

    const auto time_indices = resolve_reference_time_indices(
        options,
        equation.num_time_interval()
    );
    const int64_t point_count = static_cast<int64_t>(time_indices.size());
    const int64_t batch_size = options.batch_size == 0
        ? options.sample_size
        : std::min(options.batch_size, options.sample_size);
    const auto accumulator_options = torch::TensorOptions()
        .dtype(torch::kFloat64)
        .device(solver.device());
    auto reference_sum = torch::zeros({point_count}, accumulator_options);
    auto reference_square_sum =
        torch::zeros({point_count}, accumulator_options);
    auto direct_error_sum = torch::zeros({point_count}, accumulator_options);
    auto direct_error_square_sum =
        torch::zeros({point_count}, accumulator_options);
    auto propagated_error_sum =
        torch::zeros({point_count}, accumulator_options);
    auto propagated_error_square_sum =
        torch::zeros({point_count}, accumulator_options);
    auto consistency_square_sum =
        torch::zeros({point_count}, accumulator_options);
    const auto time_index_tensor = torch::tensor(
        time_indices,
        torch::TensorOptions()
            .dtype(torch::kInt64)
            .device(solver.device())
    );

    torch::manual_seed(splitmix64(seed ^ 0xA0761D6478BD642FULL));
    int64_t evaluated_count = 0;
    for (int64_t begin = 0; begin < options.sample_size; begin += batch_size)
    {
        const int64_t current_batch =
            std::min(batch_size, options.sample_size - begin);
        const auto [dw, x] = equation.sample(current_batch);
        const auto path_evaluation = solver.evaluate_path_values(
            y0,
            beta,
            dw,
            x
        );
        const PathValueEvaluation evaluation{
            path_evaluation.t.index_select(1, time_index_tensor).contiguous(),
            path_evaluation.x.index_select(1, time_index_tensor).contiguous(),
            path_evaluation.direct_value
                .index_select(1, time_index_tensor)
                .contiguous(),
            path_evaluation.propagated_value
                .index_select(1, time_index_tensor)
                .contiguous()
        };
        const auto reference = equation.reference_solution(
            evaluation.t,
            evaluation.x
        ).to(solver.device());
        TORCH_CHECK(
            reference.sizes() == evaluation.direct_value.sizes(),
            "reference solution must return shape ",
            evaluation.direct_value.sizes(),
            ", but got ", reference.sizes()
        );

        const auto reference_matrix = reference
            .reshape({current_batch, point_count})
            .to(torch::kFloat64);
        const auto direct_error = (
            evaluation.direct_value - reference
        ).reshape({current_batch, point_count}).to(torch::kFloat64);
        const auto propagated_error = (
            evaluation.propagated_value - reference
        ).reshape({current_batch, point_count}).to(torch::kFloat64);
        const auto consistency_error = (
            evaluation.propagated_value - evaluation.direct_value
        ).reshape({current_batch, point_count}).to(torch::kFloat64);

        reference_sum += reference_matrix.sum(0);
        reference_square_sum += reference_matrix.square().sum(0);
        direct_error_sum += direct_error.sum(0);
        direct_error_square_sum += direct_error.square().sum(0);
        propagated_error_sum += propagated_error.sum(0);
        propagated_error_square_sum += propagated_error.square().sum(0);
        consistency_square_sum += consistency_error.square().sum(0);
        evaluated_count += current_batch;
    }

    const double count = static_cast<double>(evaluated_count);
    const auto reference_mean = reference_sum / count;
    const auto reference_variance = (
        reference_square_sum / count - reference_mean.square()
    ).clamp_min(0.0);
    const auto reference_std = reference_variance.sqrt();
    const auto direct_rmse = (direct_error_square_sum / count).sqrt();
    const auto propagated_rmse =
        (propagated_error_square_sum / count).sqrt();
    const auto consistency_rmse =
        (consistency_square_sum / count).sqrt();
    const auto direct_bias = direct_error_sum / count;
    const auto propagated_bias = propagated_error_sum / count;
    const auto direct_normalized_rmse =
        direct_rmse / reference_std.clamp_min(1.0e-12);
    const auto propagated_normalized_rmse =
        propagated_rmse / reference_std.clamp_min(1.0e-12);

    const auto reference_mean_cpu = reference_mean.cpu();
    const auto reference_std_cpu = reference_std.cpu();
    const auto direct_rmse_cpu = direct_rmse.cpu();
    const auto direct_normalized_rmse_cpu = direct_normalized_rmse.cpu();
    const auto direct_bias_cpu = direct_bias.cpu();
    const auto propagated_rmse_cpu = propagated_rmse.cpu();
    const auto propagated_normalized_rmse_cpu =
        propagated_normalized_rmse.cpu();
    const auto propagated_bias_cpu = propagated_bias.cpu();
    const auto consistency_rmse_cpu = consistency_rmse.cpu();

    std::vector<InternalSolutionError> errors;
    errors.reserve(static_cast<size_t>(point_count));
    for (int64_t i = 0; i < point_count; ++i)
    {
        errors.push_back(InternalSolutionError{
            equation.delta_t() * static_cast<double>(
                time_indices[static_cast<size_t>(i)]),
            evaluated_count,
            reference_mean_cpu[i].item<double>(),
            reference_std_cpu[i].item<double>(),
            direct_rmse_cpu[i].item<double>(),
            direct_normalized_rmse_cpu[i].item<double>(),
            direct_bias_cpu[i].item<double>(),
            propagated_rmse_cpu[i].item<double>(),
            propagated_normalized_rmse_cpu[i].item<double>(),
            propagated_bias_cpu[i].item<double>(),
            consistency_rmse_cpu[i].item<double>()
        });
    }
    return errors;
}

void print_internal_solution_errors(
    const std::vector<InternalSolutionError>& errors)
{
    for (const auto& error : errors)
    {
        std::cout
            << "reference eval t=" << error.time
            << " samples=" << error.sample_count
            << " reference_mean=" << error.reference_mean
            << " reference_std=" << error.reference_std
            << " direct_rmse=" << error.direct_rmse
            << " direct_normalized_rmse="
            << error.direct_normalized_rmse
            << " direct_bias=" << error.direct_bias
            << " propagated_rmse=" << error.propagated_rmse
            << " propagated_normalized_rmse="
            << error.propagated_normalized_rmse
            << " propagated_bias=" << error.propagated_bias
            << " consistency_rmse=" << error.consistency_rmse
            << std::endl;
    }
}
