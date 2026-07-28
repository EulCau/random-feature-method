# include "rfm_solver.h"
#include "config.h"
#include "equation_factory.h"
#include "register_all_eqn.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

uint64_t splitmix64(uint64_t x);

namespace
{
constexpr auto kDefaultConfigPath = "config/hjb_lq_d100.json";

struct CommandLineOptions
{
    std::string config_path{kDefaultConfigPath};
    std::optional<uint64_t> seed;
    bool reference_evaluation{false};
};

int64_t default_batch_size(const int64_t hidden_dim)
{
    return 4 * (1 + hidden_dim);
}

LinearSolverOptions get_linear_solver_options_from_config(
    const SolverConfig& config)
{
    LinearSolverOptions options;
    TORCH_CHECK(config.linear.batch_size >= 0, "linear batch_size must be nonnegative");
    options.qr_batch_size = config.linear.batch_size == 0
        ? default_batch_size(config.hidden_dim)
        : config.linear.batch_size;
    options.ridge_lambda = config.linear.ridge_lambda;

    //TODO: Replace string-based solver selection with a configurable registry or factory.
    if (config.linear.solver == "constant")
    {
        options.solver_type = LinearSolverType::Constant;
    }
    else if (config.linear.solver == "ridge_dual")
    {
        options.solver_type = LinearSolverType::RidgeDual;
    }
    else if (config.linear.solver == "qr")
    {
        options.solver_type = LinearSolverType::QR;
    }
    else if (config.linear.solver == "batched_qr")
    {
        options.solver_type = LinearSolverType::BatchedQR;
    }
    else
    {
        TORCH_CHECK(false, "unknown linear solver: ", config.linear.solver);
    }

    return options;
}

std::vector<int64_t> resolve_reference_time_indices(
    const ReferenceEvaluationOptions& options,
    const int64_t time_interval_count)
{
    TORCH_CHECK(
        time_interval_count >= 2,
        "reference evaluation requires at least two time intervals"
    );
    TORCH_CHECK(
        !options.time_fractions.empty(),
        "reference time_fractions must not be empty"
    );

    std::vector<int64_t> indices;
    indices.reserve(options.time_fractions.size());
    for (const float fraction : options.time_fractions)
    {
        TORCH_CHECK(
            std::isfinite(fraction) && 0.0f < fraction && fraction < 1.0f,
            "reference time fraction must satisfy 0 < fraction < 1, got ",
            fraction
        );
        indices.push_back(std::clamp(
            static_cast<int64_t>(std::llround(
                fraction * static_cast<float>(time_interval_count)
            )),
            int64_t{1},
            time_interval_count - 1
        ));
    }
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    return indices;
}

void evaluate_reference_solution(
    const RFMSolver& solver,
    const Equation& equation,
    const torch::Tensor& y0,
    const torch::Tensor& beta,
    const ReferenceEvaluationOptions& options,
    const uint64_t seed)
{
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

    torch::manual_seed(splitmix64(seed ^ 0xA0761D6478BD642FULL));
    int64_t evaluated_count = 0;
    for (int64_t begin = 0; begin < options.sample_size; begin += batch_size)
    {
        const int64_t current_batch =
            std::min(batch_size, options.sample_size - begin);
        const auto [dw, x] = equation.sample(current_batch);
        const auto evaluation = solver.evaluate_internal_paths(
            y0,
            beta,
            dw,
            x,
            time_indices
        );
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
        propagated_error_square_sum +=
            propagated_error.square().sum(0);
        consistency_square_sum += consistency_error.square().sum(0);
        evaluated_count += current_batch;
    }

    const auto count = static_cast<double>(evaluated_count);
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
    const auto direct_normalized_rmse_cpu =
        direct_normalized_rmse.cpu();
    const auto direct_bias_cpu = direct_bias.cpu();
    const auto propagated_rmse_cpu = propagated_rmse.cpu();
    const auto propagated_normalized_rmse_cpu =
        propagated_normalized_rmse.cpu();
    const auto propagated_bias_cpu = propagated_bias.cpu();
    const auto consistency_rmse_cpu = consistency_rmse.cpu();

    for (int64_t i = 0; i < point_count; ++i)
    {
        const float time = equation.delta_t() *
            static_cast<float>(time_indices[static_cast<size_t>(i)]);
        std::cout
            << "reference eval t=" << time
            << " samples=" << evaluated_count
            << " reference_mean=" << reference_mean_cpu[i].item<double>()
            << " reference_std=" << reference_std_cpu[i].item<double>()
            << " direct_rmse=" << direct_rmse_cpu[i].item<double>()
            << " direct_normalized_rmse="
            << direct_normalized_rmse_cpu[i].item<double>()
            << " direct_bias=" << direct_bias_cpu[i].item<double>()
            << " propagated_rmse="
            << propagated_rmse_cpu[i].item<double>()
            << " propagated_normalized_rmse="
            << propagated_normalized_rmse_cpu[i].item<double>()
            << " propagated_bias="
            << propagated_bias_cpu[i].item<double>()
            << " consistency_rmse="
            << consistency_rmse_cpu[i].item<double>()
            << std::endl;
    }
}
}

uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ x >> 30) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ x >> 27) * 0x94D049BB133111EBULL;
    return x ^ x >> 31;
}

uint64_t get_seed()
{
    constexpr uint64_t default_seed = 0xC02E7A5B3F91A8C3ULL;

    std::cout << "Enter hex seed (e.g. C02E7A5B3F91A8C3), or press Enter: ";

    std::string input;
    std::getline(std::cin, input);

    if (input.empty())
    {
        return default_seed;
    }

    uint64_t user_seed = 0;
    try
    {
        user_seed = std::stoull(input, nullptr, 16);
    } catch (...)
    {
        std::cerr << "Invalid input, fallback to default seed.\n";
        return default_seed;
    }

    return splitmix64(user_seed);
}

uint64_t parse_seed(const std::string& input)
{
    uint64_t user_seed = 0;
    try
    {
        user_seed = std::stoull(input, nullptr, 16);
    } catch (...)
    {
        throw std::runtime_error("Invalid hex seed: " + input);
    }

    return splitmix64(user_seed);
}

void print_usage(const char* program_name)
{
    std::cout
        << "Usage: " << program_name << " [--config PATH] [--seed HEX]\n"
        << "       " << program_name
        << " [-c PATH] [-s HEX] [--reference-eval]\n"
        << "\n"
        << "Defaults:\n"
        << "  config: " << kDefaultConfigPath << "\n"
        << "  seed: ask interactively, then fallback to built-in default on empty input\n"
        << "  reference evaluation: controlled by config unless --reference-eval is set\n";
}

CommandLineOptions parse_args(const int argc, char* argv[])
{
    CommandLineOptions options;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(0);
        }

        if (arg == "--config" || arg == "-c")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(arg + " requires a path argument");
            }
            options.config_path = argv[++i];
            continue;
        }

        if (arg == "--seed" || arg == "-s")
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(arg + " requires a hex seed argument");
            }
            options.seed = parse_seed(argv[++i]);
            continue;
        }

        if (arg == "--reference-eval")
        {
            options.reference_evaluation = true;
            continue;
        }

        throw std::runtime_error("Unknown argument: " + arg);
    }

    return options;
}

int main(const int argc, char* argv[])
{
    CommandLineOptions options;
    try
    {
        options = parse_args(argc, argv);
    } catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n\n";
        print_usage(argv[0]);
        return 1;
    }

    const uint64_t seed = options.seed.has_value() ? options.seed.value() : get_seed();

    const auto t_start = std::chrono::high_resolution_clock::now();

    force_link_all_equations();
    const Config cfg = load_config(options.config_path);
    const auto device = torch::cuda::is_available()?torch::kCUDA:torch::kCPU;
    const auto pde = EquationFactory::instance().create(cfg.eqn_config.equation_name, cfg.eqn_config);

    auto rfm_solver = RFMSolver(cfg, pde, device, seed);
    rfm_solver.options(std::nullopt, std::nullopt, std::nullopt);

    if (rfm_solver.is_linear())
    {
        rfm_solver.linear_options(get_linear_solver_options_from_config(
            cfg.solver_config
        ));
    }

    if (torch::cuda::is_available()) torch::cuda::synchronize();

    const auto [y0, beta, rmse] = rfm_solver.solve(true);
    const float test_mse = rfm_solver.test(y0, beta);
    const auto& diagnostics = rfm_solver.diagnostics();
    auto reference_options = cfg.solver_config.reference_evaluation;
    reference_options.enabled =
        reference_options.enabled || options.reference_evaluation;
    if (reference_options.enabled)
    {
        if (pde->has_reference_solution())
        {
            evaluate_reference_solution(
                rfm_solver,
                *pde,
                y0,
                beta,
                reference_options,
                seed
            );
        }
        else
        {
            std::cout
                << "reference evaluation unavailable for equation: "
                << cfg.eqn_config.equation_name
                << std::endl;
        }
    }

    if (torch::cuda::is_available()) torch::cuda::synchronize();

    const auto t_end = std::chrono::high_resolution_clock::now();
    const float elapsed =
        std::chrono::duration<float, std::milli>(t_end - t_start).count();

    std::cout << "y0 = " << y0.item<float>() << std::endl;
    std::cout << "rmse = " << rmse << std::endl;
    std::cout << "test rmse = " << test_mse << std::endl;
    std::cout << "test terminal std = "
              << diagnostics.test_terminal_std << std::endl;
    std::cout << "normalized test rmse = "
              << diagnostics.normalized_test_rmse << std::endl;
    std::cout << "explained terminal variance = "
              << diagnostics.explained_terminal_variance << std::endl;
    std::cout << "beta norm = " << diagnostics.beta_norm << std::endl;
    if (!rfm_solver.is_linear())
    {
        std::cout << "objective rmse = "
                  << diagnostics.objective_rmse << std::endl;
        std::cout << "final gradient inf norm = "
                  << diagnostics.final_gradient_inf_norm << std::endl;
        std::cout << "accepted lm iterations = "
                  << diagnostics.accepted_lm_iterations << std::endl;
        std::cout << "final damping = "
                  << diagnostics.final_damping << std::endl;
    }
    std::cout << "dtype: " << beta.dtype() << std::endl;
    std::cout << "eqn dim: " << cfg.eqn_config.dimension << std::endl;
    std::cout << "hidden dim: " << cfg.solver_config.hidden_dim << std::endl;
    std::cout << "samples num: " << cfg.solver_config.sample_size << std::endl;
    std::cout << "test samples num: " << cfg.solver_config.test_sample_size << std::endl;
    std::cout << "total time: " << elapsed << " ms" << std::endl;
    std::cout << "device: " << device << std::endl;

    return 0;
}
