# include "rfm_solver.h"
#include "config.h"
#include "equation_factory.h"
#include "register_all_eqn.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

namespace
{
constexpr auto kDefaultConfigPath = "config/hjb_lq_d100.json";

struct CommandLineOptions
{
    std::string config_path{kDefaultConfigPath};
    std::optional<uint64_t> seed;
};

int64_t default_batch_size(const int64_t dimension, const int64_t hidden_dim)
{
    return 4 * (1 + dimension * hidden_dim);
}

LinearSolverOptions get_linear_solver_options_from_config(
    const SolverConfig& config,
    const EqnConfig& eqn_config)
{
    LinearSolverOptions options;
    TORCH_CHECK(config.linear.batch_size >= 0, "linear batch_size must be nonnegative");
    options.qr_batch_size = config.linear.batch_size == 0
        ? default_batch_size(eqn_config.dimension, config.hidden_dim)
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
        << "       " << program_name << " [-c PATH] [-s HEX]\n"
        << "\n"
        << "Defaults:\n"
        << "  config: " << kDefaultConfigPath << "\n"
        << "  seed: ask interactively, then fallback to built-in default on empty input\n";
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
            cfg.solver_config,
            cfg.eqn_config
        ));
    }

    if (torch::cuda::is_available()) torch::cuda::synchronize();

    const auto [y0, alpha, rmse] = rfm_solver.solve(true);
    const float test_mse = rfm_solver.test(y0, alpha);

    if (torch::cuda::is_available()) torch::cuda::synchronize();

    const auto t_end = std::chrono::high_resolution_clock::now();
    const float elapsed =
        std::chrono::duration<float, std::milli>(t_end - t_start).count();

    std::cout << "y0 = " << y0.item<float>() << std::endl;
    std::cout << "rmse = " << rmse << std::endl;
    std::cout << "test rmse = " << test_mse << std::endl;
    std::cout << "dtype: " << alpha.dtype() << std::endl;
    std::cout << "eqn dim: " << cfg.eqn_config.dimension << std::endl;
    std::cout << "hidden dim: " << cfg.solver_config.hidden_dim << std::endl;
    std::cout << "samples num: " << cfg.solver_config.sample_size << std::endl;
    std::cout << "test samples num: " << cfg.solver_config.test_sample_size << std::endl;
    std::cout << "total time: " << elapsed << " ms" << std::endl;
    std::cout << "device: " << device << std::endl;

    return 0;
}
