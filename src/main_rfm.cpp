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

struct LinearSolveSelection
{
    LinearSolverType solver_type{LinearSolverType::RidgeDual};
    solver_utils::QRMethod qr_method{solver_utils::QRMethod::Householder};
};
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

int read_choice(const std::string& prompt, const int default_choice)
{
    std::cout << prompt;

    std::string input;
    std::getline(std::cin, input);

    if (input.empty())
    {
        return default_choice;
    }

    try
    {
        return std::stoi(input);
    } catch (...)
    {
        std::cerr << "Invalid input, fallback to default choice " << default_choice << ".\n";
        return default_choice;
    }
}

LinearSolveSelection get_linear_solve_selection()
{
    LinearSolveSelection selection;

    //TODO: Replace terminal-only solver selection with a configurable registry or factory.
    const int solver_choice = read_choice(
        "Select linear solver: [1] ridge dual [2] QR, or press Enter for 1: ",
        1
    );

    if (solver_choice == 2)
    {
        selection.solver_type = LinearSolverType::QR;

        const int qr_choice = read_choice(
            "Select QR method: [1] Householder [2] Givens, or press Enter for 1: ",
            1
        );

        if (qr_choice == 2)
        {
            selection.qr_method = solver_utils::QRMethod::Givens;
        }
        else if (qr_choice != 1)
        {
            std::cerr << "Unknown QR method, fallback to Householder.\n";
        }
    }
    else if (solver_choice != 1)
    {
        std::cerr << "Unknown linear solver, fallback to ridge dual.\n";
    }

    return selection;
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
        const auto selection = get_linear_solve_selection();
        rfm_solver.linear_options(selection.solver_type, selection.qr_method);
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
    std::cout << "total time: " << elapsed << " ms" << std::endl;
    std::cout << "device: " << device << std::endl;

    return 0;
}
