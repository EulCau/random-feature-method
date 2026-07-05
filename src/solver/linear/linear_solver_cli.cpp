#include "linear_solver_cli.h"

#include <iostream>
#include <string>

namespace
{

constexpr int default_choice = 1;

int read_choice(const std::string& prompt)
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

int64_t read_positive_int64(const std::string& prompt, const int64_t default_value)
{
    std::cout << prompt;

    std::string input;
    std::getline(std::cin, input);

    if (input.empty())
    {
        return default_value;
    }

    try
    {
        if (const int64_t value = std::stoll(input); value > 0)
        {
            return value;
        }
    } catch (...)
    {
    }

    std::cerr << "Invalid input, fallback to default value " << default_value << ".\n";
    return default_value;
}

} // namespace

LinearSolverOptions get_linear_solver_options_from_terminal(
    const int64_t default_qr_batch_size,
    const double ridge_lambda)
{
    LinearSolverOptions options;
    options.qr_batch_size = default_qr_batch_size;
    options.ridge_lambda = ridge_lambda;

    //TODO: Replace terminal-only solver selection with a configurable registry or factory.
    const int solver_choice = read_choice(
        "Select linear solver: [1] ridge dual [2] QR [3] batched QR, or press Enter for 1: "
    );

    if (solver_choice == 2)
    {
        options.solver_type = LinearSolverType::QR;
    }
    else if (solver_choice == 3)
    {
        options.solver_type = LinearSolverType::BatchedQR;
        options.qr_batch_size = read_positive_int64(
            "Enter batched QR batch size, or press Enter for "
                + std::to_string(default_qr_batch_size) + ": ",
            default_qr_batch_size
        );
    }
    else if (solver_choice != 1)
    {
        std::cerr << "Unknown linear solver, fallback to ridge dual.\n";
    }

    return options;
}
