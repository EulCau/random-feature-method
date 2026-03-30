# include "rfm_solver.h"
#include "config.h"
#include "equation_factory.h"
#include "register_all_eqn.h"
#include <iostream>
#include <chrono>

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

int main()
{
    const uint64_t seed = get_seed();

    const auto t_start = std::chrono::high_resolution_clock::now();

    force_link_all_equations();
    const Config cfg = load_config("config/bsm_d100.json");
    const auto device = torch::cuda::is_available()?torch::kCUDA:torch::kCPU;
    const auto pde = EquationFactory::instance().create(cfg.eqn_config.eqn_name, cfg.eqn_config);

    const auto rfm_solver = RFMSolver(cfg, pde, device, seed);

    if (torch::cuda::is_available()) torch::cuda::synchronize();

    const auto t_mid = std::chrono::high_resolution_clock::now();

    const auto [y0, alpha, rmse] = rfm_solver.Solve_linear();

    if (torch::cuda::is_available()) torch::cuda::synchronize();

    const auto t_end = std::chrono::high_resolution_clock::now();
    const float elapsed =
        std::chrono::duration<float, std::milli>(t_end - t_start).count();
    const float elapsed_repeat =
        std::chrono::duration<float, std::milli>(t_end - t_mid).count();

    std::cout << "y0 = " << y0.item<float>() << std::endl;
    std::cout << "rmse = " << rmse << std::endl;
    std::cout << "dtype: " << alpha.dtype() << std::endl;
    std::cout << "eqn dim: " << cfg.eqn_config.dim << std::endl;
    std::cout << "hidden dim: " << cfg.net_config.num_hiddens[0] << std::endl;
    std::cout << "samples num: " << cfg.net_config.valid_size << std::endl;
    std::cout << "total time: " << elapsed << " ms" << std::endl;
    std::cout << "time which will repeat: " << elapsed_repeat << " ms" << std::endl;
    std::cout << "device: " << device << std::endl;

    return 0;
}
