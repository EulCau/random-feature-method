# include "rfm_solver.h"
#include "config.h"
#include "equation_factory.h"
#include "register_all_eqn.h"
#include <iostream>

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
    force_link_all_equations();
    const Config cfg = load_config("bsm_d100.json");
    const auto pde = EquationFactory::instance().create(cfg.eqn_config.eqn_name, cfg.eqn_config);

    const auto rfm_solver = RFMSolver(cfg, pde, seed);

    const auto [y0, alpha] = rfm_solver.Solve();

    std::cout << "y0 = \n" << y0.data() << std::endl;
    // std::cout << "alpha = \n" << alpha.data() << std::endl;

    return 0;
}
