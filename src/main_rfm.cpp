# include "rfm_solver.h"
#include "config.h"
#include "equation_factory.h"
#include "register_all_eqn.h"
#include <iostream>

int main()
{
    constexpr uint64_t seed = 42;
    force_link_all_equations();
    const Config cfg = load_config("bsm_d100.json");
    const auto pde = EquationFactory::instance().create(cfg.eqn_config.eqn_name, cfg.eqn_config);

    const auto rfm_solver = RFMSolver(cfg, pde, seed);

    const auto [y0, alpha] = rfm_solver.Solve();

    std::cout << y0 << std::endl;

    return 0;
}
