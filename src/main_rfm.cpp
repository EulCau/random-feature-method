# include "rfm_solver.h"
#include "config.h"
#include "equation_factory.h"
#include "register_all_eqn.h"
#include <iostream>

int main()
{
    force_link_all_equations();
    const Config cfg = load_config("***"); // TODO: linear equation

    const auto pde = EquationFactory::instance().create(cfg.eqn_config.eqn_name, cfg.eqn_config);
    auto result = solve(cfg, pde);
    std::cout << result.alpha << std::endl;
    std::cout << result.terminal_err << std::endl;

    return 0;
}
