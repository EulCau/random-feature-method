# include "rfm_solver.h"
#include "config.h"
#include "equation_factory.h"
#include "register_all_eqn.h"
#include <iostream>

int main()
{
    uint64_t seed = 42;
    force_link_all_equations();
    const Config cfg = load_config("***"); // TODO: linear equation
    const auto pde = EquationFactory::instance().create(cfg.eqn_config.eqn_name, cfg.eqn_config);

    auto rfm_solver = RFMSolver(cfg, pde, seed);

    float result = 0.0f;
    std::cout << result << std::endl;

    return 0;
}
