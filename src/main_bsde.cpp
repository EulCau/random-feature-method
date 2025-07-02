#include "bsde_solver.h"
#include "config.h"
#include "equation_factory.h"
#include "register_all_eqn.h"
#include <iostream>

int main()
{
	force_link_all_equations();
	const Config cfg = load_config("hjb_lq_d100.json");

	const auto bsde = EquationFactory::instance().create(cfg.eqn_config.eqn_name, cfg.eqn_config);
	BSDESolver solver(cfg, bsde);
	std::cout << "Starting training..." << std::endl;
	solver.train();

	return 0;
}
