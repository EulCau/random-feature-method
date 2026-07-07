#pragma once
extern "C" void force_link_AllenCahn();
extern "C" void force_link_AsymmetricHeat();
extern "C" void force_link_HJBLQ();
extern "C" void force_link_BSM();
extern "C" void force_link_Heat();

inline void force_link_all_equations() {
	force_link_AllenCahn();
	force_link_AsymmetricHeat();
	force_link_HJBLQ();
	force_link_BSM();
	force_link_Heat();
}
