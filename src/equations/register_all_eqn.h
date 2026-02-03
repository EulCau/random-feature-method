#pragma once
extern "C" void force_link_AllenCahn();
extern "C" void force_link_HJBLQ();
extern "C" void force_link_BSM();
// TODO: linear equation

inline void force_link_all_equations() {
	force_link_AllenCahn();
	force_link_HJBLQ();
	force_link_BSM();
}
