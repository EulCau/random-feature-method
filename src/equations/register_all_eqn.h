extern "C" void force_link_AllenCahn();
extern "C" void force_link_HJBLQ();

void force_link_all_equations() {
	force_link_AllenCahn();
	force_link_HJBLQ();
}
