extern "C" void force_link_AllenCahn();
extern "C" void force_link_HJBLQ();  // TODO: linear equation

void force_link_all_equations() {
	force_link_AllenCahn();
	force_link_HJBLQ();
}
