add_library(
	solver
	STATIC
	bsde_solver.cpp
	rfm_solver.cpp)

target_include_directories(
	solver
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
	solver
	PUBLIC
	model
	config
	equations
	${TORCH_LIBRARIES})
