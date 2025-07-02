add_subdirectory(third_party)
add_subdirectory(config)
add_subdirectory(equations)
add_subdirectory(solver)
add_subdirectory(model)
add_subdirectory(mlp)

add_executable(bsde_solver main_bsde.cpp)

target_link_libraries(
	bsde_solver
	PRIVATE
		config
		equations
		solver
		model
		mlp
		${TORCH_LIBRARIES})

target_include_directories(
	bsde_solver
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
		${TORCH_INCLUDE_DIRS})
