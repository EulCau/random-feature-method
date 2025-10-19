add_subdirectory(third_party)
add_subdirectory(config)
add_subdirectory(equations)
add_subdirectory(mlp)
add_subdirectory(rff)
add_subdirectory(model)
add_subdirectory(solver)
set(TORCH_DLL_DIR "$ENV{TORCH_DIR}/lib")
file(GLOB TORCH_DLLS "${TORCH_DLL_DIR}/*.dll")
file(GLOB CONFIG_FILES "${CMAKE_SOURCE_DIR}/configs/*.json")

add_executable(bsde_solver main_bsde.cpp)

target_link_libraries(
	bsde_solver
	PRIVATE
	config
	equations
	solver
	${TORCH_LIBRARIES})

target_include_directories(
	bsde_solver
	PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${TORCH_INCLUDE_DIRS})

set_target_properties(
	bsde_solver PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

foreach(dll ${TORCH_DLLS})
	add_custom_command(
		TARGET bsde_solver POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${dll}"
		$<TARGET_FILE_DIR:bsde_solver>)
endforeach()

foreach(cfg ${CONFIG_FILES})
	add_custom_command(
		TARGET bsde_solver POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${cfg}"
		$<TARGET_FILE_DIR:bsde_solver>)
endforeach()


add_executable(rfm_solver main_rfm.cpp)

target_link_libraries(
		rfm_solver
		PRIVATE
		config
		equations
		solver
		${TORCH_LIBRARIES})

target_include_directories(
		rfm_solver
		PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
		${TORCH_INCLUDE_DIRS})

set_target_properties(
		rfm_solver PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

foreach(dll ${TORCH_DLLS})
	add_custom_command(
			TARGET rfm_solver POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${dll}"
			$<TARGET_FILE_DIR:rfm_solver>)
endforeach()

foreach(cfg ${CONFIG_FILES})
	add_custom_command(
			TARGET rfm_solver POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${cfg}"
			$<TARGET_FILE_DIR:rfm_solver>)
endforeach()
