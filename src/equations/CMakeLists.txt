add_subdirectory(PDEs)

target_include_directories(
	equations PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
	${TORCH_INCLUDE_DIRS})
