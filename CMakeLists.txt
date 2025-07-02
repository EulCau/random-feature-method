cmake_minimum_required(VERSION 3.10)
project(bsde_solver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
	add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
	set_target_properties(
		CUDA::nvToolsExt PROPERTIES
		INTERFACE_LINK_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvToolsExt.lib"
		INTERFACE_INCLUDE_DIRECTORIES "${CUDA_TOOLKIT_ROOT_DIR}/include")
endif()

find_package(Torch REQUIRED)

add_subdirectory(src)

set_target_properties(
	bsde_solver PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(TORCH_DLL_DIR "$ENV{TORCH_DIR}/lib")
file(GLOB TORCH_DLLS "${TORCH_DLL_DIR}/*.dll")
foreach(dll ${TORCH_DLLS})
	add_custom_command(
		TARGET bsde_solver POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${dll}"
		$<TARGET_FILE_DIR:bsde_solver>)
endforeach()

file(GLOB CONFIG_FILES "${CMAKE_SOURCE_DIR}/configs/*.json")
foreach(cfg ${CONFIG_FILES})
	add_custom_command(
		TARGET bsde_solver POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${cfg}"
		$<TARGET_FILE_DIR:bsde_solver>)
endforeach()
