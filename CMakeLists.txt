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
