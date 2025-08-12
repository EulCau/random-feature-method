cmake_minimum_required(VERSION 3.18)
project(bsde_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit REQUIRED)

find_package(Torch REQUIRED)

add_subdirectory(src)
