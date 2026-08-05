add_subdirectory(linear)
add_subdirectory(nonlinear)

add_library(
    solver
    STATIC
    rfm_solver.cpp)

add_subdirectory(utils)

target_include_directories(
    solver
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    solver
    PUBLIC
    config
    equations
    rff
    solver_linear
    solver_nonlinear
    ${TORCH_LIBRARIES})
