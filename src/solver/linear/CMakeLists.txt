add_library(
    solver_linear
    STATIC
    linear_least_squares_solver.cpp
    linear_solver_cli.cpp)

target_include_directories(
    solver_linear
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    solver_linear
    PUBLIC
    ${TORCH_LIBRARIES})
