add_library(
    solver_nonlinear
    STATIC
    nonlinear_solver_utils.cpp)

target_include_directories(
    solver_nonlinear
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    solver_nonlinear
    PUBLIC
    ${TORCH_LIBRARIES})
