add_library(
    solver_utils
    STATIC
    nonlinear_solver_utils.cpp
    qr_decomposition.cpp)

target_include_directories(
    solver_utils
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    solver_utils
    PUBLIC
    ${TORCH_LIBRARIES})
