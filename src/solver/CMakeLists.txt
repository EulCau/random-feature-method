add_library(
    solver
    STATIC
    rfm_solver.cpp
    utils/nonlinear_solver_utils.cpp
    utils/qr_decomposition.cpp)

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
    ${TORCH_LIBRARIES})
