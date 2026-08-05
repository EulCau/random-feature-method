target_sources(
    solver
    PRIVATE
    internal_solution_validator.cpp)

target_include_directories(
    solver
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})
