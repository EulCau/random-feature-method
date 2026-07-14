add_library(
    equations
    STATIC
    allencahn.cpp
    asymmetric_allencahn.cpp
    asymmetric_heat.cpp
    asymmetric_hjblq.cpp
    hjblq.cpp
    bsm.cpp
    heat.cpp)

target_include_directories(
    equations
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    equations
    PUBLIC
    config
    ${TORCH_LIBRARIES})
