add_library(
    equations STATIC
    allencahn.cpp
    hjblq.cpp)

target_include_directories(equations PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
	equations PRIVATE
	config
	${TORCH_LIBRARIES})
