add_library(
	rff
	STATIC
	rff.cpp)

target_include_directories(
	rff
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
	rff
	PUBLIC
	config
	${TORCH_LIBRARIES})
