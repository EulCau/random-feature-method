add_library(
	mlp
	STATIC
	mlp.cpp)

target_include_directories(
	mlp
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
	mlp
	PUBLIC
	config
	${TORCH_LIBRARIES})
