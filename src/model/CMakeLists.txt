add_library(model STATIC
	non_shared_model.cpp)

target_include_directories(model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
