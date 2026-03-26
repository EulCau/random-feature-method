add_subdirectory(third_party)
add_subdirectory(config)
add_subdirectory(equations)
add_subdirectory(rff)
add_subdirectory(solver)

set(BIN_DIR "${CMAKE_SOURCE_DIR}/bin")

# 检查 Torch 运行时目录
set(TORCH_RUNTIME_DIR "$ENV{TORCH_DIR}/lib")
if(NOT EXISTS "${TORCH_RUNTIME_DIR}")
    message(FATAL_ERROR "TORCH runtime dir not found: ${TORCH_RUNTIME_DIR}")
endif()

add_executable(rfm_solver main_rfm.cpp)

target_link_libraries(
    rfm_solver
    PRIVATE
    config
    equations
    solver
    ${TORCH_LIBRARIES}
)

target_include_directories(
    rfm_solver
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

set_target_properties(
    rfm_solver
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}"
)

# 设置 Linux 运行时库搜索路径
if(UNIX AND NOT APPLE)
    set_target_properties(
        rfm_solver
        PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "\$ORIGIN/lib"
        INSTALL_RPATH_USE_LINK_PATH FALSE
    )
endif()

# 创建 lib 和 config 目录
add_custom_command(
    TARGET rfm_solver POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:rfm_solver>/lib"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:rfm_solver>/config"
    VERBATIM
)

# 复制配置文件
file(GLOB CONFIG_FILES "${CMAKE_SOURCE_DIR}/config/*.json")
if(CONFIG_FILES)
    add_custom_command(
        TARGET rfm_solver POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CONFIG_FILES}
                "$<TARGET_FILE_DIR:rfm_solver>/config"
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
endif()

# 复制 Torch 运行时库
file(GLOB TORCH_DLLS "${TORCH_RUNTIME_DIR}/*.dll")
if(WIN32 AND TORCH_DLLS)
    add_custom_command(
        TARGET rfm_solver POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                "$<TARGET_FILE_DIR:rfm_solver>/lib"
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
endif()

file(GLOB TORCH_SOS "${TORCH_RUNTIME_DIR}/*.so*")
if(UNIX AND NOT APPLE AND TORCH_SOS)
    add_custom_command(
        TARGET rfm_solver POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_SOS}
                "$<TARGET_FILE_DIR:rfm_solver>/lib"
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
endif()
