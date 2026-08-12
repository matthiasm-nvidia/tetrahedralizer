cmake_minimum_required(VERSION 3.20)
project(tetrahedralizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glfw)

add_executable(tetrahedralizer
    src/main.cpp
    src/FileDialog.cpp
    src/GlCore.cpp
    src/TriMesh.cpp
    src/TriMeshRenderer.cpp
    src/Tetrahedralizer.cpp
    src/TetMeshRenderer.cpp
    src/Camera.cpp
    src/LastPath.cpp
    src/viewer_imgui_theme.cpp
    src/viewer_imgui_widgets.cpp
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_impl_glfw.cpp
)

target_include_directories(tetrahedralizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(tetrahedralizer PRIVATE glfw)

if(WIN32)
    target_link_libraries(tetrahedralizer PRIVATE opengl32 glu32 shell32 ole32)
elseif(APPLE)
    target_link_libraries(tetrahedralizer PRIVATE "-framework OpenGL")
else()
    target_link_libraries(tetrahedralizer PRIVATE GL GLU)
endif()

add_custom_command(TARGET tetrahedralizer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:tetrahedralizer>/resources/fonts
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/resources/fonts
            $<TARGET_FILE_DIR:tetrahedralizer>/resources/fonts
    COMMENT "Copy UI fonts next to tetrahedralizer executable")
