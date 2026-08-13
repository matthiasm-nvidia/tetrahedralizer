cmake_minimum_required(VERSION 3.20)

option(TETRAHEDRALIZER_ENABLE_CUDA "Build CUDA tetrahedralization support" ON)

if(MSVC AND TETRAHEDRALIZER_ENABLE_CUDA)
    set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler")
    set(CMAKE_CUDA_COMPILER_ID_FLAGS "-allow-unsupported-compiler")
    set(ENV{NVCC_PREPEND_FLAGS} "-allow-unsupported-compiler")
endif()

project(tetrahedralizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

set(TETRAHEDRALIZER_CUDA_SOURCES "")
set(TETRAHEDRALIZER_CUDA_LIBS "")

if(TETRAHEDRALIZER_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 86-real 120-real 90-virtual)
        endif()
        find_package(CUDAToolkit REQUIRED)
        set(TETRAHEDRALIZER_CUDA_SOURCES
            src/utils/CudaUtils.cpp
            src/utils/GpuBVH.cu
            src/tetrahedralizers/GpuTetrahedralizer.cu
        )
        set(TETRAHEDRALIZER_CUDA_LIBS CUDA::cudart)
        set(TETRAHEDRALIZER_HAS_CUDA ON)
        message(STATUS "CUDA tetrahedralization enabled")
    else()
        message(WARNING "CUDA toolkit not found; building without tetrahedralization")
    endif()
endif()

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
    ${TETRAHEDRALIZER_CUDA_SOURCES}
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

target_link_libraries(tetrahedralizer PRIVATE glfw ${TETRAHEDRALIZER_CUDA_LIBS})

if(TETRAHEDRALIZER_HAS_CUDA)
    target_compile_definitions(tetrahedralizer PRIVATE TETRAHEDRALIZER_HAS_CUDA)
    if(MSVC)
        target_compile_options(tetrahedralizer PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    endif()
endif()

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
