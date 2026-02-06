cmake_minimum_required(VERSION 4.0)
project(project C CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(glfw3 3.3 REQUIRED)

add_executable(project
  src/main.cpp
  src/glad.c
)

target_include_directories(project PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(project PRIVATE
  glfw
  "-framework OpenGL"
  "-framework Cocoa"
  "-framework IOKit"
  "-framework CoreVideo"

)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(TON_TARGET PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(TON_TARGET PRIVATE USE_OMP=1)
endif()

target_sources(project PRIVATE
    include/imgui/imgui.cpp
    include/imgui/imgui_draw.cpp
    include/imgui/imgui_tables.cpp
    include/imgui/imgui_widgets.cpp
    include/imgui/backends/imgui_impl_glfw.cpp
    include/imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories(project PRIVATE
    include/imgui
    include/imgui/backends
)

if(APPLE)

    set(LIBOMP_DIR /opt/homebrew/opt/libomp)
    find_library(LIBOMP libomp.dylib HINTS ${LIBOMP_DIR}/lib)
    if(LIBOMP)
        message(STATUS "Found libomp: ${LIBOMP}")
        target_compile_options(project PUBLIC -Xpreprocessor -fopenmp)
        target_link_libraries(project PUBLIC ${LIBOMP})
        target_include_directories(project PUBLIC ${LIBOMP_DIR}/include)
    else()
        message(FATAL_ERROR "libomp not found. Install with: brew install libomp")
    endif()
endif()
