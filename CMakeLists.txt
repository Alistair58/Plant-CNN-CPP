cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
#Enable maximum optimization and native vector instructions
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-O3 -march=native -ffast-math -funroll-loops -fopt-info-vec)
endif()
message(STATUS "C++ Compiler Flags: ${CMAKE_CXX_FLAGS}")
project(Plant-CNN-CPP CXX)

add_executable(Plant-CNN-CPP
    src/main.cpp
    src/cnn.cpp 
    src/cnnutils.cpp
    src/dataset.cpp 
    src/plantimage.cpp
    src/utils.cpp
    src/tensor.cpp
)


target_link_libraries(Plant-CNN-CPP
        )

target_include_directories(Plant-CNN-CPP PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src
        ${CMAKE_CURRENT_LIST_DIR}/lib
)



