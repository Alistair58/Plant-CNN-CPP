cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#Enable maximum optimization and native vector instructions
add_compile_options(-mavx2) 
#-O3 -march=native -ffast-math -funroll-loops 
#Takes a very long time as it goes through all libraries e.g. vector.h and isn't particularly effective 
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

target_compile_definitions(Plant-CNN-CPP PRIVATE
    DEBUG=${DEBUG}
)
if(${DEBUG})
    set(CMAKE_BUILD_TYPE Debug)
endif()

target_link_libraries(Plant-CNN-CPP
        )

target_include_directories(Plant-CNN-CPP PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src
        ${CMAKE_CURRENT_LIST_DIR}/lib
)



