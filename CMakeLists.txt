cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#Enable maximum optimization and native vector instructions

#Takes a very long time as it goes through all libraries e.g. vector.h and isn't particularly effective 

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

if(${DEBUG})
    set(CMAKE_BUILD_TYPE Debug)
    target_compile_definitions(Plant-CNN-CPP PRIVATE DEBUG=1)
    target_compile_options(Plant-CNN-CPP PRIVATE 
    -mavx2
    ) 
    target_link_options(Plant-CNN-CPP PRIVATE 
    )
else()    
    target_compile_definitions(Plant-CNN-CPP PRIVATE DEBUG=0)
    target_compile_options(Plant-CNN-CPP PRIVATE
        -mavx2
        -O3
        -ffast-math
        -funroll-loops
        -march=native
        
    )
endif()


target_compile_definitions(Plant-CNN-CPP PRIVATE
     ${EXTRA_FLAGS} 
)


target_link_libraries(Plant-CNN-CPP
        )

target_include_directories(Plant-CNN-CPP PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src
        ${CMAKE_CURRENT_LIST_DIR}/lib
)



