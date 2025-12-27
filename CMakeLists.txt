cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


project(Plant-CNN-CPP CXX)
add_executable(Plant-CNN-CPP
    src/main.cpp
    src/cnn.cpp
    src/cnnutils.cpp
    src/dataset.cpp
    src/plantimage.cpp
    src/utils.cpp
    src/tensor.cpp
    src/imageutils.cpp
)


if(${DEBUG})
    set(CMAKE_BUILD_TYPE Debug)
    target_compile_definitions(Plant-CNN-CPP PRIVATE DEBUG=${DEBUG})
    target_compile_options(Plant-CNN-CPP PRIVATE
    -mavx2
    -march=native
    ) 
elseif(${PROFILING})
    target_compile_definitions(Plant-CNN-CPP PRIVATE DEBUG=0 PROFILING=${PROFILING})
    target_compile_options(Plant-CNN-CPP PRIVATE
        -O3
        -march=native
        -mavx2
        -ffast-math
        -funroll-loops
        -mfma
        -ffp-contract=fast
    )
elseif(${ASAN})
    set(CMAKE_BUILD_TYPE Debug)

    target_compile_definitions(Plant-CNN-CPP PRIVATE
        DEBUG=0
        PROFILING=0
    )

    target_compile_options(Plant-CNN-CPP PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
        -O1
        -march=native
        -mavx2
    )

    target_link_options(Plant-CNN-CPP PRIVATE
        -fsanitize=address,undefined
    )
else()
    target_compile_definitions(Plant-CNN-CPP PRIVATE DEBUG=0 PROFILING=0)
    target_compile_options(Plant-CNN-CPP PRIVATE
        -O3
        -march=native
        -mavx2
        -ffast-math
        -funroll-loops
        -mfma
        -ffp-contract=fast
        
    )
endif()


target_compile_definitions(Plant-CNN-CPP PRIVATE
     ${EXTRA_FLAGS} 
)



target_include_directories(Plant-CNN-CPP PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src
        ${CMAKE_CURRENT_LIST_DIR}/lib
)



