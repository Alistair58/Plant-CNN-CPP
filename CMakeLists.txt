cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


project(Plant-CNN-CPP CXX)
add_executable(Plant-CNN-CPP
    src/cnn.cpp
    src/cnnutils.cpp
    src/dataset.cpp
    src/plantimage.cpp
    src/utils.cpp
    src/tensor.cpp
    src/imageutils.cpp
    src/globals.cpp
)


if(${DEBUG})
    target_sources(Plant-CNN-CPP PRIVATE 
        src/main.cpp
    )
    set(CMAKE_BUILD_TYPE Debug)
    target_compile_definitions(Plant-CNN-CPP PRIVATE DEBUG=${DEBUG})
    target_compile_options(Plant-CNN-CPP PRIVATE
    -mavx2
    -march=native
    ) 
elseif(${PROFILING})
    target_sources(Plant-CNN-CPP PRIVATE 
        src/main.cpp
    )

    set(CMAKE_BUILD_TYPE Release)
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
    target_sources(Plant-CNN-CPP PRIVATE 
        src/main.cpp
    )
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
elseif(${TEST})
    target_sources(Plant-CNN-CPP PRIVATE 
        test/cnnutilstest.cpp
        test/tensortest.cpp
        test/cnntest.cpp
    )
    set(CMAKE_BUILD_TYPE Debug)

    target_compile_definitions(Plant-CNN-CPP PRIVATE
        DEBUG=0
        PROFILING=0
    )

    include(FetchContent)
    FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    
    target_link_libraries(Plant-CNN-CPP
        GTest::gtest_main
        GTest::gmock
    )

    include(GoogleTest)
    gtest_discover_tests(Plant-CNN-CPP)

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

    target_include_directories(Plant-CNN-CPP PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/test
    )

else()
    target_sources(Plant-CNN-CPP PRIVATE 
        src/main.cpp
    )
    set(CMAKE_BUILD_TYPE Release)

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



