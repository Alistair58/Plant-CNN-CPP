cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(plantcnn CXX)

add_executable(plantcnn
    main.cpp
    cnn.cpp 
    cnnutils.cpp
    dataset.cpp 
    plantimage.cpp
)


target_link_libraries(plantcnn
        )

target_include_directories(plantcnn PRIVATE
        ${CMAKE_CURRENT_LIST_DIR/src}
        ${CMAKE_CURRENT_LIST_DIR/lib}
)



