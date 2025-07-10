       cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(plantcnn CXX)

include(FetchContent)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
)
FetchContent_MakeAvailable(json)

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



