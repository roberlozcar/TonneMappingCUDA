cmake_minimum_required(VERSION 3.10)

# set the project name
project(ToneMapping)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB_RECURSE opencvhpp "${CMAKE_SOURCE_DIR}/include/opencv2/*.hpp")
file(GLOB_RECURSE librerias "${CMAKE_SOURCE_DIR}/libs/*.lib")

add_executable(ToneMapping 
"${CMAKE_SOURCE_DIR}/scr/main.cpp"
"${CMAKE_SOURCE_DIR}/scr/funcHDR.cu"
"${CMAKE_SOURCE_DIR}/scr/loadSaveImage.cpp"
"${CMAKE_SOURCE_DIR}/scr/loadSaveImage.h"
"${CMAKE_SOURCE_DIR}/scr/timer.h"
"${CMAKE_SOURCE_DIR}/scr/tonemapping.cu"
${opencvhpp}
)

target_link_libraries(ToneMapping ${librerias})

target_include_directories(ToneMapping PUBLIC 
"${CMAKE_SOURCE_DIR}/include"
"${CMAKE_SOURCE_DIR}/libs"
)