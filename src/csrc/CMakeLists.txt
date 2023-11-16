cmake_minimum_required(VERSION 3.12)
project(paraformer.cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")

set(SOURCE_FILES
        paraformer-contextual-offline.h
        paraformer-contextual-offline.cpp
        )

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()


include_directories(${PROJECT_SOURCE_DIR}/third-party/ggml/include/ggml ${PROJECT_SOURCE_DIR}/third-party/ggml/src)
add_subdirectory(third-party/ggml)

add_library(paraformer-contextual-offline STATIC paraformer-contextual-offline.cpp)

add_executable(model_test test/model_load_test.cpp)
target_link_libraries(paraformer-contextual-offline PRIVATE ggml)
target_link_libraries(model_test PRIVATE paraformer-contextual-offline)