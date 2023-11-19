cmake_minimum_required(VERSION 3.12)
project(paraformer.cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")

set(SOURCE_FILES
        paraformer-contextual-offline.cpp
        paraformer-frontend.cpp
        )

option(PARAFORMER_BUILD_TESTS "paraformer: build tests" ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()


include_directories(${PROJECT_SOURCE_DIR}/third-party/ggml/include/ggml ${PROJECT_SOURCE_DIR}/third-party/ggml/src)
add_subdirectory(third-party/ggml)
add_library(paraformer-core STATIC ${SOURCE_FILES})

if (PARAFORMER_BUILD_TESTS)
    message(DEBUG "Test scripts")
    add_subdirectory(test)
endif ()

target_link_libraries(paraformer-core PRIVATE ggml)

