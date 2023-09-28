cmake_minimum_required(VERSION 3.25)
project(paraformer_cpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(paraformer_cpp main.cpp backend/ggml/ggml.c backend/ggml/ggml.h backend/ggml/paraformer.h backend/ggml/paraformer.cpp backend/ggml/ggml-alloc.h backend/ggml/ggml-alloc.c)
