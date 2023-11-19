
include_directories(${PROJECT_SOURCE_DIR})
add_executable(model_load_test model-load-test.cpp)
target_link_libraries(model_load_test PRIVATE paraformer-core)

add_executable(feature_extract_test feature-extract-test.cpp)
target_link_libraries(feature_extract_test PRIVATE paraformer-core)