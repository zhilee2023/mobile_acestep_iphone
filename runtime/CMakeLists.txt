cmake_minimum_required(VERSION 3.24)
project(AceStepRuntime VERSION 0.1.0 LANGUAGES CXX)

add_library(ace_runtime STATIC
    src/AceEngine.cpp
    src/WavWriter.cpp
)
target_include_directories(ace_runtime PUBLIC include)
target_compile_features(ace_runtime PUBLIC cxx_std_20)

add_executable(ace_runtime_smoke src/smoke_main.cpp)
target_link_libraries(ace_runtime_smoke PRIVATE ace_runtime)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME wav_smoke COMMAND ace_runtime_smoke "${CMAKE_CURRENT_BINARY_DIR}/smoke-cpp.wav")
endif()

