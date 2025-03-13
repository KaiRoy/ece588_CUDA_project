cmake_minimum_required(VERSION 3.17)

project(CUDAMatrixSolver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# set(SOURCES
#     MatrixSolver/main.cpp
# )

add_executable(${CMAKE_PROJECT_NAME})

target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    CUDAMatrixSolver/main.cu
)

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: >
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr>
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 89)