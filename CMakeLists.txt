cmake_minimum_required(VERSION 3.8...3.12 FATAL_ERROR)
project(gpuvsim LANGUAGES CXX CUDA)

add_executable(gpuvsim ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu)

target_sources(gpuvsim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/directioncosines.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/functions.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MSFITSIO.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rngs.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rvgs.cu
)


if(PREFIX)
  set(BINARY_DIR ${PREFIX})
  set_target_properties(gpuvsim
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR}
  )
else()
  set(BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
  set_target_properties(gpuvsim
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR}
  )
endif()

message(WARNING "We are currently using the latest casacore stable version v3.1.2")
message(WARNING "If this version is not installed in your computer, please make sure to install it")
message(WARNING "For more information please read the README.md file")
set(CASACORE_DIR /usr/local/include/casacore)
target_include_directories(gpuvsim PUBLIC
${CASACORE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/include
)

get_target_property(TEMP gpuvsim COMPILE_FLAGS)
if(TEMP STREQUAL "TEMP-NOTFOUND")
  set(TEMP "") # Set to empty string
else()
  set(TEMP "${TEMP} ") # A space to cleanly separate from existing content
endif()

set(COMPFLAGS "-D_FORCE_INLINES -w -O3")

set(TEMP "${TEMP}${COMPFLAGS}" )
set_target_properties(gpuvsim PROPERTIES COMPILE_FLAGS ${TEMP} CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 11)
# Search the compute capability of the first GPU and add CUDA specific FLAGS
include(${CMAKE_CURRENT_SOURCE_DIR}/cuda_check.cmake)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3 -Xcompiler -fopenmp ${CUDA_NVCC_ARCH_FLAGS}")

target_link_libraries(gpuvsim cfitsio m casa_casa casa_tables casa_ms casa_measures cudart cuda cufft stdc++ gomp)
