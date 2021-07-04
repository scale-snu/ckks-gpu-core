cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86)
endif()

project(DeviceFunctions LANGUAGES CUDA CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

find_package(CUDAToolkit REQUIRED)
find_package(Thrust REQUIRED)

add_library(DeviceFunctions SHARED
  src/public/DeviceVector.cu
  src/public/Context.cu
  src/public/CudaHelper.cu
  src/public/MultPtxtBatch.cu
  src/public/NttImple.cu
)

target_link_directories(DeviceFunctions PUBLIC /usr/local/cuda/lib64)
target_link_libraries(DeviceFunctions
  PUBLIC Thrust CUDA::cudart rmm
  PRIVATE CUDA::nvToolsExt)
target_include_directories(DeviceFunctions
  INTERFACE include
  PRIVATE include/public
)
target_compile_features(DeviceFunctions PUBLIC cxx_std_17 cuda_std_17)
target_compile_options(DeviceFunctions PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp,-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RELWITHDEBINFO>>:-lineinfo>
)

add_subdirectory(unittest)
add_subdirectory(thirdparty)
add_subdirectory(benchmark)
