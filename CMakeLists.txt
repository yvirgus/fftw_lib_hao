cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(fftwhao)

#For find module path
set(CMAKE_MODULE_PATH "${MODULE_EXTRA_PATH}" ${CMAKE_MODULE_PATH})

#For definitions
add_definitions(${COMPILER_EXTRA_DEF})

#For c++ flag
set(CMAKE_CXX_FLAGS "${COMPILER_EXTRA_FLAG}" ${CMAKE_CXX_FLAGS})

find_package(fftw REQUIRED)
include_directories(${fftw_include_dirs})
set(all_lib "${fftw_libraries}")

find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
set(all_lib "${CUDA_CUFFT_LIBRARIES}" "${CUDA_LIBRARIES}" "${all_lib}")

#if(USE_CUDA)
#  find_package(CUDA)
#  include_directories(${CUDA_INCLUDE_DIRS})
#  message("CUDA DIRS: ${CUDA_INCLUDE_DIRS}")
#  set(all_lib "${CUDA_CUFFT_LIBRARIES}" "${CUDA_LIBRARIES}")
#  #set(all_lib -L/usr/local/cuda-5.5/lib64 -lcudart "${CUDA_CUFFT_LIBRARIES}")      
#  message("all libs: ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES}")
#else()
#  find_package(fftw REQUIRED)
#  include_directories(${fftw_include_dirs})
#  set(all_lib "${fftw_libraries}")
#endif()

include_directories(inc)

add_subdirectory(inc)

add_subdirectory(src)

add_subdirectory(test)
