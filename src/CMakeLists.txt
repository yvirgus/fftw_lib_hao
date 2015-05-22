#set(fftw_srcs fftw_hao.cpp fftw_cuda.cpp)
set(fftw_srcs fftw_cuda.cpp)

add_library(fftwhao STATIC ${fftw_srcs})

install(TARGETS fftwhao  DESTINATION lib)
