#set(header fftw_define.h fftw_hao.h fftw_cuda.h)
set(header fftw_define.h fftw_cuda.h)

install(FILES ${header}  DESTINATION include)
