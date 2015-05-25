set(fftw_test_srcs fftw_hao_test.cpp cufft_hao_test.cpp test_all.cpp time_hao.cpp fft_size_test.cpp)

add_executable(fftwtest ${fftw_test_srcs})

target_link_libraries(fftwtest fftwhao)

target_link_libraries(fftwtest ${all_lib})

install(TARGETS fftwtest  DESTINATION test)
