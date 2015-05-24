#ifndef _FFT_HAO_CUFFTW
#define _FFT_HAO_CUFFTW

#include "fftw_define.h"
#include "fft_base.h"

class FFTServer_cu: public FFTServer_base
{
    cufftHandle plan;
    cufftDoubleComplex *dataforw = nullptr;
    cufftDoubleComplex *databack = nullptr;
    std::complex<double> *outforw_host = nullptr;
    std::complex<double> *outback_host = nullptr;

 public:
    FFTServer_cu();
    FFTServer_cu(int Dc, const int* Nc, char format); //'C' Column-major: fortran style; 'R' Row-major: c style;
    FFTServer_cu(const FFTServer_cu& x);
    ~FFTServer_cu();
   
    FFTServer_cu& operator  = (const FFTServer_cu& x);
   
    std::complex<double>* fourier_forw(const std::complex<double>* inarray);
    std::complex<double>* fourier_back(const std::complex<double>* inarray);
   
    friend void FFTServer_cu_void_construction_test();
    friend void FFTServer_cu_param_construction_test();
    friend void FFTServer_cu_equal_construction_test();
    friend void FFTServer_cu_equal_test();
};

#endif
