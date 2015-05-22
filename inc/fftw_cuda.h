#ifndef CU_FFTW
#define CU_FFTW

#include "fftw_define.h"

class FFTServer_cu
{
    int  dimen;
    int* n;
    int  L;
    cufftHandle plan;
    cufftDoubleComplex *inforw;
    cufftDoubleComplex *outforw;
    std::complex<double> *outforw_host;
    cufftDoubleComplex *inback;
    cufftDoubleComplex *outback;
    std::complex<double> *outback_host;
 public:
    FFTServer_cu();
    FFTServer_cu(int Dc, const int* Nc, char format); //'C' Column-major: fortran style; 'R' Row-major: c style;
    FFTServer_cu(const FFTServer_cu& x);
    ~FFTServer_cu();
   
    FFTServer_cu& operator  = (const FFTServer_cu& x);
   
    std::complex<double>* fourier_forw(const std::complex<double>* inarray, int *n);
    std::complex<double>* fourier_back(const std::complex<double>* inarray, int *n);
   
    friend void FFTServer_cu_void_construction_test();
    friend void FFTServer_cu_param_construction_test();
    friend void FFTServer_cu_equal_construction_test();
    friend void FFTServer_cu_equal_test();
};

#endif
