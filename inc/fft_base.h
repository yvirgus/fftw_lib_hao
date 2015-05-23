#ifndef _FFT_BASE
#define _FFT_BASE

class FFTServer_base
{
    int dimen;
    int *n;
    int L;

    virtual ~FFTServer_base() = 0;

    virtual std::complex<double>* fourier_forw(const std::complex<double>* inarray) = 0;
    virtual std::complex<double>* fourier_back(const std::complex<double>* inarray) = 0;

    int const get_dimen const()
    {
        return dimen;
    }

    const int* const get_n()
    {
        return n;
    }

    int const get_L const()
    {
        return L;
    }
    
};

#endif
