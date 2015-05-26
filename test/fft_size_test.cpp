#include <iostream>
#include <cstdlib>
#include "fftw_hao.h"
#include "cufft_hao.h"
#include "time_hao.h"
#include <sys/time.h>

double magma_wtime( void )
{
    struct timeval t;
    gettimeofday( &t, NULL );
    return t.tv_sec + t.tv_usec*1e-6;
}

using namespace std;

void fill_random( complex<double>* A, int L)
{
    double R, I;

    srand(5323);

    for (int i=0; i < L; i++){

            R = (double)rand() / RAND_MAX;
            I = (double)rand() / RAND_MAX;

            A[i] = {R,I};
    }
}

void cufft_fftw_size_1D_test()
{
    //int dimen = 2, Nx = 1000, Ny = 1000, L = Nx*Ny;
    int dimen = 1, L = 1000000;
    int n[1] = {L};
    //int n[dimen] = {Nx,Ny};
    double cpu_time, gpu_time;

    cout << "M: " << L << endl;
    complex<double> *A = new complex<double>[L];
    complex<double> *B_f;

    Timer_hao timer; timer.init();
    fill_random(A, L);
    timer.end();    
    cout << "time: " << timer.seconds << endl;

    //cout << "A: " << A[0] << endl;
    
    cout << "fftw starts!" << endl;
    cpu_time = magma_wtime();
    FFTServer fft(dimen, n, 'R');
    B_f = fft.fourier_forw(A);
    cpu_time = magma_wtime() - cpu_time;
    cout << "B_f: " << B_f[2] << endl;
    cout << "cpu_time: " << cpu_time << endl;
   
    //    for (int i=0; i<M; i++){
    //    cout << "B_f: " << B_f[i] << endl;
    //}

    cout << "cufft starts!" << endl;
    gpu_time = magma_wtime();
    FFTServer_cu cufft(dimen, n, 'R');
    B_f = cufft.fourier_forw(A);
    gpu_time = magma_wtime() - gpu_time;
    cout << "gpu_time: " << gpu_time << endl;
    //for (int i=0; i<M; i++){
    //    cout << "B_f: " << B_f[i] << endl;
    // }
    cout << "cufft ends!" << endl;    

    //timer.print_init();
    //timer.print_end();
    //timer.print_accumulation();

}

void fft_size_test()
{
    cufft_fftw_size_1D_test();
}
