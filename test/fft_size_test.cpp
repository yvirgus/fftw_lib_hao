#include <iostream>
#include <cstdlib>
#include "fftw_hao.h"
#include "cufft_hao.h"
#include "time_hao.h"
#include <sys/time.h>
#include <vector>
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

void cufft_fftw_size_1D_test(int L)
{
    int dimen = 1;
    int n[1] = {L};
    double cpu_time_plan, cpu_time, gpu_time_plan, gpu_time;

    complex<double> *A = new complex<double>[L];
    complex<double> *Bf_cpu,  *Bf_gpu,  *Bb_cpu,  *Bb_gpu;

    size_t flag=0;

    //Timer_hao timer; timer.init();
    fill_random(A, L);
    //timer.end();    
    //cout << "time: " << timer.seconds << endl;
    //timer.print_init();
    //timer.print_end();
    //timer.print_accumulation();

    cpu_time_plan = magma_wtime();
    FFTServer fft(dimen, n, 'R');
    cpu_time_plan = magma_wtime() - cpu_time_plan;

    cpu_time = magma_wtime();
    //for (int i = 0; i< 1000; i++) {
    Bf_cpu = fft.fourier_forw(A);
    Bb_cpu = fft.fourier_back(A);
    //}
    cpu_time = magma_wtime() - cpu_time;

    //cout << "cpu_time: " << cpu_time << endl;
    //    for (int i=0; i<M; i++){
    //    cout << "B_f: " << B_f[i] << endl;
    //}

    //cout << "cufft starts!" << endl;

    gpu_time_plan = magma_wtime();
    FFTServer_cu cufft(dimen, n, 'R');
    gpu_time_plan = magma_wtime() - gpu_time_plan;

    gpu_time = magma_wtime();
    //for (int i = 0; i< 1000; i++) {
    Bf_gpu = cufft.fourier_forw(A);
    Bb_gpu = cufft.fourier_back(A);
    //}
    gpu_time = magma_wtime() - gpu_time;
    //cout << "gpu_time: " << gpu_time << endl;
    //for (int i=0; i<M; i++){
    //    cout << "B_f: " << B_f[i] << endl;
    // }
    //cout << "cufft ends!" << endl;    

    for (int i=0; i<L; i++){
        if (abs(Bf_cpu[i] - Bf_gpu[i]) > 1e-10)    flag++;
        if (abs(Bb_cpu[i] - Bb_gpu[i]) > 1e-10)    flag++;
    }

    delete[] A;
    //cout << "flag: " << flag << endl;

    printf("%8d           %7.6f                  %7.6f             %7.6f           %7.6f          %7.6f          %s\n", 
           L, cpu_time_plan, gpu_time_plan, cpu_time*1000, gpu_time*1000, cpu_time/gpu_time, (flag == 0 ? "ok" : "failed"));  

}

void cufft_fftw_size_1D_test()
{
    vector<int> sizes;
    int size[22] = {6,9,12,15,18,24,36,80,108,210,504,1000,1960,4725,10368,27000,75600,165375,362880,1562500,3211264,6250000};

    for (int i = 4; i < 10000000; i *= 2){
        sizes.push_back(i);
    }

    cout << "\n1-dimensional, fftw (CPU) vs cufft (GPU), testing :\n";
    cout << "       L     CPU plan creation (s)     GPU plan creation (s)     CPU_exec (ms)    GPU_exec (ms)   CPU_exec/GPU_exec    result\n";
    cout << "==============================================================================================================================\n";

    for (vector<int>:: iterator it = sizes.begin() ; it!= sizes.end(); ++it){
        cufft_fftw_size_1D_test(*it);
    }
    /*
    cout << "\n1-dimensional, fftw (CPU) vs cufft (GPU), testing :\n";
    cout << "       L     CPU plan creation (s)     GPU plan creation (s)     CPU_exec (ms)    GPU_exec (ms)   CPU_exec/GPU_exec    result\n";
    cout << "==============================================================================================================================\n";

    for (int i = 0 ; i < 22; ++i){
        cufft_fftw_size_1D_test(size[i]);
    }
    */

}

void cufft_fftw_size_2D_test(int Nx, int Ny)
{
    int dimen = 2, L = Nx*Ny;
    int n[2] = {Nx,Ny};
    double cpu_time_plan, cpu_time, gpu_time_plan, gpu_time;

    complex<double> *A = new complex<double>[L];
    complex<double> *Bf_cpu,  *Bf_gpu,  *Bb_cpu,  *Bb_gpu;

    size_t flag=0;

    //Timer_hao timer; timer.init();
    fill_random(A, L);
    //timer.end();    
    //cout << "time: " << timer.seconds << endl;
    //timer.print_init();
    //timer.print_end();
    //timer.print_accumulation();

    cpu_time_plan = magma_wtime();
    FFTServer fft(dimen, n, 'R');
    cpu_time_plan = magma_wtime() - cpu_time_plan;

    cpu_time = magma_wtime();
    //for (int i = 0; i< 1000; i++) {
    Bf_cpu = fft.fourier_forw(A);
    Bb_cpu = fft.fourier_back(A);
    //}
    cpu_time = magma_wtime() - cpu_time;
    //cout << "cpu_time: " << cpu_time << endl;
    //    for (int i=0; i<M; i++){
    //    cout << "B_f: " << B_f[i] << endl;
    //}

    //cout << "cufft starts!" << endl;

    gpu_time_plan = magma_wtime();
    FFTServer_cu cufft(dimen, n, 'R');
    gpu_time_plan = magma_wtime() - gpu_time_plan;

    gpu_time = magma_wtime();
    //for (int i = 0; i< 1000; i++) {
    Bf_gpu = cufft.fourier_forw(A);
    Bb_gpu = cufft.fourier_back(A);
    //}
    gpu_time = magma_wtime() - gpu_time;
    //cout << "gpu_time: " << gpu_time << endl;
    //for (int i=0; i<M; i++){
    //    cout << "B_f: " << B_f[i] << endl;
    // }
    //cout << "cufft ends!" << endl;    

    for (int i=0; i<L; i++){
        if (abs(Bf_cpu[i] - Bf_gpu[i]) > 1e-10)    flag++;
        if (abs(Bb_cpu[i] - Bb_gpu[i]) > 1e-10)    flag++;
    }

    delete[] A;
    //cout << "flag: " << flag << endl;

    printf("%5d     %5d        %7.6f                  %7.6f             %7.6f           %7.6f          %7.6f          %s\n", 
           Nx, Ny, cpu_time_plan, gpu_time_plan, cpu_time*1000, gpu_time*1000, cpu_time/gpu_time, (flag == 0 ? "ok" : "failed"));  

}

void cufft_fftw_size_2D_test()
{
    //int L;
    //vector<int> sizes;
    //int size[19][2] = {{4,4},{8,4},{4,8},{8,8},{16,16},{32,32},{64,64},{16,512},{128,64},{128,128},{256,128},{512,64},{64,1024},
    //                   {256,256},{512,512},{1024,1024},{2048,2048},{4096,4096}, {8192,8192} }; // -- the last one fails when creating plan (in particle)
    int size1[36][2] = { {5,5},{6,6},{7,7},{9,9},{10,10},{11,11},{12,12},{13,13},{14,14},{15,15},{25,24},{48,48},{49,49},{60,60},
                       {72,56},{75,75},{80,80},{84,84},{96,96},{100,100},{105,105},{112,112},{120,120},{144,144},{180,180},{240,240},
                         {360,360},{1000,1000},{1050,1050},{1458,1458},{1960,1960},{2916,2916},{4116,4116}}; //,{5832,5832},{8400,8400},{10368,10368} };

    cout << "\n2-dimensional, fftw (CPU) vs cufft (GPU), testing :\n";
    cout << "     Nx     Ny     CPU plan creation (s)     GPU plan creation (s)     CPU_exec (ms)    GPU_exec (ms)   CPU_exec/GPU_exec    result\n";
    cout << "==============================================================================================================================\n";
    
    //for (int i = 0 ; i < 19; ++i){
    //    cufft_fftw_size_2D_test(size[i][0], size[i][1]);
    //}
    
    for (int i = 0 ; i < 36; ++i){
    cufft_fftw_size_2D_test(size1[i][0], size1[i][1]);
    }

}

void cufft_fftw_size_3D_test(int Nx, int Ny, int Nz)
{
    int dimen = 3, L = Nx*Ny*Nz;
    int n[3] = {Nx,Ny,Nz};
    double cpu_time_plan, cpu_time, gpu_time_plan, gpu_time;

    complex<double> *A = new complex<double>[L];
    complex<double> *Bf_cpu,  *Bf_gpu,  *Bb_cpu,  *Bb_gpu;

    size_t flag=0;

    //Timer_hao timer; timer.init();
    fill_random(A, L);
    //timer.end();    
    //cout << "time: " << timer.seconds << endl;
    //timer.print_init();
    //timer.print_end();
    //timer.print_accumulation();

    cpu_time_plan = magma_wtime();
    FFTServer fft(dimen, n, 'R');
    cpu_time_plan = magma_wtime() - cpu_time_plan;

    cpu_time = magma_wtime();
    //for (int i = 0; i< 1000; i++) {
    Bf_cpu = fft.fourier_forw(A);
    Bb_cpu = fft.fourier_back(A);
    //}
    cpu_time = magma_wtime() - cpu_time;
    //cout << "cpu_time: " << cpu_time << endl;
    //    for (int i=0; i<M; i++){
    //    cout << "B_f: " << B_f[i] << endl;
    //}

    //cout << "cufft starts!" << endl;

    gpu_time_plan = magma_wtime();
    FFTServer_cu cufft(dimen, n, 'R');
    gpu_time_plan = magma_wtime() - gpu_time_plan;

    gpu_time = magma_wtime();
    //for (int i = 0; i< 1000; i++) {
    Bf_gpu = cufft.fourier_forw(A);
    Bb_gpu = cufft.fourier_back(A);
    //}
    gpu_time = magma_wtime() - gpu_time;
    //cout << "gpu_time: " << gpu_time << endl;
    //for (int i=0; i<M; i++){
    //    cout << "B_f: " << B_f[i] << endl;
    // }
    //cout << "cufft ends!" << endl;    

    for (int i=0; i<L; i++){
        if (abs(Bf_cpu[i] - Bf_gpu[i]) > 1e-10)    flag++;
        if (abs(Bb_cpu[i] - Bb_gpu[i]) > 1e-10)    flag++;
    }

    delete[] A;
    //cout << "flag: " << flag << endl;

    printf("%5d     %5d     %5d        %7.6f                  %7.6f             %7.6f           %7.6f          %7.6f          %s\n", 
           Nx, Ny, Nz,  cpu_time_plan, gpu_time_plan, cpu_time*1000, gpu_time*1000, cpu_time/gpu_time, (flag == 0 ? "ok" : "failed"));  

}

void cufft_fftw_size_3D_test()
{
    //int size[14][3] = { {4,4,4},{8,8,8},{4,8,16},{16,16,16},{32,32,32},{64,64,64},{256,64,32},{16,1024,64},{128,128,128},{512,128,64},
    //                    {256,128,256},{256,256,256},{512,64,1024},{512,512,512} }; 
                      
    int size1[36][3] = { {5,5,5},{6,6,6},{7,7,7},{9,9,9},{10,10,10},{11,11,11},{12,12,12},{13,13,13},{14,14,14},{15,15,15},{24,25,28},
                         {48,48,48},{49,49,49},{60,60,60},{72,60,56},{75,75,75},{80,80,80},{84,84,84},{96,96,96},{100,100,100},
                         {105,105,105},{112,112,112},{120,120,120},{144,144,144},{180,180,180},{210,210,210},{270,270,270},{324,324,324},{420,420,420}};
    

    cout << "\n3-dimensional, fftw (CPU) vs cufft (GPU), testing :\n";
    cout << "     Nx     Ny     Nz     CPU plan creation (s)     GPU plan creation (s)     CPU_exec (ms)    GPU_exec (ms)   CPU_exec/GPU_exec    result\n";
    cout << "===========================================================================================================================================\n";
    
    //for (int i = 0 ; i < 14; ++i){
    //    cufft_fftw_size_3D_test( size[i][0], size[i][1], size[i][2]);
    //}
    
    for (int i = 0 ; i < 36; ++i){
        cufft_fftw_size_3D_test( size1[i][0], size1[i][1], size1[i][2]);
    }

}

void cufft_fftw_size_1D_test_batch(int L)
{
    int dimen = 1, batch = L;
    int n[1] = {L};
    double gpu_time_batch_plan, gpu_time_batch, cpu_time_plan, cpu_time, copy_time = 0, copy_time_temp;

    complex<double> *A = new complex<double>[L*batch];
    complex<double> *Bf_gpu = new complex<double>[L*batch];
    complex<double> *Bb_gpu = new complex<double>[L*batch];
    complex<double> *Bf_gpu_batch = new complex<double>[L*batch];
    complex<double> *Bb_gpu_batch = new complex<double>[L*batch];
    complex<double> *Bf_t, *Bb_t;

    size_t flag = 0;

    fill_random(A, L*batch);

    //cout << "A :" << A[L*batch - 1] << endl;

    cpu_time_plan = magma_wtime();
    FFTServer fft(dimen, n, 'R');
    cpu_time_plan = magma_wtime() - cpu_time_plan;

    cpu_time = magma_wtime();
    for (int i = 0; i< batch; i++) {
        Bf_t = fft.fourier_forw(A + i*L);
        Bb_t = fft.fourier_back(A + i*L);

        copy_time_temp = magma_wtime();
        std::copy(Bf_t, Bf_t + L, Bf_gpu + i*L);
        std::copy(Bb_t, Bb_t + L, Bb_gpu + i*L);
        copy_time = copy_time + (magma_wtime() - copy_time_temp);
    }
    cpu_time = magma_wtime() - cpu_time - copy_time;

    //cout << "copy_time: " << copy_time*1000 << " (ms)" << endl;

    gpu_time_batch_plan = magma_wtime();
    FFTServer_cu cufft_batch(dimen, n, 'R', batch);
    gpu_time_batch_plan = magma_wtime() - gpu_time_batch_plan;

    gpu_time_batch = magma_wtime();
    Bf_gpu_batch = cufft_batch.fourier_forw(A);
    Bb_gpu_batch = cufft_batch.fourier_back(A);
    gpu_time_batch = magma_wtime() - gpu_time_batch;


    for (int i=0; i<L*batch ; i++){
        if (abs(Bf_gpu_batch[i] - Bf_gpu[i]) > 1e-10)    flag++;
        if (abs(Bb_gpu_batch[i] - Bb_gpu[i]) > 1e-10)    flag++;
    }

    //cout << "flag: " << flag << endl;

    printf("%8d       %8d       %7.6f                  %7.6f             %7.6f           %7.6f          %7.6f          %s\n", 
           L, batch,  cpu_time_plan, gpu_time_batch_plan, cpu_time*1000, gpu_time_batch*1000, cpu_time/gpu_time_batch, (flag == 0 ? "ok" : "failed"));  

}

void cufft_fftw_size_1D_test_batch()
{
    vector<int> sizes;
    int size[15] = {6,9,12,15,18,24,36,80,108,210,504,1000,1960,4725,5325};

    for (int i = 2; i < 5000; i *= 2){
        sizes.push_back(i);
    }

    cout << "\n1-dimensional, fftw (CPU) vs cufft-batch (GPU), testing :\n";
    cout << "       L      batch       CPU plan creation (s)     GPU plan creation (s)     CPU_exec (ms)    GPU_exec (ms)   CPU_exec/GPU_exec    result\n";
    cout << "===========================================================================================================================================\n";

    for (vector<int>:: iterator it = sizes.begin() ; it!= sizes.end(); ++it){
        cufft_fftw_size_1D_test_batch(*it);
    }

    /*
    cout << "\n1-dimensional, fftw (CPU) vs cufft-batch (GPU), testing :\n";
    cout << "       L      batch       CPU plan creation (s)     GPU plan creation (s)     CPU_exec (ms)    GPU_exec (ms)   CPU_exec/GPU_exec    result\n";
    cout << "===========================================================================================================================================\n";

    for (int i = 0 ; i < 15; ++i){
        cufft_fftw_size_1D_test_batch(size[i]);
    }
    */
}



void cufft_fftw_size_2D_test_batch(int Nx, int Ny)
{
    int dimen = 2, batch = Nx*Ny, L = Nx*Ny;
    int n[2] = {Nx, Ny};
    double gpu_time_batch_plan, gpu_time_batch, cpu_time_plan, cpu_time, copy_time = 0, copy_time_temp;

    complex<double> *A = new complex<double>[L*batch];
    complex<double> *Bf_gpu = new complex<double>[L*batch];
    complex<double> *Bb_gpu = new complex<double>[L*batch];
    complex<double> *Bf_gpu_batch = new complex<double>[L*batch];
    complex<double> *Bb_gpu_batch = new complex<double>[L*batch];
    complex<double> *Bf_t, *Bb_t;

    size_t flag = 0;

    fill_random(A, L*batch);

    cpu_time_plan = magma_wtime();
    FFTServer fft(dimen, n, 'R');
    cpu_time_plan = magma_wtime() - cpu_time_plan;

    cpu_time = magma_wtime();
    for (int i = 0; i< batch; i++) {
        Bf_t = fft.fourier_forw(A + i*L);
        Bb_t = fft.fourier_back(A + i*L);

        copy_time_temp = magma_wtime();
        std::copy(Bf_t, Bf_t + L, Bf_gpu + i*L);
        std::copy(Bb_t, Bb_t + L, Bb_gpu + i*L);
        copy_time = copy_time + (magma_wtime() - copy_time_temp);
    }
    cpu_time = magma_wtime() - cpu_time - copy_time;

    //cout << "copy_time: " << copy_time*1000 << " (ms)" << endl;

    gpu_time_batch_plan = magma_wtime();
    FFTServer_cu cufft_batch(dimen, n, 'R', batch);
    gpu_time_batch_plan = magma_wtime() - gpu_time_batch_plan;

    gpu_time_batch = magma_wtime();
    Bf_gpu_batch = cufft_batch.fourier_forw(A);
    Bb_gpu_batch = cufft_batch.fourier_back(A);
    gpu_time_batch = magma_wtime() - gpu_time_batch;


    for (int i=0; i<L*batch ; i++){
        if (abs(Bf_gpu_batch[i] - Bf_gpu[i]) > 1e-10)    flag++;
        if (abs(Bb_gpu_batch[i] - Bb_gpu[i]) > 1e-10)    flag++;
    }

    //cout << "flag: " << flag << endl;

    printf("%5d       %5d       %5d       %7.6f                  %7.6f             %7.6f           %7.6f          %7.6f          %s\n", 
           Nx, Ny,  batch,  cpu_time_plan, gpu_time_batch_plan, cpu_time*1000, gpu_time_batch*1000, cpu_time/gpu_time_batch, (flag == 0 ? "ok" : "failed"));  

}

void cufft_fftw_size_2D_test_batch()
{
    //int size[15][2] = {{4,4},{8,4},{4,8},{8,8},{16,16},{32,32},{64,64},{16,512},{128,64},{128,128},{256,128},{512,64},{64,1024},
    //                   {256,256},{512,512} };
    //int size1[36][2] = { {5,5},{6,6},{7,7},{9,9},{10,10},{11,11},{12,12},{13,13},{14,14},{15,15},{25,24},{48,48},{49,49},{60,60},
    //                   {72,56},{75,75},{80,80},{84,84},{96,96},{100,100},{105,105},{112,112},{120,120},{144,144},{180,180},{240,240},
    //                     {360,360},{1000,1000},{1050,1050},{1458,1458},{1960,1960},{2916,2916},{4116,4116}}; //,{5832,5832},{8400,8400},{10368,10368} };

    cout << "\n2-dimensional, fftw (CPU) vs cufft (GPU), testing :\n";
    cout << "     Nx     Ny   batch    CPU plan creation (s)     GPU plan creation (s)     CPU_exec (ms)    GPU_exec (ms)   CPU_exec/GPU_exec    result\n";
    cout << "==========================================================================================================================================\n";
    
    vector<int> sizes;

    for (int i = 2; i < 100; i += 2){
        sizes.push_back(i);
    }

    for (vector<int>:: iterator it = sizes.begin() ; it!= sizes.end(); ++it){
        cufft_fftw_size_2D_test_batch(*it, *it);
    }

    //for (int i = 0 ; i < 15; ++i){
    //    cufft_fftw_size_2D_test_batch(size[i][0], size[i][1]);
    //}
/*
    for (int i = 0 ; i < 36; ++i){
    cufft_fftw_size_2D_test_batch(size1[i][0], size1[i][1]);
    }
*/
}

void fft_size_test()
{
    //cufft_fftw_size_1D_test_batch();
    cufft_fftw_size_2D_test_batch();
    //cufft_fftw_size_1D_test();
    //cufft_fftw_size_2D_test();
    //cufft_fftw_size_3D_test();
 
}
