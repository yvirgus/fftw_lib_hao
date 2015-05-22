#include<iostream>
#include<stdlib.h>
#include <algorithm> 
#include "fftw_cuda.h"
//#include "cuda_runtime_api.h"
using std::cout;
using std::endl;
using std::complex;

/*
typedef cufftDoubleComplex zcomplex_t;

typedef zcomplex_t *zcomplex_ptr_t;
typedef const zcomplex_t *const_zcomplex_ptr_t;

static zcomplex_ptr_t _cast_Zptr(std::complex<double> *A)
{   
    return reinterpret_cast<zcomplex_ptr_t>(A);
}
static const_zcomplex_ptr_t _cast_Zptr(const std::complex<double> *A)
{
    return reinterpret_cast<const_zcomplex_ptr_t>(A);
}
*/

FFTServer_cu::FFTServer_cu()
{
    cufftResult stat;
    dimen=1;
    n=new int[1]; n[0]=1;
    L=1;

    stat = cufftPlanMany(&plan, dimen, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 1);

    if (stat != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Plan creation failed\n");
    }

    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
    }
}

FFTServer_cu::FFTServer_cu(int Dc, const int* Nc, char format)
{
    cufftResult stat;
    dimen=Dc;
    n=new int[dimen];
    if(format=='C') std::reverse_copy(Nc,Nc+dimen,n);  //Column-major: fortran style
    else if(format=='R') std::copy(Nc,Nc+dimen,n); //Row-major: c style
    else {cout<<"Do not know the format!!!! "<<format<<endl; exit(1); }

    L=1; for(int i=0; i<dimen; i++) L*=n[i];

    stat = cufftPlanMany(&plan, dimen, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 1);

    if (stat != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Plan creation failed\n");
    }
    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
    }
}

FFTServer_cu::FFTServer_cu(const FFTServer_cu& x) 
{
    cufftResult stat;
    dimen=x.dimen;
    n=new int[dimen]; std::copy(x.n,x.n+dimen,n);
    L=x.L;

    stat = cufftPlanMany(&plan, dimen, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 1);

    if (stat != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Plan creation failed\n");
    }
    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
    }
}

FFTServer_cu::~FFTServer_cu()
{
    cout << "Destructor" << endl;
    
    if(n)       delete[] n;
    if(inforw) { cout << "inforw detected " << endl; cudaFree(inforw); } 
    if(outforw) { cout << "outforw detected " << endl; cudaFree(outforw); } 
    if(outforw_host) { cout << "outforw_host detected " << endl; delete[] outforw_host; } 
    if(inback) { cout << "inback detected " << endl; cudaFree(inback); } 
    if(outback) { cout << "outback detected " << endl; cudaFree(outback); } 
    if(outback_host) { cout << "outback_host detected " << endl; delete[] outback_host; } 
    cufftDestroy(plan);
}

FFTServer_cu& FFTServer_cu::operator  = (const FFTServer_cu& x)
{
    cufftResult stat;
    dimen=x.dimen;
    if(n) delete[] n; n=new int[dimen]; std::copy(x.n,x.n+dimen,n);
    L=x.L;
   
    cufftDestroy(plan);
    stat = cufftPlanMany(&plan, dimen, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 1);
   
    if (stat != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Plan creation failed\n");
    }

    return *this;
}

complex<double>* FFTServer_cu::fourier_forw(const complex<double>* inarray)
{
    cufftResult stat;

    cudaMalloc((void**)&inforw, sizeof(cufftDoubleComplex)*L);
    cudaMalloc((void**)&outforw, sizeof(cufftDoubleComplex)*L);

    cudaMemcpy(inforw, inarray, sizeof(cufftDoubleComplex)*L, cudaMemcpyHostToDevice);

    stat = cufftExecZ2Z(plan, inforw, outforw, CUFFT_FORWARD);

    if (stat != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Failed to execute plan\n");
    }
    
    outforw_host = new complex<double>[sizeof(cufftDoubleComplex)*L];

    cudaMemcpy(outforw_host, outforw, sizeof(cufftDoubleComplex)*L, cudaMemcpyDeviceToHost);

    //for(int i=0; i<L; i++) {cout << "b forward: " << outforw_host[i] << endl;}

    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
    }

    return outforw_host;
}

complex<double>* FFTServer_cu::fourier_back(const complex<double>* inarray)
{
    cufftResult stat;

    cudaMalloc((void**)&inback, sizeof(cufftDoubleComplex)*L);
    cudaMalloc((void**)&outback, sizeof(cufftDoubleComplex)*L);

    cudaMemcpy(inback, inarray, sizeof(cufftDoubleComplex)*L, cudaMemcpyHostToDevice);

    stat = cufftExecZ2Z(plan, inback, outback, CUFFT_INVERSE);

    if (stat != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Failed to execute plan\n");
    }
    
    outback_host = new complex<double>[sizeof(cufftDoubleComplex)*L];

    cudaMemcpy(outback_host, outback, sizeof(cufftDoubleComplex)*L, cudaMemcpyDeviceToHost);

    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
    }

    return outback_host;
}

