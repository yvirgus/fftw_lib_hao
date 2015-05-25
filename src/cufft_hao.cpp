#include<iostream>
#include<stdlib.h>
#include <algorithm> 
#include "cufft_hao.h"

using std::cout;
using std::endl;
using std::complex;

#define USE_CUDA_HOST 1

FFTServer_cu::FFTServer_cu()
{
    cufftResult stat;
    dimen=1;
    n=new int[1]; n[0]=1;
    L=1;

    cudaMalloc((void**)&dataforw, sizeof(cufftDoubleComplex)*L);
    cudaMalloc((void**)&databack, sizeof(cufftDoubleComplex)*L);

#if USE_CUDA_HOST 
    cudaMallocHost((void**)&outforw_host, sizeof(cufftDoubleComplex)*L);
    cudaMallocHost((void**)&outback_host, sizeof(cufftDoubleComplex)*L);
#else
    outforw_host = new complex<double>[sizeof(cufftDoubleComplex)*L];
    outback_host = new complex<double>[sizeof(cufftDoubleComplex)*L];
#endif 

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

    cudaMalloc((void**)&dataforw, sizeof(cufftDoubleComplex)*L);
    cudaMalloc((void**)&databack, sizeof(cufftDoubleComplex)*L);

#if USE_CUDA_HOST
    cudaMallocHost((void**)&outforw_host, sizeof(cufftDoubleComplex)*L);
    cudaMallocHost((void**)&outback_host, sizeof(cufftDoubleComplex)*L);
#else
    outforw_host = new complex<double>[sizeof(cufftDoubleComplex)*L];
    outback_host = new complex<double>[sizeof(cufftDoubleComplex)*L];
#endif 

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

#if USE_CUDA_HOST
    cudaMallocHost((void**)&outforw_host, sizeof(cufftDoubleComplex)*L);
    cudaMallocHost((void**)&outback_host, sizeof(cufftDoubleComplex)*L);
#else
    outforw_host = new complex<double>[sizeof(cufftDoubleComplex)*L];
    outback_host = new complex<double>[sizeof(cufftDoubleComplex)*L];
#endif 

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
    if(n)              delete[] n;
    if(dataforw)       cudaFree(dataforw); 
    if(databack)       cudaFree(databack);  

#if USE_CUDA_HOST
    if(outforw_host)   cudaFreeHost(outforw_host); 
    if(outback_host)    cudaFreeHost(outback_host); 
#else
    if(outforw_host)  {cout << "outforw_host detected" << endl; delete[] outforw_host;}
    if(outback_host)   delete[] outback_host; 
#endif

    cufftDestroy(plan);
}

FFTServer_cu& FFTServer_cu::operator  = (const FFTServer_cu& x)
{
    cufftResult stat;
    dimen=x.dimen;
    if(n) delete[] n; n=new int[dimen]; std::copy(x.n,x.n+dimen,n);
    L=x.L;

#if USE_CUDA_HOST
    if(outforw_host)   cudaFreeHost(outforw_host); 
    if(outback_host)    cudaFreeHost(outback_host); 
#else
    if(outforw_host)   delete[] outforw_host;
    if(outback_host)   delete[] outback_host; 
#endif

#if USE_CUDA_HOST
    cudaMallocHost((void**)&outforw_host, sizeof(cufftDoubleComplex)*L);
    cudaMallocHost((void**)&outback_host, sizeof(cufftDoubleComplex)*L);
#else
    outforw_host = new complex<double>[sizeof(cufftDoubleComplex)*L];
    outback_host = new complex<double>[sizeof(cufftDoubleComplex)*L];
#endif 
   
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

    cudaMemcpy(dataforw, inarray, sizeof(cufftDoubleComplex)*L, cudaMemcpyHostToDevice);

    stat = cufftExecZ2Z(plan, dataforw, dataforw, CUFFT_FORWARD);

    if (stat != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Failed to execute plan\n");
    }
    
    cudaMemcpy(outforw_host, dataforw, sizeof(cufftDoubleComplex)*L, cudaMemcpyDeviceToHost);

    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
    }

    return outforw_host;
}

complex<double>* FFTServer_cu::fourier_back(const complex<double>* inarray)
{
    cufftResult stat;

    cudaMemcpy(databack, inarray, sizeof(cufftDoubleComplex)*L, cudaMemcpyHostToDevice);

    stat = cufftExecZ2Z(plan, databack, databack, CUFFT_INVERSE);

    if (stat != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Failed to execute plan\n");
    }
    
    cudaMemcpy(outback_host, databack, sizeof(cufftDoubleComplex)*L, cudaMemcpyDeviceToHost);

    if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
    }

    return outback_host;
}

