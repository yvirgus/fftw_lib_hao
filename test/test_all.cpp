#include <iostream>
#ifdef MPI_HAO
#include <mpi.h>
#endif

//#include "fftw_hao.h"
#include "cufft_hao.h"

using namespace std;

void fftw_hao_test();
void cufft_hao_test();

int main(int argc, char** argv)
{
    int rank=0;

#ifdef MPI_HAO
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if(rank==0)
    {   
        cout<<"\n\n\n=======Testing======="<<endl;
        fftw_hao_test();
        cufft_hao_test();
    }


#ifdef MPI_HAO
    MPI_Finalize();
#endif

    return 0;
}
