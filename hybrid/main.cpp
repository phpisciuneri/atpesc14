#include <iostream>
#include <mpi.h>
#include <omp.h>

int main( int argc, char** argv )
{

  enum { NDIM=1 };

  const int N=10000000;
  double A[N], B[N], C[N];

  int size, rank, provided;

  MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &provided );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  // output basic info
  if ( rank == 0 ) {
    std::cout << std::endl;
    std::cout << "MPI + OpenMP Example" << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << "A[i] = B[i] + c*C[i]" << std::endl;
    std::cout << "N = 10,000,000" << std::endl << std::endl;
  }

  // perform MPI decomposition
  int nlocal = N / size;
  if ( rank < N % size ) nlocal++;

  MPI_Barrier( MPI_COMM_WORLD );
  double start_time = MPI_Wtime();

#pragma omp parallel shared(A,B,C)
  {
    
    // initialize arrays
    int i;
#pragma omp for private(i)
    for (i=rank*nlocal; i<(rank+1)*nlocal; i++) {
      A[i] = 0.;
      B[i] = 3.;
      C[i] = 4.;
    }

    // perform calculation
#pragma omp for private(i)
    for (i=rank*nlocal; i<(rank+1)*nlocal; i++) {
      A[i] = B[i] + 3.14*C[i];
    }
    
  } // omp parallel
  
  MPI_Barrier( MPI_COMM_WORLD );
  double elapsed_time = MPI_Wtime() - start_time;
  
  // output timing info
  if ( rank == 0 )
    std::cout << "Kernel time (s): " << elapsed_time << std::endl << std::endl;
  
  MPI_Finalize();

  return 0;


}
