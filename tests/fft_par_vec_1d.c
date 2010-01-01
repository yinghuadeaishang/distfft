#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <error.h>
#include <math.h>
#include <mpi.h>
#include <fftw3.h>
#include "dSFMT/dSFMT.h"
#include "fft_ser.h"
#include "fft_par.h"

const int proc_elems = 521;
const int VL = 1;
const uint32_t SEED = 42;
const int TRIALS = 1;

int main(int argc, char **argv)
{
  // Error handling scheme: this function has failed until proven otherwise.
  int ret = EXIT_FAILURE;

  if(MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    // Theoretically, an error at this point will abort the program, and this
    // code path is never followed. This is here for completeness.
    fprintf(stderr, "unable to initialize MPI\n");
    goto die_immed;
  }

  // Install the MPI error handler that returns error codes, so we can perform
  // the usual process suicide ritual.
  if(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN)
      != MPI_SUCCESS) {
    // Again, theoretically, the previous error handler (MPI_Abort) gets called
    // instead of reaching this fail point.
    fprintf(stderr, "unable to reset MPI error handler\n");
    goto die_finalize_mpi;
  }

  int size, rank;
  if(MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS ||
      MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) {
    fprintf(stderr, "unable to determine rank and size\n");
    goto die_finalize_mpi;
  }

  dsfmt_t *prng = malloc(sizeof(dsfmt_t));
  if(prng == NULL) {
    fprintf(stderr, "unable to allocate PRNG\n");
    goto die_finalize_mpi;
  }
  dsfmt_init_gen_rand(prng, SEED + rank);

  const int master_elems = VL * proc_elems * size;

  double *master = malloc(master_elems*sizeof(double));
  if(master == NULL) {
    fprintf(stderr, "unable to allocate master array\n");
    goto die_free_prng;
  }
  for(int i = 0; i < master_elems; ++i) {
    master[i]

  free(master);
die_free_prng:
  free(prng);
die_finalize_mpi:
  if(MPI_Finalize() != MPI_SUCCESS) {
    fprintf(stderr, "unable to finalize MPI\n");
    ret = EXIT_FAILURE;
  }
die_immed:
  fftw_cleanup();
  return ret;
}
