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

const size_t proc_elems = 52323;
const uint32_t SEED = 42;
const int TRIALS = 50;

typedef enum {
  SUCCESS,
  FAILURE
} status;

int main(int argc, char **argv)
{
  int err;
  // Error handling scheme: this function has failed until proven otherwise.
  int ret = EXIT_FAILURE;

  err = MPI_Init(&argc, &argv);
  if(err != MPI_SUCCESS) {
    // Theoretically, an error at this point will abort the program, and this
    // code path is never followed. This is here for completeness.
    fprintf(stderr, "unable to initialize MPI\n");
    goto die_immed;
  }

  // Install the MPI error handler that returns error codes, so we can perform
  // the usual process suicide ritual.
  err = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  if(err != MPI_SUCCESS) {
    // Again, theoretically, the previous error handler (MPI_Abort) gets called
    // instead of reaching this fail point.
    fprintf(stderr, "unable to reset MPI error handler\n");
    goto die_finalize_mpi;
  }

  int size, rank;
  err = MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(err == MPI_SUCCESS) err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to determine rank or size\n");
    goto die_finalize_mpi;
  }

  dsfmt_t *prng = malloc(sizeof(dsfmt_t));
  if(prng == NULL) {
    fprintf(stderr, "unable to allocate PRNG\n");
    goto die_finalize_mpi;
  }
  dsfmt_init_gen_rand(prng, SEED + rank);

  // Allocate master source array for FFT.
  double *const master = fftw_malloc(proc_elems*sizeof(double));
  if(master == NULL) {
    fprintf(stderr, "unable to allocate master array\n");
    goto die_free_prng;
  }
  for(unsigned int i = 0; i < proc_elems; ++i) {
    master[i] = dsfmt_genrand_open_close(prng) * 10;
  }

  /* Allocate source array for serial array. We copy the master array to this
   * array, then transform it in place, then reverse transform it. The idea is
   * that we should get the original data back, and we use this as a consistency
   * check. We need the original data to compare to.
   */
  double *const source = fftw_malloc(proc_elems*sizeof(double));
  if(source == NULL) {
    fprintf(stderr, "unable to allocate source array\n");
    goto die_free_master;
  }
  for(int i = 0; i < proc_elems; ++i) source[i] = master[i];

  /* Allocate the destination array */
  double complex *const dest = fftw_malloc(proc_elems*sizeof(double complex));
  if(dest == NULL) {
    fprintf(stderr, "unable to allocate destination array\n");
    goto die_free_source;
  }

  /* Allocate a plan to compute the FFT */
  fft_par_plan plan = fft_par_plan_r2c_1d(MPI_COMM_WORLD, proc_elems, source,
      dest, &err);
  if(plan == NULL) {
    fprintf(stderr, "unable to initialize parallel FFT plan\n");
    goto die_free_dest;
  }

  /* Execute the forward plan */
  err = fft_par_execute_fwd(plan);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "error computing forward plan\n");
    goto die_free_plan;
  }

  /* Execute the reverse plan */
  err = fft_par_execute_rev(plan);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "error computing reverse plan\n");
    goto die_free_plan;
  }

  /* Compare source to master, use supremum norm */
  int norm = 0.0;
  for(int i = 0; i < proc_elems; ++i) {
    /* Each FFT effectively multiplies by sqrt(proc_elems*num_procs) */
    norm = fmax(norm, fabs(master[i] - source[i]/proc_elems/size));
  }
  if(norm < 1.0e-6) {
    ret = EXIT_SUCCESS;
  }

die_free_plan:
  fft_par_plan_destroy(plan);
die_free_dest:
  fftw_free(dest);
die_free_source:
  fftw_free(source);
die_free_master:
  fftw_free(master);
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
