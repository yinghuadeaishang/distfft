#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <error.h>
#include <mpi.h>
#include <fftw3.h>
#include "dSFMT/dSFMT.h"
#include "fft_ser.h"
#include "fft_par.h"

const size_t proc_elems = 6;
const uint32_t SEED = 42;

typedef enum {
  SUCCESS,
  FAILURE
} status;

status do_serial_fft(size_t len, double *src, double complex *dst);
status do_parallel_fft(double *src, double complex *dst);

int main(int argc, char **argv)
{
  // Error handling scheme: this function has failed until proven otherwise.
  int ret = EXIT_FAILURE;

  if(MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    // Theoretically, an error at this point will abort the program, and this
    // code path is never followed. This is here for completeness.
    fprintf(stderr, "unable to initialize MPI\n");
    goto fail_immed;
  }

  // Install the MPI error handler that returns error codes, so we can perform
  // the usual process suicide ritual.
  if(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN)
      != MPI_SUCCESS) {
    // Again, theoretically, the previous error handler (MPI_Abort) gets called
    // instead of reaching this fail point.
    fprintf(stderr, "unable to reset MPI error handler\n");
    goto fail_finalize_mpi;
  }

  int size, rank;
  if(MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS ||
      MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) {
    fprintf(stderr, "unable to determine rank and size\n");
    goto fail_finalize_mpi;
  }

  dsfmt_t *prng = malloc(sizeof(dsfmt_t));
  if(prng == NULL) {
    fprintf(stderr, "unable to allocate PRNG\n");
    goto fail_finalize_mpi;
  }
  dsfmt_init_gen_rand(prng, SEED + rank);

  const size_t total_elems = proc_elems * size;

  // Allocate master source array for FFT.
  double *master = fftw_malloc(total_elems*sizeof(double));
  if(master == NULL) {
    fprintf(stderr, "unable to allocate master array\n");
    goto fail_free_prng;
  }
  for(unsigned int i = 0; i < total_elems; ++i) {
    master[i] = dsfmt_genrand_open_close(prng) * 10;
  }
  // Ensure all processors have the same array
  if(MPI_SUCCESS != MPI_Bcast(master, total_elems, MPI_DOUBLE, 0,
        MPI_COMM_WORLD)) {
    fprintf(stderr, "unable to synchronize master array.\n");
    goto fail_free_master;
  }

  // Allocate destinarion array for serial array.
  double complex *serial = fftw_malloc(total_elems*sizeof(double complex));
  if(serial == NULL) {
    fprintf(stderr, "unable to allocate serial array\n");
    goto fail_free_master;
  }

  // Compute the serial fft for comparison purposes
  if(do_serial_fft(total_elems, master, serial) != SUCCESS) {
    fprintf(stderr, "unable to perform serial fft\n");
    goto fail_free_serial;
  }

  // Allocate space for the parallel fft solution.
  double complex *parallel = fftw_malloc(proc_elems*sizeof(double complex));
  if(parallel == NULL) {
    fprintf(stderr, "unable to allocate parallel array\n");
    goto fail_free_serial;
  }

  if(do_parallel_fft(master + rank*proc_elems, parallel) != SUCCESS) {
    fprintf(stderr, "unable to perform parallel fft\n");
    goto fail_free_parallel;
  }

  // Error handling scheme changes here: now we have succeeded unless a
  // cleanup function chokes and dies.
  ret = EXIT_SUCCESS;
fail_free_parallel:
  fftw_free(parallel);
fail_free_serial:
  fftw_free(serial);
fail_free_master:
  fftw_free(master);
fail_free_prng:
  free(prng);
fail_finalize_mpi:
  if(MPI_Finalize() != MPI_SUCCESS) {
    fprintf(stderr, "unable to finalize MPI\n");
    ret = EXIT_FAILURE;
  }
fail_immed:
  fftw_cleanup();
  return ret;
}

status do_serial_fft(size_t const n, double * const src,
    double complex * const dst)
{
  status ret = FAILURE; // Failure until proven otherwise
  fftw_plan plan = fftw_plan_dft_r2c_1d(n, src, dst, FFTW_ESTIMATE);
  if(plan == NULL) {
    fprintf(stderr, "unable to allocate serial fft plan\n");
    goto fail_immed;
  }

  // Do the fft.
  fftw_execute(plan);
  fft_r2c_1d_finish(dst, n);

  ret = SUCCESS;

  fftw_destroy_plan(plan);
fail_immed:
  return ret;
}

status do_parallel_fft(double * const src, double complex * const dst)
{
  status ret = FAILURE;
  fft_par_plan plan = fft_par_plan_r2c_1d(MPI_COMM_WORLD, proc_elems, src, dst,
      NULL);
  if(plan == NULL) {
    fprintf(stderr, "unable to allocate parallel fft plan\n");
    goto fail_immed;
  }

  ret = fft_par_execute(plan);
  if(ret != MPI_SUCCESS) {
    fprintf(stderr, "failure while performing parallel fft\n");
    goto fail_free_plan;
  }

  ret = SUCCESS;
fail_free_plan:
  if(fft_par_plan_destroy(plan) != MPI_SUCCESS) {
    fprintf(stderr, "unable to destroy parallel fft plan\n");
    ret = FAILURE;
  }
fail_immed:
  ret = SUCCESS;
  return ret;
}
