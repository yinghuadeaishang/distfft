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
const int VL = 13;
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
  dsfmt_init_gen_rand(prng, SEED);

  const int master_elems = proc_elems * size;

  double *const master = fftw_malloc(VL*master_elems*sizeof(double));
  if(master == NULL) {
    fprintf(stderr, "unable to allocate master array\n");
    goto die_free_prng;
  }
  for(int i = 0; i < master_elems*VL; ++i) {
    master[i] = 2*dsfmt_genrand_open_close(prng) - 1;
  }

  /* Allocate the array holding the serial result */
  double complex *const serial = fftw_malloc(VL*master_elems*sizeof(*serial));
  if(serial == NULL) {
    fprintf(stderr, "unable to allocate serial array\n");
    goto die_free_master;
  }

  /* Perform serial transform */
  fftw_plan serial_plan = fftw_plan_many_dft_r2c(1, &master_elems, VL,
      master, NULL, VL, 1, serial, NULL, VL, 1, FFTW_ESTIMATE);
  if(serial_plan == NULL) {
    fprintf(stderr, "unable to construct forward transform plan\n");
    goto die_free_serial;
  }

  /* Perform the serial transform, and complete it */
  fftw_execute(serial_plan);
  fft_r2c_1d_vec_finish(serial, master_elems, VL);

  /* Allocate space to hold the parallel transform result */
  double complex *const parallel = fftw_malloc(
      proc_elems*VL*sizeof(double complex));
  if(parallel == NULL) {
    fprintf(stderr, "unable to allocate space for parallel array\n");
    goto die_destroy_serial_plan;
  }

  /* Create the parallel plan */
  fft_par_plan par_plan = fft_par_plan_r2c_1d(MPI_COMM_WORLD, proc_elems, VL,
      master + rank*proc_elems*VL, parallel, NULL);
  if(par_plan == NULL) {
    fprintf(stderr, "unable to create parallel transform plan\n");
    goto die_free_parallel;
  }

  /* Execute the parallel transform */
  if(fft_par_execute_fwd(par_plan) != MPI_SUCCESS) {
    fprintf(stderr, "unable to execute parallel transform\n");
    goto die_destroy_par_plan;
  }

  /* Compare values */
  int sup = 0.0;
  for(int i = 0; i < proc_elems*VL; ++i) {
    sup = fmax(sup, cabs(serial[rank*proc_elems*VL + i] - parallel[i]));
  }
  if(sup < 1.0e-6) {
    ret = EXIT_SUCCESS;
  }

die_destroy_par_plan:
  fft_par_plan_destroy(par_plan);
die_free_parallel:
  fftw_free(parallel);
die_destroy_serial_plan:
  fftw_destroy_plan(serial_plan);
die_free_serial:
  fftw_free(serial);
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
