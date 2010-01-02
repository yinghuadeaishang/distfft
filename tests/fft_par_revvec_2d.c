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

const int proc_elems[2] = {113, 111};
const uint32_t SEED = 42;
const int VL = 5;
const int TRIALS = 5;

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

  /* Create cartestian communicator */
  int dims[2] = {0, 0};
  int periods[2] = {1, 1};
  err = MPI_Dims_create(size, 2, dims);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to create a cartestian topology\n");
    goto die_finalize_mpi;
  }
  MPI_Comm cart;
  err = MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to create cartestian communicator\n");
    goto die_finalize_mpi;
  }
  err = MPI_Comm_size(cart, &size);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to determine the size of the cartesian grid\n");
    goto die_free_cart_comm;
  }
  err = MPI_Comm_rank(cart, &rank);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to determine the rank in the cartestian grid\n");
    goto die_free_cart_comm;
  }
  if(rank == MPI_COMM_NULL) {
    ret = EXIT_SUCCESS;
    goto die_free_cart_comm;
  }

  dsfmt_t *prng = malloc(sizeof(dsfmt_t));
  if(prng == NULL) {
    fprintf(stderr, "unable to allocate PRNG\n");
    goto die_free_cart_comm;
  }
  dsfmt_init_gen_rand(prng, SEED + rank);

  int const net_elems = proc_elems[0]*proc_elems[1];
  // Allocate master source array for FFT.
  double *const master = fftw_malloc(net_elems*VL*sizeof(double));
  if(master == NULL) {
    fprintf(stderr, "unable to allocate master array\n");
    goto die_free_prng;
  }
  for(unsigned int i = 0; i < net_elems*VL; ++i) {
    master[i] = dsfmt_genrand_open_close(prng) * 10;
  }

  /* Allocate source array for serial array. We copy the master array to this
   * array, then transform it in place, then reverse transform it. The idea is
   * that we should get the original data back, and we use this as a consistency
   * check. We need the original data to compare to.
   */
  double *const source = fftw_malloc(net_elems*VL*sizeof(double));
  if(source == NULL) {
    fprintf(stderr, "unable to allocate source array\n");
    goto die_free_master;
  }
  for(int i = 0; i < net_elems*VL; ++i) source[i] = master[i];

  /* Allocate the destination array */
  double complex *const dest = fftw_malloc(net_elems*VL*sizeof(double complex));
  if(dest == NULL) {
    fprintf(stderr, "unable to allocate destination array\n");
    goto die_free_source;
  }

  /* Allocate a plan to compute the FFT */
  fft_par_plan plan = fft_par_plan_r2c(cart, proc_elems, VL, source,
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
  for(int i = 0; i < net_elems*VL; ++i) {
    /* Each FFT effectively multiplies by sqrt(net_elems*num_procs) */
    norm = fmax(norm, fabs(master[i] - source[i]/net_elems/size));
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
die_free_cart_comm:
  if(err == MPI_SUCCESS) err = MPI_Comm_free(&cart);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to free cartestian communicator\n");
    ret = EXIT_FAILURE;
  }
die_finalize_mpi:
  if(err == MPI_SUCCESS) err = MPI_Finalize();
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to finalize MPI\n");
    ret = EXIT_FAILURE;
  }
die_immed:
  fftw_cleanup();
  return ret;
}
