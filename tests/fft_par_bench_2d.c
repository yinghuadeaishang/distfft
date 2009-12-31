#include <complex.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <fftw3.h>
#include "dSFMT/dSFMT.h"
#include "fft_par.h"
#include "fft_ser.h"

const int nelems[2] = {1131, 1113};
const uint32_t SEED = 42;
const int TRIALS = 50;

int tile_cartestian(MPI_Comm *cart)
{
  int size;
  int err = MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(err != MPI_SUCCESS) goto fail_immed;

  int dims[2] = {0, 0};
  err = MPI_Dims_create(size, 2, dims);
  if(err != MPI_SUCCESS) goto fail_immed;

  int periodic[2] = {0, 0};
  MPI_Comm comm;
  err = MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 1, &comm);
  if(err != MPI_SUCCESS) goto fail_immed;

  /* Successful return */
  *cart = comm;
  return err;

  /* Failure */
  if(err == MPI_SUCCESS) err = MPI_Comm_free(&comm);
fail_immed:
  *cart = MPI_COMM_NULL;
  return err;
}

int main(int argc, char **argv)
{
  // Error handling scheme: this function has failed until proven otherwise.
  int ret = EXIT_FAILURE;

  int err = MPI_Init(&argc, &argv);
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

  MPI_Comm cart;
  err = tile_cartestian(&cart);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to form cartesian topology.\n");
    goto die_finalize_mpi;
  }

  int rank;
  err = MPI_Comm_rank(cart, &rank);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to determinte rank in cartesian topology\n");
    goto die_finalize_mpi;
  }
  /* Die gracefully if we're not needed */
  if(rank == MPI_COMM_NULL) {
    ret = EXIT_SUCCESS;
    goto die_finalize_mpi;
  }

  dsfmt_t *prng = malloc(sizeof(dsfmt_t));
  if(prng == NULL) {
    fprintf(stderr, "unable to allocate PRNG\n");
    goto die_free_cart;
  }
  dsfmt_init_gen_rand(prng, SEED + rank);

  /* Allocate the Parallel Source Array and Initialize */
  double *par_source = fftw_malloc(nelems[0]*nelems[1]*sizeof(double));
  if(par_source == NULL) {
    fprintf(stderr, "unable to allocate parallel source array\n");
    goto die_free_prng;
  }
  for(int r = 0; r < nelems[0]; ++r) {
    for(int c = 0; c < nelems[1]; ++c) {
      par_source[r*nelems[1] + c] = dsfmt_genrand_close_open(prng);
    }
  }

  /* Allocate the parallel destination array */
  double complex *parallel = fftw_malloc(
      nelems[0]*nelems[1]*sizeof(double complex));
  if(parallel == NULL) {
    fprintf(stderr, "unable to allocate parallel destination array\n");
    goto die_free_par_source;
  }
  for(int i = 0; i < nelems[0]*nelems[1]; ++i) parallel[i] = 0.0;

  /* Create a parallel plan */
  fft_par_plan par_plan = fft_par_plan_r2c(cart, nelems, 1, par_source,
      parallel, &err);
  if(par_plan == NULL) {
    fprintf(stderr, "unable to allocate parallel plan\n");
    goto die_free_parallel;
  }

  /* Before starting time trails, do a run in order to touch the allocated
   * memory. Memory on modern systems is mapped into address space on a call for
   * memory, but not mapped to physical memory until it is touched. An untimed
   * run ensures all the pages are allocated.
   */
  fft_par_execute_fwd(par_plan);
  /* Time the parallel transform */
  err = MPI_Barrier(cart);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to synchronize processors\n");
    goto die_free_par_plan;
  }
  double start_time = MPI_Wtime();
  for(int t = 0; t < TRIALS; ++t) {
    fft_par_execute_fwd(par_plan);
  }
  double end_time = MPI_Wtime();
  if(rank == 0) {
    printf("average parallel time: %f\n", (end_time - start_time)/TRIALS);
  }

  ret = EXIT_SUCCESS;

  /* Error handling scheme changes here: now we have succeeded unless a cleanup
   * function chokes and dies.
   */
die_free_par_plan:
  fft_par_plan_destroy(par_plan);
die_free_parallel:
  fftw_free(parallel);
die_free_par_source:
  fftw_free(par_source);
die_free_prng:
  free(prng);
die_free_cart:
  if(err == MPI_SUCCESS) err = MPI_Comm_free(&cart);
die_finalize_mpi:
  if(err == MPI_SUCCESS && MPI_Finalize() != MPI_SUCCESS) {
    fprintf(stderr, "unable to finalize MPI\n");
    ret = EXIT_FAILURE;
  }
die_immed:
  fftw_cleanup();
  return ret;
}
