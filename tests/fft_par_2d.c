#include <complex.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <fftw3.h>
#include "dSFMT/dSFMT.h"
#include "fft_par.h"
#include "fft_ser.h"

const int nelems[2] = {51, 41};
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

// Allocate a master array and initialize it so that it's the same across the
// cluster. The master array should be freed with fftw_free.
int create_and_sync_master(MPI_Comm comm, dsfmt_t *rng, double **dst)
{
  *dst = NULL;
  int P[2], per[2], loc[2];
  int err = MPI_Cart_get(comm, 2, P, per, loc);
  if(err != MPI_SUCCESS) goto fail_immed;
  int rank;
  err = MPI_Comm_rank(comm, &rank);
  if(err != MPI_SUCCESS) goto fail_immed;

  const int master_len = P[0]*nelems[0]*P[1]*nelems[1];
  double *master = fftw_malloc(master_len*sizeof(double));
  if(master == NULL) goto fail_immed;

  /* First processor initializes */
  if(rank == 0) {
    for(int i = 0; i < master_len; ++i) {
      master[i] = dsfmt_genrand_close_open(rng);
    }
  }

  /* Synchronize it across the cluster */
  err = MPI_Bcast(master, master_len, MPI_DOUBLE, 0, comm);
  if(err != MPI_SUCCESS) goto fail_free_master;

  *dst = master;
  return err;

fail_free_master:
  fftw_free(master);
fail_immed:
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

  /* Allocate master array and synch across processes */
  double *master;
  err = create_and_sync_master(cart, prng, &master);
  if(master == NULL) {
    fprintf(stderr, "unable to create master array\n");
    goto die_free_prng;
  }

  /* Create a serial destination array */
  int P[2], per[2], loc[2];
  err = MPI_Cart_get(cart, 2, P, per, loc);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to retrieve cartesion topology configuration\n");
    goto die_free_master;
  }

  const int masterdim[2] = {P[0]*nelems[0], P[1]*nelems[1]};

  double complex *serial = fftw_malloc(
      masterdim[0]*masterdim[1]*sizeof(double complex));
  if(serial == NULL) {
    fprintf(stderr, "unable to allocate serial destination array\n");
    goto die_free_master;
  }


  /* Create a serial plan */
  fftw_plan serial_plan = fftw_plan_dft_r2c_2d(masterdim[0], masterdim[1],
      master, serial, FFTW_ESTIMATE);
  if(serial_plan == NULL) {
    fprintf(stderr, "unable to create serial plan\n");
    goto die_free_serial;
  }

  /* Time the serial transform */
  err = MPI_Barrier(cart);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to synchronize processes to measure serial fft\n");
    goto die_free_serial_plan;
  }
  double start_time = MPI_Wtime();
  for(int t = 0; t < TRIALS; ++t) {
    fftw_execute(serial_plan);
    fft_r2c_finish_packed(serial, 2, masterdim);
  }
  double end_time = MPI_Wtime();
  if(rank == 0) {
    printf("average serial time: %f\n", (end_time - start_time)/TRIALS);
  }

  /* Allocate the Parallel Source Array and Initialize */
  double *par_source = fftw_malloc(nelems[0]*nelems[1]*sizeof(double));
  if(par_source == NULL) {
    fprintf(stderr, "unable to allocate parallel source array\n");
    goto die_free_serial_plan;
  }
  for(int r = 0; r < nelems[0]; ++r) {
    for(int c = 0; c < nelems[1]; ++c) {
      par_source[r*nelems[1] + c] = master[
        ((r + loc[0]*nelems[0])*P[1] + loc[1])*nelems[1] + c];
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

  /* Time the parallel transform */
  err = MPI_Barrier(cart);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to synchronize processors\n");
    goto die_free_par_plan;
  }
  start_time = MPI_Wtime();
  for(int t = 0; t < TRIALS; ++t) {
    fft_par_execute_fwd(par_plan);
  }
  end_time = MPI_Wtime();
  if(rank == 0) {
    printf("average parallel time: %f\n", (end_time - start_time)/TRIALS);
  }

  /* Compare the two transforms to establish equality */
  double sup = 0.0;
  for(int r = 0; r < nelems[0]; ++r) {
    for(int c = 0; c < nelems[1]; ++c) {
      sup = fmax(sup, cabs(parallel[r*nelems[1] + c] -
            serial[((r + loc[0]*nelems[0])*P[1] + loc[1])*nelems[1] + c]));
    }
  }
  if(sup < 1.0e-6) {
    ret = EXIT_SUCCESS;
  }

  /* Error handling scheme changes here: now we have succeeded unless a cleanup
   * function chokes and dies.
   */
die_free_par_plan:
  fft_par_plan_destroy(par_plan);
die_free_parallel:
  fftw_free(parallel);
die_free_par_source:
  fftw_free(par_source);
die_free_serial_plan:
  fftw_destroy_plan(serial_plan);
die_free_serial:
  fftw_free(serial);
die_free_master:
  fftw_free(master);
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
