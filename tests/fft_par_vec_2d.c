#include <complex.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <fftw3.h>
#include "dSFMT/dSFMT.h"
#include "fft_par.h"
#include "fft_ser.h"

const int nelems[2] = {51, 41};
const int VL = 3;
const uint32_t SEED = 42;

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

  /* Create two dimensional cartestian processor layout */
  int size;
  err = MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to determine global processor rank\n");
    goto die_finalize_mpi;
  }
  int pdims[2] = {0, 0};
  err = MPI_Dims_create(size, 2, pdims);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to determine processor layout\n");
    goto die_finalize_mpi;
  }
  int periods[2] = {1,1};
  MPI_Comm cart;
  err = MPI_Cart_create(MPI_COMM_WORLD, 2, pdims, periods, 1, &cart);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to create cartestian communicator\n");
    goto die_finalize_mpi;
  }
  int rank;
  err = MPI_Comm_rank(cart, &rank);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to determine cartestian processor rank\n");
    goto die_free_cart;
  }
  if(rank == MPI_COMM_NULL) {
    /* This process is not in the cartestian grid */
    ret = EXIT_SUCCESS;
    goto die_free_cart;
  }
  err = MPI_Comm_size(cart, &size);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to determine cartestian processor grid size\n");
    goto die_free_cart;
  }

  /* Create master source array */
  const int master_elems = pdims[0]*nelems[0]*pdims[1]*nelems[1];
  double *const master = fftw_malloc(master_elems*VL*sizeof(double));
  if(master == NULL) {
    fprintf(stderr, "unable to allocate master source array\n");
    goto die_free_cart;
  }

  dsfmt_t *const prng = malloc(sizeof(dsfmt_t));
  if(prng == NULL) {
    fprintf(stderr, "unable to allocate prng\n");
    goto die_free_master;
  }
  dsfmt_init_gen_rand(prng, SEED);
  for(int i = 0; i < master_elems*VL; ++i) {
    master[i] = 2*(dsfmt_genrand_open_open(prng) - 0.5);
  }

  // TODO: Delete this printme code
  for(int p = 0; p < size; ++p) {
    if(p == rank) {
      for(int r = 0; r < pdims[0]*nelems[0]; r++) {
        printf("master(%d) %d:", rank, r);
        for(int c = 0; c < pdims[1]*nelems[1]; c++) {
          printf(" (%g", master[(r*pdims[1]*nelems[1] + c)*VL]);
          for(int j = 1; j < VL; ++j) {
            printf(",%g", master[(r*pdims[1]*nelems[1] + c)*VL + j]);
          }
          printf(") ");
        }
        printf("\n");
      }
    }
    MPI_Barrier(cart);
  }

  /* Allocate serial destination array */
  double complex *const serial = fftw_malloc(
      master_elems*VL*sizeof(double complex));
  if(serial == NULL) {
    fprintf(stderr, "unable to allocate serial destination array\n");
    goto die_free_prng;
  }
  /* Carry out serial transform */
  int serdim[2] = {pdims[0]*nelems[0], pdims[1]*nelems[1]};
  fftw_plan splan = fftw_plan_many_dft_r2c(2, serdim, VL, master, NULL, VL, 1,
      serial, NULL, VL, 1, FFTW_ESTIMATE);
  if(splan == NULL) {
    fprintf(stderr, "unable to create serial vector transform plan\n");
    goto die_free_serial;
  }
  fftw_execute(splan);

  /* Create parallel source array and fill it withV the appropriate rows from
   * master
   */
  double *const psrc = fftw_malloc(nelems[0]*nelems[1]*VL*
      sizeof(double complex));
  if(psrc == NULL) {
    fprintf(stderr, "unable to allocate parallel source array\n");
    goto die_free_splan;
  }
  int loc[2];
  err = MPI_Cart_coords(cart, rank, 2, loc);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to determine processor location\n");
    goto die_free_psrc;
  }
  for(int r = 0; r < nelems[0]; ++r) {
    memcpy(psrc + r*nelems[1]*VL,
        master + ((r + loc[0]*nelems[0])*serdim[1] + loc[1]*nelems[1])*VL,
        nelems[1]*VL*sizeof(double));
  }

  // // TODO: Delete this printme code
  // for(int p = 0; p < size; ++p) {
  //   if(p == rank) {
  //     for(int r = 0; r < nelems[0]; r++) {
  //       printf("psrc(%d) %d:", rank, r);
  //       for(int c = 0; c < nelems[1]; c++) {
  //         printf(" (%g", psrc[(r*nelems[1] + c)*VL]);
  //         for(int j = 1; j < VL; ++j) {
  //           printf(",%g", psrc[(r*nelems[1] + c)*VL + j]);
  //         }
  //         printf(") ");
  //       }
  //       printf("\n");
  //     }
  //   }
  //   MPI_Barrier(cart);
  // }

  /* Allocate parallel destination array */
  double complex *const parallel = fftw_malloc(nelems[0]*nelems[1]*VL*
      sizeof(double complex));
  if(parallel == NULL) {
    fprintf(stderr, "unable to allocate parallel destination array\n");
    goto die_free_psrc;
  }

  /* Plan and execute parallel transform */
  fft_par_plan pplan = fft_par_plan_r2c(cart, nelems, VL, psrc, parallel,
      &err);
  if(pplan == NULL || err != MPI_SUCCESS) {
    fprintf(stderr, "unable to allocate parallel transform plan\n");
    goto die_free_psrc;
  }

  err = fft_par_execute_fwd(pplan);
  if(err != MPI_SUCCESS) {
    fprintf(stderr, "unable to execute parallel transform\n");
    goto die_free_pplan;
  }

  /* Compare the two results using sup norm */
  double sup = 0.0;
  for(int r = 0; r < nelems[0]; ++r) {
    for(int c = 0; c < nelems[1]*VL; ++c) {
      sup = fmax(sup, cabs(psrc[r*nelems[1]*VL + c] -
        master[((r + loc[0]*nelems[0])*serdim[1] + loc[1]*nelems[1])*VL + c]));
      }
  }
  if(sup < 1.0e-6) {
    ret = EXIT_SUCCESS;
  }

die_free_pplan:
  fft_par_plan_destroy(pplan);
die_free_psrc:
  fftw_free(psrc);
die_free_splan:
  fftw_destroy_plan(splan);
die_free_serial:
  fftw_free(serial);
die_free_prng:
  free(prng);
die_free_master:
  fftw_free(master);
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
