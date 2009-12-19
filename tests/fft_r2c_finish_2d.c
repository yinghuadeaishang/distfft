#include <complex.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <fftw3.h>
#include "dSFMT/dSFMT.h"
#include "fft_ser.h"

static const int size[2] = {4, 5};
static const uint32_t SEED = 2323;

int main(int argc, char *argv[])
{
  int ret = EXIT_FAILURE;

  // Set up the PRNG
  dsfmt_t *dsfmt = malloc(sizeof(dsfmt_t));
  if(dsfmt == NULL) {
    fprintf(stdout, "unable to allocate PRNG\n");
    goto skip_deallocate_prng;
  }
  dsfmt_init_gen_rand(dsfmt, SEED);

  // Set up the source values
  double *src = fftw_malloc(size[0]*size[1]*sizeof(double));
  if(src == NULL) {
    fprintf(stdout, "unable to allocate source vector\n");
    goto skip_deallocate_src;
  }
  for(unsigned int i = 0; i < size[0]*size[1]; ++i) {
    src[i] = 2*dsfmt_genrand_open_close(dsfmt) - 1;
  }

  // Allocate the FFT destination array
  double complex *fft = fftw_malloc(size[1]*size[0]*sizeof(double complex));
  if(fft == NULL) {
    fprintf(stdout, "unable to allocate fft vector\n");
    goto skip_deallocate_fft;
  }

  // Execute the forward FFT
  fftw_plan fwd_plan = fftw_plan_dft_r2c(2, size, src, fft,
      FFTW_ESTIMATE);
  if(fwd_plan == NULL) {
    fprintf(stdout, "unable to allocate fft forward plan\n");
    goto skip_deallocate_fwd_plan;
  }
  fftw_execute(fwd_plan);

  // Fill in the rest of the destination values using the Hermitian property.
  fft_r2c_finish(fft, 2, size);

  // Allocate the reverse FFT destination array
  double complex *dst = fftw_malloc(size[0]*size[1]*sizeof(double complex));
  if(dst == NULL) {
    fprintf(stdout, "unable to allocate dst vector\n");
    goto skip_deallocate_dst;
  }

  // Perform the reverse FFT
  fftw_plan rev_plan = fftw_plan_dft_2d(size[0], size[1], fft, dst,
      FFTW_BACKWARD, FFTW_ESTIMATE);
  if(rev_plan == NULL) {
    fprintf(stdout, "unable to allocate fft reverse plan\n");
    goto skip_deallocate_rev_plan;
  }
  fftw_execute(rev_plan);

  // Compare the two vectors by sup norm
  double norm = 0.0;
  for(unsigned int i = 0; i < size[0]*size[1]; ++i) {
    // Divide the resulting by N, because FFTW computes the un-normalized DFT:
    // the forward followed by reverse transform scales the data by N.
    dst[i] /= size[0]*size[1];
    norm = fmax(norm, cabs(dst[i] - src[i]));
  }
  if(norm <= 1e-6) {
    ret = EXIT_SUCCESS;
  }

  fftw_destroy_plan(rev_plan);
skip_deallocate_rev_plan:
  fftw_free(dst);
skip_deallocate_dst:
  fftw_destroy_plan(fwd_plan);
skip_deallocate_fwd_plan:
  fftw_free(fft);
skip_deallocate_fft:
  fftw_free(src);
skip_deallocate_src:
  free(dsfmt);
skip_deallocate_prng:
  // Keep valgrind happy by having fftw clean up its internal structures. This
  // helps ensure we aren't leaking memory.
  fftw_cleanup();
  return ret;
}
