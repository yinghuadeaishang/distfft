#include "fft_ser.h"

void fft_r2c_1d_finish(complex double * const v, const size_t n)
{
  // Fill in the remaining values using the hermitian property
  for(unsigned int i = n/2 + 1; i < n; ++i) {
    v[i] = conj(v[n - i]);
  }
}
