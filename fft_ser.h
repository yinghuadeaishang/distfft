/* Utility functions for serial FFTs. */
#ifndef HEADER_FFT_SER_INCLUDED
#define HEADER_FFT_SER_INCLUDED

#include <complex.h>
#include <stddef.h>

/* Only the values in indices 0 through N/2 (rounded down towards zero)
 * inclusive are filled by FFTW's real to complex DFT routines. This routine
 * takes a pointer to the resulting buffer and the number of elements in the
 * buffer, and computes the remaining values.
 */
void fft_r2c_1d_finish(double complex *v, size_t n);

#endif /* HEADER_FFT_SER_INCLUDED */
