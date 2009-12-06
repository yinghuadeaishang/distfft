/* Utility functions for serial FFTs. */
#ifndef HEADER_FFT_SER_INCLUDED
#define HEADER_FFT_SER_INCLUDED

#include <complex.h>
#include <stddef.h>

//! Fill in uncomputed values of a real-to-complex fourier transform
/*! The FFTW real-to-complex fourier transform routines only compute the
 *  positive frequencies of the DFT. This routine fills in the uncomputed
 *  negative frequencies by exploiting the property that they are the complex
 *  conjugates of the corresponding positive frequency.
 *
 *  \param v The result of the fourier transform, with the positive and zero
 *  frequencies computed.
 *  \param n The size length of the array.
 */
void fft_r2c_1d_finish(double complex *v, size_t n);

#endif /* HEADER_FFT_SER_INCLUDED */
