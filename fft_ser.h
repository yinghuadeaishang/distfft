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
void fft_r2c_1d_finish(double complex *v, int n);

//! Fill in uncomputed values of a real-to-complex DFT of vectors.
/*! The FFTW real-to-complex fourier transform routines only compute the
 * positive frequencies of the DFT. This routine fills in the uncomputed
 * negative frequencies by exploiting the property that they are the complex
 * conjugates of the corresponding positive frequency. This is a vectorized
 * version of <tt>fft_r2c_1d_finish</tt> which itself is equivalent to calling
 * this function with vl = 1.
 *
 *  \param v The result of the fourier transform, with the positive and zero
 *  frequencies computed.
 *  \param n The size length of the array (in numbers of vectors).
 *  \param vl The length of each vector in the array
 */
void fft_r2c_1d_vec_finish(double complex *v, int n, int vl);

void fft_r2c_finish_unpacked(double complex *v, int dims, const int *size);
void fft_r2c_finish_unpacked_vec(double complex *v, int dims, const int *size,
    int VL);

void fft_r2c_finish_packed(double complex *v, int dims, const int *size);
void fft_r2c_finish_packed_vec(double complex *v, int dims, const int *size,
    int VL);

#endif /* HEADER_FFT_SER_INCLUDED */
