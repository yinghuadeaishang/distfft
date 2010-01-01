#include <alloca.h>
#include <string.h>
#include "fft_ser.h"

void fft_r2c_1d_finish(double complex * const v, const int n)
{
  // Fill in the remaining values using the hermitian property
  // v[i] = conj(v[n - i]) for i in [n/2+1,n) where n/2 is rounded downward
  double complex *dst = v + n/2 + 1;
  double complex *src = v + n - n/2 - 1;
  for(int i = n/2 + 1; i < n; ++i) {
    *dst++ = conj(*src--);
  }
}

void fft_r2c_1d_vec_finish(double complex * const v, const int n, const int vl)
{
  // Fill in the remaining values using the hermitian property
  double complex *dst = v + (n/2 + 1)*vl;
  double complex *src = v + (n - n/2 - 1)*vl;
  for(int i = n/2 + 1; i < n; ++i) {
    for(int j = 0; j < vl; ++j) {
      /* What we want to do is this:
       * v[i*vl + j] = conj(v[(n - i)*vl + j]);
       */
      *dst++ = conj(*src++);
    }
    /* Reset the source vector back 2 vector lengths. The inner loop has moved
     * us forward one vector length, so we need to subtract that away. The next
     * loop should start one vector length before the previous, so we need to
     * subtract yet another vector length.
     */
    src -= 2*vl;
  }
}

void fft_r2c_finish_packed(double complex * const v, int const dims,
    const int *const size)
{
  // The output we get from an r2c routine is a row major dimensioned matrix of
  // the form {size[0],size[1],...,size[dims-1]/2 + 1}. The first thing to do
  // is to sscatter this back into an array of size {size[0]...size[dims-1]}
  // with some holes in it.
  int leading_size = 1;
  for(int d = 0; d < dims - 1; ++d) leading_size *= size[d];
  int last_len = size[dims - 1]/2 + 1;

  // There are size[0]*size[1]*..size[dims-2] "rows" in the matrix. We move the
  // data into its final position from the back of the array, so we don't
  // overwrite the data we have yet to deal with. Additionally, we don't need
  // to worry about the first row, since it's already at index zero.
  for(int i = leading_size - 1; i > 0; --i) {
    memmove(v + size[dims - 1]*i, v + last_len*i,
        last_len*sizeof(complex double));
  }

  // Now we have the unpacked format
  fft_r2c_finish_unpacked(v, dims, size);
}

void fft_r2c_finish_packed_vec(double complex * const v, int const dims,
    const int *const size, const int VL)
{
  /* The output we get from an r2c routine is a row major dimensioned matrix of
   * the form {size[0],size[1],...,size[dims-1]/2 + 1} where each element is a
   * VL-length vector. The first thing to do is to scatter this back into an
   * array of VL-length vectors, the array being of size
   * {size[0]...size[dims-1]} with some holes in it.
   */
  int leading_size = 1;
  for(int d = 0; d < dims - 1; ++d) leading_size *= size[d];
  int last_len = size[dims - 1]/2 + 1;

  // There are size[0]*size[1]*..size[dims-2] "rows" in the matrix. We move the
  // data into its final position from the back of the array, so we don't
  // overwrite the data we have yet to deal with. Additionally, we don't need
  // to worry about the first row, since it's already at index zero.
  for(int i = leading_size - 1; i > 0; --i) {
    memmove(v + size[dims - 1]*i*VL, v + last_len*i*VL,
        last_len*VL*sizeof(complex double));
  }

  // Now we have the unpacked format
  fft_r2c_finish_unpacked_vec(v, dims, size, VL);
}

void fft_r2c_finish_unpacked(double complex * const v, int const dims,
    const int *const size)
{
  const int last_len = size[dims - 1]/2 + 1;
  // Now to fill in the remaining values, we need to exploit hermitian
  // symmetry. Since we don't know how many dimensions we have, we'll use
  // multi-indices for looping.
  int *idx = alloca(dims*sizeof(int));

  /* Cache the stride in each dimension. In row major format, the last dimension
   * is fastest, ie, stored contiguously, so initialize this stride to one. For
   * successive dimensions, going backward, the stride increases by a factor of
   * the length of the preceeding dimension.
   */
  int *stride = alloca(dims*sizeof(int));
  stride[dims - 1] = 1;
  for(int d = dims - 2; d >= 0; --d) {
    stride[d] = stride[d + 1]*size[d + 1];
  }

  // Initialize the multi index to the first value that needs to be filled.
  idx[dims - 1] = last_len;
  for(int d = dims - 2; d >= 0; --d) {
    idx[d] = 0;
  }

  // If we should continue the iteration
  int cont = 1;
  for(int d = 0; d < dims; ++d) {
    cont = cont && idx[d] < size[d];
  }

  double complex *dst = v + last_len;
  double complex *src = v + size[dims - 1] - last_len;

  while(cont) {
    *dst++ = conj(*src--);

    // Increment the multi-index. We increment the last index so that we write
    // to contiguous memory blocks.
    ++idx[dims-1];
    for(int d = dims - 1; d >= 1; --d) {
      if(idx[d] >= size[d]) {
        if(d == dims - 1) {
          idx[d] = last_len;
          dst += last_len;
          src += size[d] - last_len;
        } else {
          idx[d] = 0;
        }
        // If we're about to increment a zero index, we need to wrap to the
        // highest hyperrow in this dimension.
        if(idx[d-1] == 0) {
          src += size[d-1]*stride[d-1];
        }
        // Wrap to the next line
        src -= stride[d - 1];
        ++idx[d-1];
      }
    }

    cont = idx[0] < size[0];
  }
}

void fft_r2c_finish_unpacked_vec(double complex * const v, int const dims,
    const int *const size, const int VL)
{
  const int last_len = size[dims - 1]/2 + 1;
  // Now to fill in the remaining values, we need to exploit hermitian
  // symmetry. Since we don't know how many dimensions we have, we'll use
  // multi-indices for looping.
  int *idx = alloca(dims*sizeof(int));

  /* Cache the stride in each dimension. In row major format, the last dimension
   * is fastest, ie, stored contiguously, so initialize this stride to one. For
   * successive dimensions, going backward, the stride increases by a factor of
   * the length of the preceeding dimension.
   */
  int *stride = alloca(dims*sizeof(int));
  stride[dims - 1] = VL;
  for(int d = dims - 2; d >= 0; --d) {
    stride[d] = stride[d + 1]*size[d + 1];
  }

  // Initialize the multi index to the first value that needs to be filled.
  idx[dims - 1] = last_len;
  for(int d = dims - 2; d >= 0; --d) {
    idx[d] = 0;
  }

  // If we should continue the iteration
  int cont = 1;
  for(int d = 0; d < dims; ++d) {
    cont = cont && idx[d] < size[d];
  }

  double complex *dst = v + last_len*VL;
  double complex *src = v + (size[dims - 1] - last_len)*VL;

  while(cont) {
    for(int j = 0; j < VL; ++j) {
      *dst++ = conj(*src++);
    }
    src -= 2*VL;

    // Increment the multi-index. We increment the last index so that we write
    // to contiguous memory blocks.
    ++idx[dims-1];
    for(int d = dims - 1; d >= 1; --d) {
      if(idx[d] >= size[d]) {
        if(d == dims - 1) {
          idx[d] = last_len;
          dst += last_len*VL;
          src += (size[d] - last_len)*VL;
        } else {
          idx[d] = 0;
        }
        // If we're about to increment a zero index, we need to wrap to the
        // highest hyperrow in this dimension.
        if(idx[d-1] == 0) {
          src += size[d-1]*stride[d-1];
        }
        // Wrap to the next line
        src -= stride[d - 1];
        ++idx[d-1];
      }
    }

    cont = idx[0] < size[0];
  }
}
