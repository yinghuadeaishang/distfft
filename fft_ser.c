#include <alloca.h>
#include "fft_ser.h"

void fft_r2c_1d_finish(double complex * const v, const int n)
{
  // Fill in the remaining values using the hermitian property
  for(unsigned int i = n/2 + 1; i < n; ++i) {
    v[i] = conj(v[n - i]);
  }
}

void fft_r2c_finish(double complex * const v, int const dims,
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
  for(int i = leading_size - 1; i >= 0; --i) {
    for(int j = last_len - 1; j >= 0; --j) {
      *(v + size[dims - 1]*i + j) = *(v + last_len*i + j);
    }
  }

  // Now to fill in the remaining values, we need to exploit hermitian
  // symmetry. Since we don't know how many dimensions we have, we'll use
  // multi-indices for looping.
  int *idx = alloca(dims*sizeof(int));

  // Initialize the multi index to the first value that needs to be filled.
  idx[dims - 1] = last_len;
  for(int d = dims - 2; d >= 0; --d) idx[d] = 0;

  // If we should continue the iteration
  int cont = idx[dims - 1] < size[dims - 1];

  int netsize = 1; for(int d = 0; d < dims; ++d) netsize *= size[d];

  while(cont) {
    // compute the destination from the row index using the row-major formula
    int dst = idx[0]; for(int d = 1; d < dims; ++d) dst = dst*size[d] + idx[d];
    // compute the offset of the index {size[0] - i[0], size[1] - i[1], ...}
    int src = idx[0] ? size[0] - idx[0] : 0;
    for(int d = 1; d < dims; ++d) {
      src = idx[d] ? (src + 1)*size[d] - idx[d] : src*size[d];
    }

    v[dst] = conj(v[src]);

    // Increment the multi-index. We increment the last index so that we write
    // to contiguous memory blocks.
    ++idx[dims-1];
    for(int d = dims - 1; d >= 1; --d) {
      if(idx[d] >= size[d]) {
        idx[d] = d == dims - 1 ? last_len : 0;
        ++idx[d-1];
      }
    }

    cont = idx[0] < size[0];
  }
}
