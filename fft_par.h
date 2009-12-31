#ifndef HEADER_FFT_PAR_INCLUDED
#define HEADER_FFT_PAR_INCLUDED

#include <complex.h>
#include <stddef.h>
#include <mpi.h>

//! An opaque pointer type hiding the details of a parallel Fourier transform.
/*! Values of this type are obtained from the <tt>fft_par_plan_*</tt> methods,
 *  which allocate the necessary storage and MPI structures for executing the
 *  transform.
 *
 *  Any plan created by the user should be destroyed using
 *  fft_par_plan_destroy(fft_par_plan) to release system resources.
 */
typedef struct fft_par_plan_s *fft_par_plan;

//! Create a plan to compute a parallel one dimensional fourier transform.
/*! Create a plan to compute a one dimensional fourier transform of data which
 *  is distributed across several processors.
 *
 *  This is a collective operation: all processors in the communicator should
 *  call this function with the same arguments for <tt>nelems</tt>.
 *
 *  This function can encounter errors from two sources: the MPI system or
 *  memory allocation. If there is any problem creating the plan, then
 *  <tt>NULL</tt> is returned. If the <tt>err</tt> parameter is not
 *  <tt>NULL</tt>, it is taken as the location to store the MPI return status.
 *  If this function returns null, and <tt>MPI_SUCCESS</tt> was returned in
 *  <tt>err</tt> then the error was due to a failure to allocate memory.
 *
 *  \param comm The communicator consisting of the nodes over which the data is
 *  distributed.
 *  \param nelems The number of elements on each processor.
 *  \param src The source array
 *  \param dst The destination array
 *  \param err The return status from the MPI system.
 *  \return A new plan, or <tt>NULL</tt> if there was an error creating the plan.
 */
fft_par_plan fft_par_plan_r2c_1d(MPI_Comm comm, int nelems, double *src,
     double complex *dst, int *err);

//! Create a plan to compute a distributed discrete fourier transform.
/*! Create a plan to compute a discrete fourier transform of data which is
 *  distributed across processors. The transform will be computed in parallel.
 *
 *  This is a collective operation: all processors in the communicator should
 *  call this function with the same arguments for <tt>nelems</tt> and
 *  <tt>len</tt>.
 *
 *  Errors can be obtained from two sources: the MPI system or memory
 *  allocation. If there are any problems creating the plan, then <tt>NULL</tt>
 *  will be returned. If the <tt>err</tt> parameter is not <tt>NULL</tt>, the
 *  return code from the MPI operations are returned in this location. If the
 *  function returns null, but <tt>MPI_SUCCESS</tt> is returned in the error
 *  location, then the erro was do to a failure to allocate memory.
 *
 *  \param comm The communicator consisting of nodes over which the data is
 *  distributed. This is assumed to have cartesian topology information attached
 *  which reflects the distribution of data across the processors. In
 *  particular, the number of dimensions in this transform is taken as the
 *  rank of the transform (the number of dimensions).
 *  \param size An array of the length of data in the corresponding direction
 *  <i>on each processor.</i>
 *  \param len The number of data elements which are to be transformed in
 *  parallel. For example, if you are fourier transforming an array of
 *  three-dimensional vectors, set this to three.
 *  \param err The return status from the MPI system. If <tt>NULL</tt> is given,
 *  then the MPI return status is not returned.
 *  \return A new plan, or <tt>NULL</tt> if there was an error creating the
 *  plan.
 */
fft_par_plan fft_par_plan_r2c(MPI_Comm comm, const int *size, int len,
    double *src, double complex *dst, int *err);

//! Execute a plan
/*! Accept a reference to a previously created plan and execute it on the data
 *  provided when the plan was created.
 *
 *  This is a collective operation: all processors in the communicator should
 *  call this function with the plan which they all obtained by synchronous
 *  calls to <tt>fft_par_plan_r2c</tt>
 *
 *  \param plan The plan to execute
 *  \return The MPI status code
 */
int fft_par_execute_fwd(fft_par_plan);

//! Destroy a plan for computing parallel fourier transforms.
/*! The plans contain buffers and other system resources which need to be
 *  freed. Any plan allocation should be matched with this function to
 *  deallocate the plan when finished.
 *
 *  \param plan The plan to destroy.
 *  \return The return status from the underlying MPI system.
 */
int fft_par_plan_destroy(fft_par_plan plan);

#endif /* HEADER_FFT_PAR_INCLUDED */
