#define _XOPEN_SOURCE 500
#include <alloca.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <fftw3.h>
#include "fft_par.h"
#include "fft_ser.h"

struct fft_par_plan_s {
  void *src;
  void *srcbuffer;
  void *dstbuffer;
  void *dst;
  /* Communication Displacements */
  int *cycle_senddispls;
  int *cycle_recvdispls;
  int *swizzle_senddispls;
  int *swizzle_recvdispls;
  /* Communication Types */
  MPI_Datatype *cycle_sendtypes;
  MPI_Datatype *cycle_recvtypes;
  MPI_Datatype *swizzle_sendtypes;
  MPI_Datatype *swizzle_recvtypes;
  /* Persistent communication requests for asynch communication */
  MPI_Request *swizzle_sendreqs_fwd;
  MPI_Request *swizzle_recvreqs_fwd;
  MPI_Request *swizzle_sendreqs_rev;
  MPI_Request *swizzle_recvreqs_rev;
  /* FFT Plans */
  fftw_plan innerplan_fwd;
  fftw_plan outerplan_fwd;
  fftw_plan innerplan_rev;
  fftw_plan outerplan_rev;
  int *ones;
  int *map; /* [procs][ndims] location of each processor */
  int *outerjobs; /* [procs][ndims] Number of outer jobs on each processor */
  int *nelems; /* [ndims] Number of elements in each direction */
  int *nprocs; /* [ndims] Number of processors in each direction */
  int ndims; /* How many dimensions there are */
  int veclen; /* The length of each element (parallel transforms) */
  int procid;
  MPI_Comm comm;
};

static inline int max(int a, int b)
{
  return (a > b ? a : b);
}

//! Compute the send and receive offsets and types for a cyclic permutation.
/*! Compute the send and receive offsets and types for a given processor which
 *  will be used by <tt>MPI_Alltoallw</tt> to perform a cyclic data permutation.
 *  The comm, map, procs, ndims, nelem, and len members of the plan should be
 *  set to the values for the communication.
 *
 *  \param plan The plan to write the cyclic_{send,recv}{displ,type}s members
 *  of. This function makes no attempt to avoid overwriting these values in case
 *  of failure.
 *  \return The MPI return code
 */
int cycle_data_parameter_fill(fft_par_plan plan);

//! Compute the send and receive offsets and types for swizzling data.
/*! Compute the send and receive offsets and types for a given processor which
 *  will be used by <tt>MPI_Alltoallw</tt> to perform second stage data
 *  swizzling. The cyc_[sr][dt] members of the plan structure should be
 *  initialized to arrays of at least plan->nprocs length, and the nprocs,
 *  rank, and outerjobs members of the plan should be set to the values for the
 *  communication.
 *
 *  \param plan The plan to write the swiz_[sr][dt] members of.
 *  \return The MPI return code
 */
int swizzle_data_parameter_fill(fft_par_plan plan);

//! Initialize the persistent send/recv requests for second stage communication
/*! Initialize the persistent send and receive requests for asynchronous second
 * stage communication. Assumes that the swizzle_{send,recv}reqs buffers of plan
 * have been initialized.
 *
 * \param plan The plan in which the swizzle_{send,recv}reqs members will be
 * written.
 * \return The MPI return code
 */
int fft_par_init_swizzle_reqs(fft_par_plan plan);

//! Apply the twiddle factors to the intermediate data within a plan.
/*! In order to compose two smaller length fourier transforms into a fourier
 *  transform of a larger composite data set, we need to multiply twiddle
 *  factors onto the result of the inner fourier transform before performing
 *  the outer fourier transform. This routine multiplies the data in the
 *  outersrc array by the appropriate complex values.
 *
 *  The process proceeds orthogonally for each rank, and operates on the data
 *  coming from a single specified processor.
 *
 *  \param plan The plan owning the outersrc array to apply twiddle factors to.
 *  \param p The elements coming from processor p have the twiddle factors
 *  applied.
 *  \return the MPI error code, if any
 */
typedef enum {
  TWIDDLE_FORWARD,
  TWIDDLE_REVERSE
} twiddle_dir;
int apply_twiddle_factors(fft_par_plan plan, int p, twiddle_dir dir)
{
  int procid;
  int err = MPI_Comm_rank(plan->comm, &procid);
  if(err != MPI_SUCCESS) goto fail_immed;

  const int dims = plan->ndims;
  const int *const procs = plan->nprocs;
  const int *const elems = plan->nelems;
  const int *const ojobs = plan->outerjobs + procid*dims;
  const int *const loc = plan->map + procid*dims;
  const int vl = plan->veclen;

  /* Multi-index over the work unit */
  int *wu = alloca(dims*sizeof(int));

  /* Lenght of a workunit */
  int wulen = vl;
  for(int d = 0; d < dims; ++d) { wulen *= procs[d]; }

  /* Zero the multi-index */
  for(int d = 0; d < dims; ++d) { wu[d] = 0; }

  /* The processor index */
  const int *const pidx = plan->map + p*dims;

  /* The displacement of this processor's element within a workunit (in
   * multiples of veclen)
   */
  int displ = 0;
  for(int d = 0; d < dims; ++d) {
    displ = displ*procs[d] + pidx[d];
  }

  /* The current target element we are applying the swizzle factor to */
  double complex *tgt = (double complex *)plan->srcbuffer + displ*vl;

  /* Root of unity in each direction */
  double complex *w0 = alloca(dims*sizeof(complex double));
  switch(dir) {
    case TWIDDLE_FORWARD:
      for(int d = 0; d < dims; ++d) {
        w0[d] = cexp(-2*M_PI*I/elems[d]/procs[d]);
      }
      break;
    case TWIDDLE_REVERSE:
    default:
      for(int d = 0; d < dims; ++d) {
        w0[d] = cexp(2*M_PI*I/elems[d]/procs[d]);
      }
      break;
  }

  /* What to multiply the twiddle factor by each iteration, the ratio of twiddle
   * factors between successive work units.
   */
  double complex *wprod = alloca(dims*sizeof(double complex));
  for(int d = 0; d < dims; ++d) {
    wprod[d] = cpow(w0[d], pidx[d]*procs[d]);
  }

  /* The current twiddle factor */
  double complex *w = alloca(dims*sizeof(double complex));
  for(int d = 0; d < dims; ++d) {
    w[d] = cpow(w0[d], pidx[d]*loc[d]);
  }

  int cont = 1; /* Continue iteration over work units */
  for(int d = 0; d < dims; ++d) { cont = cont && wu[d] < ojobs[d]; }
  while(cont) {
    /* Apply twiddle factors */
    for(int j = 0; j < vl; ++j) {
      for(int d = 0; d < dims; ++d) {
        /* We essentially want to perform this operation
         * tgt[j] *= cpow(w0[d], pidx[d]*(wu[d]*procs[d] + loc[d]));
         */
        tgt[j] *= w[d];
      }
    }

    /* Increment the work unit's outermost indices, and spill backwards.
     * Increment the twiddle factor as we go
     */
    ++wu[dims - 1];
    w[dims - 1] *= wprod[dims - 1];
    for(int d = dims - 1; d >= 1; --d) {
      if(wu[d] >= ojobs[d]) {
        wu[d] = 0;
        /* Reset twiddle factor in this dimension */
        w[d] = cpow(w0[d], pidx[d]*loc[d]);
        ++wu[d-1];
        w[d-1] *= wprod[d-1];
      }
    }
    cont = wu[0] < ojobs[0];
    /* Increment the target */
    tgt += wulen;
  }

fail_immed:
  return err;
}

fft_par_plan fft_par_plan_r2c_nc(MPI_Comm comm, int ndims, const int *proc_dim,
    const int VL, const int *map, const int *size, double *src,
    complex double *dst, int *errloc);

fft_par_plan fft_par_plan_r2c_1d(MPI_Comm comm, int const nelems, int const vl,
    double * const src, double complex * const dst, int * const errloc)
{
  fft_par_plan plan = NULL;
  int size;
  int err = MPI_Comm_size(comm, &size);
  if(err != MPI_SUCCESS) goto fail_immed;

  /* Create a trivial map for the one dimensional case */
  int *const map = malloc(size*sizeof(int));
  for(int i = 0; i < size; ++i) { map[i] = i; }

  plan = fft_par_plan_r2c_nc(comm, 1, &size, vl, map, &nelems,
      src, dst, &err);

  free(map);
fail_immed:
  if(errloc != NULL) *errloc = err;
  return plan;
}

fft_par_plan fft_par_plan_r2c(MPI_Comm comm, const int *const size,
    const int vl, double *const src, double complex *const dst,
    int *const errloc)
{
  fft_par_plan plan = NULL;
  int ndims;
  int err = MPI_Cartdim_get(comm, &ndims);
  if(err != MPI_SUCCESS) {
    goto fail_immed;
  }

  int *proc_dim = malloc(ndims*sizeof(int));
  if(proc_dim == NULL) goto fail_immed;

  int *period = malloc(ndims*sizeof(int));
  if(period == NULL) goto fail_free_proc_dim;

  int *loc = malloc(ndims*sizeof(int));
  if(loc == NULL) goto fail_free_period;

  err = MPI_Cart_get(comm, ndims, proc_dim, period, loc);
  if(err != MPI_SUCCESS) goto fail_free_loc;

  int procs;
  err = MPI_Comm_size(comm, &procs);
  if(err != MPI_SUCCESS) goto fail_free_loc;

  int *map = malloc(sizeof(int)*ndims*procs);
  if(map == NULL) goto fail_free_loc;

  /* Fill in the processor map */
  for(int p = 0; p < procs; ++p) {
    err = MPI_Cart_coords(comm, p, ndims, map + p*ndims);
    if(err != MPI_SUCCESS) goto fail_free_map;
  }

  // Call this helper function to actually initialize, it doesn't assume
  // cartesian topology information is attached to the communicator.
  plan = fft_par_plan_r2c_nc(comm, ndims, proc_dim, vl, map, size,
      src, dst, &err);

  /* Resources freed when this function failed */
fail_free_map:
  free(map);
fail_free_loc:
  free(loc);
fail_free_period:
  free(period);
fail_free_proc_dim:
  free(proc_dim);
fail_immed:
  if(errloc != NULL) *errloc = err;
  return plan;
}

fft_par_plan fft_par_plan_r2c_nc(MPI_Comm comm, const int ndims,
    const int *const proc_dim, const int vl, const int *const map, const int
    *const size, double * const src, double complex * const dst, int *errloc)
{
  int rank;
  int err = MPI_Comm_rank(comm, &rank);
  if(err != MPI_SUCCESS) goto fail_immed;

  fft_par_plan plan = malloc(sizeof(struct fft_par_plan_s));
  if(plan == NULL) goto fail_immed;

  plan->comm = comm;
  plan->ndims = ndims;
  plan->src = src;
  plan->dst = dst;
  plan->procid = rank;
  plan->veclen = vl;

  /* The total number of data elements on each processor */
  int netlen = 1; for(int i = 0; i < ndims; ++i) netlen *= size[i];

  /* The total number of processors on the grid */
  int nprocs = 1; for(int i = 0; i < ndims; ++i) nprocs *= proc_dim[i];

  /* Compute the number of second stage work units in each direction for each
   * processor.
   */
  plan->outerjobs = malloc(nprocs*ndims*sizeof(int));
  if(plan->outerjobs == NULL) goto fail_free_plan;
  for(int p = 0; p < nprocs; ++p) {
    const int * const p_loc = map + p*ndims; // The coordinates of processor p
    for(int d = 0; d < ndims; ++d) {
      *(plan->outerjobs + p*ndims + d) = size[d]/proc_dim[d];

      if(p_loc[d] < size[d] % proc_dim[d]) {
        ++*(plan->outerjobs + p*ndims + d);
      }
    }
  }
  /* For convenience, cache the total number of outer jobs for this processor */
  int ojobs = 1;
  for(int d = 0; d < ndims; ++d) ojobs *= *(plan->outerjobs + rank*ndims + d);

  /* Copy the processor map */
  plan->map = malloc(nprocs*ndims*sizeof(int));
  if(plan->map == NULL) goto fail_free_outerjobs;
  /* We don't use memcpy because memcpy's source parameter isn't const void *,
   * thus the compiler has a hissy fit because we discard the const qualifier
   * on the input map.
   */
  for(int i = 0; i < nprocs*ndims; ++i) plan->map[i] = map[i];

  /* Copy the number of elements in each direction */
  plan->nelems = malloc(ndims*sizeof(int));
  if(plan->nelems == NULL) goto fail_free_map;
  for(int d = 0; d < ndims; ++d) plan->nelems[d] = size[d];

  /* Copy the number of processors in each direction */
  plan->nprocs = malloc(ndims*sizeof(int));
  if(plan->nprocs == NULL) goto fail_free_nelems;
  for(int d = 0; d < ndims; ++d) plan->nprocs[d] = proc_dim[d];

  /* Allocate the storage arrays */
  const int maxlen = max(netlen, ojobs*nprocs);
  plan->srcbuffer = fftw_malloc(vl*maxlen*sizeof(double complex));
  if(plan->srcbuffer == NULL) goto fail_free_nprocs;
  plan->dstbuffer = fftw_malloc(vl*maxlen*sizeof(double complex));
  if(plan->dstbuffer == NULL) goto fail_free_srcbuffer;

  /* Allocate an array of ones to use in MPI_Alltoallw as the send count. The
   * types encode all the information.
   */
  plan->ones = malloc(nprocs*sizeof(int));
  if(plan->ones == NULL) goto fail_free_dstbuffer;
  for(int i = 0; i < nprocs; ++i) plan->ones[i] = 1;

  /* Array of MPI requests to be used in send/recv operations for the second
   * stage communication
   */
  plan->swizzle_sendreqs_fwd = malloc(nprocs*sizeof(MPI_Request));
  if(plan->swizzle_sendreqs_fwd == NULL) goto fail_free_ones;

  plan->swizzle_recvreqs_fwd = malloc(nprocs*sizeof(MPI_Request));
  if(plan->swizzle_recvreqs_fwd == NULL) goto fail_free_swizzle_sendreqs_fwd;

  plan->swizzle_sendreqs_rev = malloc(nprocs*sizeof(MPI_Request));
  if(plan->swizzle_sendreqs_rev == NULL) goto fail_free_swizzle_recvreqs_fwd;

  plan->swizzle_recvreqs_rev = malloc(nprocs*sizeof(MPI_Request));
  if(plan->swizzle_recvreqs_rev == NULL) goto fail_free_swizzle_sendreqs_rev;

  /* Initialize the cyclic communication types */
  err = cycle_data_parameter_fill(plan);
  if(err != MPI_SUCCESS) goto fail_free_swizzle_recvreqs_rev;

  /* Initialize the swizzle communication types */
  err = swizzle_data_parameter_fill(plan);
  if(err != MPI_SUCCESS) goto fail_free_cycle_params;

  plan->innerplan_fwd = fftw_plan_many_dft_r2c(ndims, size, vl,
      plan->srcbuffer, size, vl, 1, plan->dstbuffer, size, vl, 1,
      FFTW_ESTIMATE);
  if(plan->innerplan_fwd == NULL) goto fail_free_swizzle_params;

  plan->innerplan_rev = fftw_plan_many_dft_c2r(ndims, size, vl,
      plan->dstbuffer, size, vl, 1, plan->srcbuffer, size, vl, 1,
      FFTW_ESTIMATE);
  if(plan->innerplan_rev == NULL) goto fail_free_innerplan_fwd;

  /* Initialize the outer plans */
  fftw_iodim howmany[2] = {
    { .n = ojobs, .is = vl*nprocs, .os = vl*nprocs },
    { .n = vl, .is = 1, .os = 1}
  };
  fftw_iodim *outerdims = malloc(ndims*sizeof(fftw_iodim));
  if(outerdims == NULL) goto fail_free_innerplan_rev;
  outerdims[ndims - 1].n = plan->nprocs[ndims - 1];
  outerdims[ndims - 1].is = vl;
  outerdims[ndims - 1].os = vl;
  for(int d = ndims - 2; d >= 0; --d) {
    outerdims[d].n = plan->nprocs[d];
    outerdims[d].is = plan->nprocs[d+1]*outerdims[d+1].is;
    outerdims[d].os = plan->nprocs[d+1]*outerdims[d+1].os;
  }
  plan->outerplan_fwd = fftw_plan_guru_dft(ndims, outerdims, 2, howmany,
      plan->srcbuffer, plan->dstbuffer, FFTW_FORWARD, FFTW_ESTIMATE);
  if(plan->outerplan_fwd == NULL) goto fail_free_outerdims;
  plan->outerplan_rev = fftw_plan_guru_dft(ndims, outerdims, 2, howmany,
      plan->dstbuffer, plan->srcbuffer, FFTW_BACKWARD, FFTW_ESTIMATE);
  if(plan->outerplan_rev == NULL) goto fail_free_outerplan_fwd;

  free(outerdims);

  /* Initialize persistent communication requests */
  err = fft_par_init_swizzle_reqs(plan);
  if(err != MPI_SUCCESS) goto fail_free_outerplan_rev;

  if(errloc != NULL) *errloc = err;
  return plan;

  for(int i = 0; i < nprocs; ++i) {
    if(err == MPI_SUCCESS) {
      err = MPI_Request_free(plan->swizzle_sendreqs_fwd + i);
    }
    if(err == MPI_SUCCESS) {
      err = MPI_Request_free(plan->swizzle_recvreqs_fwd + i);
    }
    if(err == MPI_SUCCESS) {
      err = MPI_Request_free(plan->swizzle_sendreqs_rev + i);
    }
    if(err == MPI_SUCCESS) {
      err = MPI_Request_free(plan->swizzle_recvreqs_rev + i);
    }
  }
fail_free_outerplan_rev:
  fftw_destroy_plan(plan->outerplan_rev);
fail_free_outerplan_fwd:
  fftw_destroy_plan(plan->outerplan_fwd);
fail_free_outerdims:
  free(outerdims);
fail_free_innerplan_rev:
  fftw_destroy_plan(plan->innerplan_rev);
fail_free_innerplan_fwd:
  fftw_destroy_plan(plan->innerplan_fwd);
fail_free_swizzle_params:
  for(int i = 0; i < nprocs; ++i) {
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->swizzle_sendtypes + i);
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->swizzle_recvtypes + i);
  }
  free(plan->swizzle_sendtypes);
  free(plan->swizzle_recvtypes);
  free(plan->swizzle_senddispls);
  free(plan->swizzle_recvdispls);
fail_free_cycle_params:
  for(int i = 0; i < nprocs; ++i) {
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->cycle_sendtypes + i);
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->cycle_recvtypes + i);
  }
  free(plan->cycle_sendtypes);
  free(plan->cycle_recvtypes);
  free(plan->cycle_senddispls);
  free(plan->cycle_recvdispls);
fail_free_swizzle_recvreqs_rev:
  free(plan->swizzle_recvreqs_rev);
fail_free_swizzle_sendreqs_rev:
  free(plan->swizzle_sendreqs_rev);
fail_free_swizzle_recvreqs_fwd:
  free(plan->swizzle_recvreqs_fwd);
fail_free_swizzle_sendreqs_fwd:
  free(plan->swizzle_sendreqs_fwd);
fail_free_ones:
  free(plan->ones);
fail_free_dstbuffer:
  fftw_free(plan->dstbuffer);
fail_free_srcbuffer:
  fftw_free(plan->srcbuffer);
fail_free_nprocs:
  free(plan->nprocs);
fail_free_nelems:
  free(plan->nelems);
fail_free_map:
  free(plan->map);
fail_free_outerjobs:
  free(plan->outerjobs);
fail_free_plan:
  free(plan);
fail_immed:
  if(errloc != NULL) *errloc = err;
  return NULL;
}

int fft_par_plan_destroy(fft_par_plan plan)
{
  int err = MPI_SUCCESS;

  int procs = 1, ojobs = 1;
  for(int d = 0; d < plan->ndims; ++d) {
    procs *= plan->nprocs[d];
    ojobs *= plan->outerjobs[plan->procid*plan->ndims + d];
  }

  /* Free up MPI resources */
  for(int i = 0; i < procs; ++i) {
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->cycle_sendtypes + i);
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->cycle_recvtypes + i);
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->swizzle_sendtypes + i);
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->swizzle_recvtypes + i);
    if(err == MPI_SUCCESS) {
      err = MPI_Request_free(plan->swizzle_sendreqs_fwd + i);
    }
    if(err == MPI_SUCCESS) {
      err = MPI_Request_free(plan->swizzle_recvreqs_fwd + i);
    }
    if(err == MPI_SUCCESS) {
      err = MPI_Request_free(plan->swizzle_sendreqs_rev + i);
    }
    if(err == MPI_SUCCESS) {
      err = MPI_Request_free(plan->swizzle_recvreqs_rev + i);
    }
  }

  /* Free up memory buffers */
  fftw_destroy_plan(plan->outerplan_rev);
  fftw_destroy_plan(plan->innerplan_rev);
  fftw_destroy_plan(plan->outerplan_fwd);
  fftw_destroy_plan(plan->innerplan_fwd);
  free(plan->swizzle_recvdispls);
  free(plan->swizzle_recvtypes);
  free(plan->swizzle_senddispls);
  free(plan->swizzle_sendtypes);
  free(plan->cycle_recvdispls);
  free(plan->cycle_recvtypes);
  free(plan->cycle_senddispls);
  free(plan->cycle_sendtypes);
  free(plan->swizzle_recvreqs_rev);
  free(plan->swizzle_sendreqs_rev);
  free(plan->swizzle_recvreqs_fwd);
  free(plan->swizzle_sendreqs_fwd);
  free(plan->ones);
  fftw_free(plan->dstbuffer);
  fftw_free(plan->srcbuffer);
  free(plan->nprocs);
  free(plan->nelems);
  free(plan->map);
  free(plan->outerjobs);
  free(plan);

  return err;
}

int fft_par_init_swizzle_reqs(fft_par_plan plan)
{
  int err = MPI_SUCCESS;
  int procs = 1;
  for(int d = 0; d < plan->ndims; ++d) procs *= plan->nprocs[d];

  /* Queue up recvs */
  int rc = -1; /* Index of receive request successfully committed */
  for(int p = 0; p < procs; ++p) {
    err = MPI_Recv_init(
        (char *)plan->srcbuffer + plan->swizzle_recvdispls[p],
        1, plan->swizzle_recvtypes[p], p, 0, plan->comm,
        plan->swizzle_recvreqs_fwd + p);
    if(err != MPI_SUCCESS) goto fail_cancel_recvreqs;
    rc = p;
  }

  /* Queue up sends */
  int sc = -1; /* Index of last send request successfully completed. */
  for(int p = 0; p < procs; ++p) {
    err = MPI_Send_init(
        (char *)plan->dstbuffer + plan->swizzle_senddispls[p],
        1, plan->swizzle_sendtypes[p], p, 0, plan->comm,
        plan->swizzle_sendreqs_fwd + p);
    if(err != MPI_SUCCESS) goto fail_cancel_sendreqs;
    sc = p;
  }

  /* Queue up reverse recvs */
  int rrc = -1; /* Index of receive request successfully committed */
  for(int p = 0; p < procs; ++p) {
    err = MPI_Recv_init(
        (char *)plan->dstbuffer + plan->swizzle_senddispls[p],
        1, plan->swizzle_sendtypes[p], p, 0, plan->comm,
        plan->swizzle_recvreqs_rev + p);
    if(err != MPI_SUCCESS) goto fail_cancel_revrecvreqs;
    rrc = p;
  }

  /* Queue up reverse sends */
  int rsc = -1; /* Index of last send request successfully completed. */
  for(int p = 0; p < procs; ++p) {
    err = MPI_Send_init(
        (char *)plan->srcbuffer + plan->swizzle_recvdispls[p],
        1, plan->swizzle_recvtypes[p], p, 0, plan->comm,
        plan->swizzle_sendreqs_rev + p);
    if(err != MPI_SUCCESS) goto fail_cancel_revsendreqs;
    rsc = p;
  }

  /* Success */
  return err;

  /* Error handling -- this currently does nothing because if we get here, err
   * is not MPI_SUCCESS. It's here as a stub for better error handling.
   */
fail_cancel_revsendreqs:
  for(int p = rsc; p >= 0; --p) {
    if(err == MPI_SUCCESS) {
      err = MPI_Request_free(plan->swizzle_sendreqs_rev + p);
    }
  }
fail_cancel_revrecvreqs:
  for(int p = rrc; p >= 0; --p) {
    if(err == MPI_SUCCESS) {
      err = MPI_Request_free(plan->swizzle_recvreqs_rev + p);
    }
  }
fail_cancel_sendreqs:
  for(int p = sc; p >= 0; --p) {
    if(err == MPI_SUCCESS) {
      err = MPI_Request_free(plan->swizzle_sendreqs_fwd + p);
    }
  }
fail_cancel_recvreqs:
  for(int p = rc; p >= 0; --p) {
    if(err == MPI_SUCCESS) {
      err = MPI_Request_free(plan->swizzle_recvreqs_fwd + p);
    }
  }
  return err;
}

int fft_par_execute_fwd(fft_par_plan plan)
{
  int err = MPI_SUCCESS;
  int procs = 1; for(int d = 0; d < plan->ndims; ++d) procs *= plan->nprocs[d];

  // Stage one: cyclic permutation.
  err = MPI_Alltoallw(plan->src, plan->ones, plan->cycle_senddispls,
      plan->cycle_sendtypes, plan->srcbuffer, plan->ones, plan->cycle_recvdispls,
      plan->cycle_recvtypes, plan->comm);
  if (err != MPI_SUCCESS) goto fail_immed;
  // Stage two: inner fft
  fftw_execute(plan->innerplan_fwd);
  // Stage 2.5: finish the fourier transform
  fft_r2c_finish_unpacked_vec(plan->dstbuffer, plan->ndims, plan->nelems,
      plan->veclen);
  // Stage three: swizzle data
  /* Actually, we just start the swizzle communication. In stage four, we'll
   * wait for a receive to complete and then apply twiddle factors for that
   * processor. Before stage 5, we ensure all sends complete.
   */
  err = MPI_Startall(procs, plan->swizzle_recvreqs_fwd);
  if(err != MPI_SUCCESS) goto fail_immed;
  err = MPI_Startall(procs, plan->swizzle_sendreqs_fwd);
  if(err != MPI_SUCCESS) goto fail_immed;
  //Stage four: twiddle factors
  for(int procs_remain = procs; procs_remain > 0; --procs_remain) {
    int idx = MPI_UNDEFINED;
    err = MPI_Waitany(procs, plan->swizzle_recvreqs_fwd, &idx,
        MPI_STATUSES_IGNORE);
    if(err != MPI_SUCCESS) goto fail_immed;
    err = apply_twiddle_factors(plan, idx, TWIDDLE_FORWARD);
    if(err != MPI_SUCCESS) goto fail_immed;
  }
  err = MPI_Waitall(procs, plan->swizzle_sendreqs_fwd, MPI_STATUSES_IGNORE);
  if(err != MPI_SUCCESS) goto fail_immed;
  // Stage five: outer fft
  fftw_execute(plan->outerplan_fwd);
  // Stage six: rearrange data back to destination
  // This is the reverse of the previous swizzling step
  err = MPI_Alltoallw(plan->dstbuffer, plan->ones, plan->swizzle_recvdispls,
      plan->swizzle_recvtypes, plan->dst, plan->ones, plan->swizzle_senddispls,
      plan->swizzle_sendtypes, plan->comm);
fail_immed:
  return err;
}

int fft_par_execute_rev(fft_par_plan plan)
{
  int err = MPI_SUCCESS;
  int procs = 1; for(int d = 0; d < plan->ndims; ++d) procs *= plan->nprocs[d];

  /* We need to do the forward transform, but in reverse */

  /* Stage Six: bring the data from the destination array back to the dstbuffer
   */
  err = MPI_Alltoallw(plan->dst, plan->ones, plan->swizzle_senddispls,
      plan->swizzle_sendtypes, plan->dstbuffer, plan->ones,
      plan->swizzle_recvdispls, plan->swizzle_recvtypes, plan->comm);
  if(err != MPI_SUCCESS) goto fail_immed;
  /* Stage five: outer fft */
  fftw_execute(plan->outerplan_rev);
  /* Stage four: twiddle factors
   * Actually, we kick off the second stage sends as we finish the computations
   * in order to hide the computational cost of twiddling in the latency of the
   * communication
   */
  MPI_Startall(procs, plan->swizzle_recvreqs_rev);
  for(int p = 0; p < procs; ++p) {
    err = apply_twiddle_factors(plan, p, TWIDDLE_REVERSE);
    if(err != MPI_SUCCESS) goto fail_immed;
    err = MPI_Start(plan->swizzle_sendreqs_rev + p);
    if(err != MPI_SUCCESS) goto fail_immed;
  }
  /* Stage three: swizzle communication
   * This was already kicked off in the last stage, now wait for it to complete.
   * We need to wait for the sends to finish as well before we clobber the dst
   * buffer they're sending from with the inner fft.
   */
  err = MPI_Waitall(procs, plan->swizzle_recvreqs_rev, MPI_STATUSES_IGNORE);
  if(err != MPI_SUCCESS) goto fail_immed;
  err = MPI_Waitall(procs, plan->swizzle_sendreqs_rev, MPI_STATUSES_IGNORE);
  if(err != MPI_SUCCESS) goto fail_immed;
  /* Stage two: inner fft
   * No finishing of the incomplete data is necessary because the data comes
   * from a complete complex plan.
   */
  fftw_execute(plan->innerplan_rev);
  /* Stage one, cycling permutation */
  err = MPI_Alltoallw(plan->srcbuffer, plan->ones, plan->cycle_recvdispls,
      plan->cycle_recvtypes, plan->src, plan->ones, plan->cycle_senddispls,
      plan->cycle_sendtypes, plan->comm);

fail_immed:
  return err;
}

int cycle_data_parameter_fill(const fft_par_plan plan)
{
  /* We only read these fields of plan */
  const int *const elems = plan->nelems;
  const int *const procs = plan->nprocs;
  const int dims = plan->ndims;
  const MPI_Comm comm = plan->comm;
  const int *const map = plan->map;

  MPI_Aint lb, ext;
  int err = MPI_Type_get_extent(MPI_DOUBLE, &lb, &ext);
  if(err != MPI_SUCCESS) goto fail_immed;

  int nprocs;
  err = MPI_Comm_size(comm, &nprocs);
  if(err != MPI_SUCCESS) goto fail_immed;

  int procid;
  err = MPI_Comm_rank(comm, &procid);
  if(err != MPI_SUCCESS) goto fail_immed;

  /* Allocate buffers for the types and offsets to use when sending */
  plan->cycle_sendtypes = malloc(nprocs*sizeof(MPI_Datatype));
  if(plan->cycle_sendtypes == NULL) goto fail_immed;
  plan->cycle_senddispls = malloc(nprocs*sizeof(int));
  if(plan->cycle_senddispls == NULL) goto fail_free_cycle_sendtypes;
  /* Allocate buffers for the types and offsets to use when receiving */
  plan->cycle_recvtypes = malloc(nprocs*sizeof(MPI_Datatype));
  if(plan->cycle_recvtypes == NULL) goto fail_free_cycle_senddispls;
  plan->cycle_recvdispls = malloc(nprocs*sizeof(int));
  if(plan->cycle_recvdispls == NULL) goto fail_free_cycle_recvtypes;

  const int *const loc = map + procid*dims;

  // Compute offsets to use when sending
  for(int r = 0; r < nprocs; ++r) {
    const int *const rloc = map + r*dims;
    int displ = 0;
    // Apply the same offset formula in conjunction with the row-major access
    // formula to find the offset.
    for(int d = 0; d < dims; ++d) {
      displ = elems[d]*displ +
        ((rloc[d] - loc[d]*elems[d]) % procs[d] + procs[d]) % procs[d];
    }
    plan->cycle_senddispls[r] = displ*ext*plan->veclen;
  }
  // Compute offsets to use when receiving
  for(int s = 0; s < nprocs; ++s) {
     int displ = 0;
     for(int d = 0; d < dims; ++d) {
       const int *const sloc = map + s*dims;
       displ *= elems[d];
       for(int ss = 0; ss < sloc[d]; ++ss) {
         displ += elems[d]/procs[d];
         if(((loc[d] - ss*elems[d])%procs[d] + procs[d])%procs[d] <
             elems[d] % procs[d]) {
           ++displ;
         }
       }
     }
     plan->cycle_recvdispls[s] = displ*ext*plan->veclen;
  }
  int rc = -1; /* The last sendtype index with a type that needs to be freed */
  for(int r = 0; r < nprocs; ++r) {
    const int *const rloc = map + r*dims;
    /* initialize type */
    err = MPI_Type_vector(1, plan->veclen, plan->veclen, MPI_DOUBLE,
        plan->cycle_sendtypes + r);
    if(err != MPI_SUCCESS) goto fail_free_sendtypes;
    rc = r;
    int stride = ext*plan->veclen;
    /* Work downard along the dimensions, thus handling the fast indices first */
    for(int d = dims - 1; d >= 0; --d) {
      // Compute how many indices in this direction
      int count = elems[d]/procs[d];
      if(((rloc[d] - loc[d]*elems[d])%procs[d] + procs[d])%procs[d] <
          elems[d] % procs[d]) {
        ++count;
      }
      MPI_Datatype t,s;
      err = MPI_Type_hvector(count, 1, procs[d]*stride, plan->cycle_sendtypes[r],
          &t);
      if(err != MPI_SUCCESS) goto fail_free_sendtypes;
      // Swap the new and old data types. This way, cycle_sendtypes[r] contains
      // a valid type which needs to be deleted if a calamity occurs. Thus, we
      // can skip out of the loop and free cycle_sendtypes[r] if we hit a
      // problem.
      s = t; t = plan->cycle_sendtypes[r]; plan->cycle_sendtypes[r] = s;
      // Delete the old data type
      err = MPI_Type_free(&t);
      if(err != MPI_SUCCESS) goto fail_free_sendtypes;
      stride *= elems[d];
    }
    err = MPI_Type_commit(plan->cycle_sendtypes + r);
    if(err != MPI_SUCCESS) goto fail_free_sendtypes;
  }

  int sc = -1;
  for(int s = 0; s < nprocs; ++s) {
    const int *const sloc = map + s*dims;
    /* initialize receive type */
    err = MPI_Type_vector(1, plan->veclen, plan->veclen, MPI_DOUBLE,\
        plan->cycle_recvtypes + s);
    if(err != MPI_SUCCESS) goto fail_free_recvtypes;
    sc = s;
    int stride = ext*plan->veclen;
    /* Work down through indices, from fastest to slowest */
    for(int d = dims - 1; d >= 0; --d) {
      int count = elems[d]/procs[d];
      if(((loc[d] - sloc[d]*elems[d])%procs[d] + procs[d])%procs[d] <
          elems[d] % procs[d]) {
        ++count;
      }
      MPI_Datatype t, q;
      err = MPI_Type_hvector(count, 1, stride, plan->cycle_recvtypes[s], &t);
      if(err != MPI_SUCCESS) goto fail_free_recvtypes;
      /* As before, if t is a valid type, we want to swap it into recvtypes
       * before freeing the old type.
       */
      q = t; t = plan->cycle_recvtypes[s]; plan->cycle_recvtypes[s] = q;
      /* Delete the old data type */
      err = MPI_Type_free(&t);
      if(err != MPI_SUCCESS) goto fail_free_recvtypes;
      stride *= elems[d];
    }
    err = MPI_Type_commit(plan->cycle_recvtypes + s);
    if(err != MPI_SUCCESS) goto fail_free_recvtypes;
  }

  return err;

fail_free_recvtypes:
  while(sc >= 0) {
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->cycle_sendtypes + rc);
    sc--;
  }
fail_free_sendtypes:
  while(rc >= 0) {
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->cycle_sendtypes + rc);
    rc--;
  }
  free(plan->cycle_recvdispls);
fail_free_cycle_recvtypes:
  free(plan->cycle_recvtypes);
fail_free_cycle_senddispls:
  free(plan->cycle_senddispls);
fail_free_cycle_sendtypes:
  free(plan->cycle_sendtypes);
fail_immed:
  return err;
}

int swizzle_data_parameter_fill(const fft_par_plan plan)
{
  /* We only read these members of plan */
  MPI_Comm comm = plan->comm;
  const int dims = plan->ndims;
  const int *const map = plan->map;
  const int *const elems = plan->nelems;
  const int *const procs = plan->nprocs;
  const int *const ojobs = plan->outerjobs;

  MPI_Aint lb, ext;
  int err = MPI_Type_get_extent(MPI_C_DOUBLE_COMPLEX, &lb, &ext);
  if(err != MPI_SUCCESS) goto fail_immed;

  int nprocs;
  err = MPI_Comm_size(comm, &nprocs);
  if(err != MPI_SUCCESS) goto fail_immed;

  int procid;
  err = MPI_Comm_rank(comm, &procid);
  if(err != MPI_SUCCESS) goto fail_immed;

  /* Allocate buffers for the types and offsets to use when sending */
  plan->swizzle_sendtypes = malloc(nprocs*sizeof(MPI_Datatype));
  if(plan->swizzle_sendtypes == NULL) goto fail_immed;
  plan->swizzle_senddispls = malloc(nprocs*sizeof(int));
  if(plan->swizzle_senddispls == NULL) goto fail_free_swizzle_sendtypes;
  /* Allocate buffers for the types and offsets to use when receiving */
  plan->swizzle_recvtypes = malloc(nprocs*sizeof(MPI_Datatype));
  if(plan->swizzle_recvtypes == NULL) goto fail_free_swizzle_senddispls;
  plan->swizzle_recvdispls = malloc(nprocs*sizeof(int));
  if(plan->swizzle_recvdispls == NULL) goto fail_free_swizzle_recvtypes;

  // Compute offsets to use when sending
  for(int r = 0; r < nprocs; ++r) {
    const int *const rloc = map + r*dims;
    int displ = 0;
    // Apply the same offset formula in conjunction with the row-major access
    // formula to find the offset.
    for(int d = 0; d < dims; ++d) {
      displ = elems[d]*displ + rloc[d];
    }
    plan->swizzle_senddispls[r] = displ*ext*plan->veclen;
  }
  // Compute offsets to use when receiving
  for(int s = 0; s < nprocs; ++s) {
    const int *const sloc = map + s*dims;
    int displ = 0;
    // Apply the same offset formula in conjunction with the row-major access
    // formula to find the offset.
    for(int d = 0; d < dims; ++d) {
      displ = procs[d]*displ + sloc[d];
    }
    plan->swizzle_recvdispls[s] = displ*ext*plan->veclen;
  }
  int rc = -1; /* The last sendtype index with a type that needs to be freed */
  for(int r = 0; r < nprocs; ++r) {
    /* initialize type */
    err = MPI_Type_vector(1, plan->veclen, plan->veclen, MPI_C_DOUBLE_COMPLEX,
        plan->swizzle_sendtypes + r);
    if(err != MPI_SUCCESS) goto fail_free_sendtypes;
    rc = r;
    int stride = ext*plan->veclen;
    /* Work downard along the dimensions, thus handling the fast indices first */
    for(int d = dims - 1; d >= 0; --d) {
      // Compute how many indices in this direction
      int count = ojobs[r*dims + d];
      MPI_Datatype t,s;
      err = MPI_Type_hvector(count, 1, procs[d]*stride, plan->swizzle_sendtypes[r],
          &t);
      if(err != MPI_SUCCESS) goto fail_free_sendtypes;
      // Swap the new and old data types. This way, swizzle_sendtypes[r] contains
      // a valid type which needs to be deleted if a calamity occurs. Thus, we
      // can skip out of the loop and free swizzle_sendtypes[r] if we hit a
      // problem.
      s = t; t = plan->swizzle_sendtypes[r]; plan->swizzle_sendtypes[r] = s;
      // Delete the old data type
      err = MPI_Type_free(&t);
      if(err != MPI_SUCCESS) goto fail_free_sendtypes;
      stride *= elems[d];
    }
    err = MPI_Type_commit(plan->swizzle_sendtypes + r);
    if(err != MPI_SUCCESS) goto fail_free_sendtypes;
  }
  int sc = -1; /* The last recvtype index with a type that needs to be freed */
  for(int s = 0; s < nprocs; ++s) {
    /* Compute the length of a work unit */
    int wulen = 1;
    for(int d = 0; d < dims; ++d) { wulen *= procs[d]; }

    /* Compute the number of work units */
    int nwu = 1;
    for(int d = 0; d < dims; ++d) { nwu *= ojobs[procid*dims + d]; }

    /* Initialize type */
    err = MPI_Type_vector(nwu, plan->veclen, wulen*plan->veclen,
        MPI_C_DOUBLE_COMPLEX, plan->swizzle_recvtypes + s);
    if(err != MPI_SUCCESS) goto fail_free_recvtypes;
    sc = s;

    err = MPI_Type_commit(plan->swizzle_recvtypes + s);
    if(err != MPI_SUCCESS) goto fail_free_sendtypes;
  }

  return err;

fail_free_recvtypes:
  while(sc >= 0) {
    if(err != MPI_SUCCESS) err = MPI_Type_free(plan->swizzle_recvtypes + sc);
    --sc;
  }
fail_free_sendtypes:
  while(rc >= 0) {
    if(err != MPI_SUCCESS) err = MPI_Type_free(plan->swizzle_sendtypes + rc);
    --rc;
  }
  free(plan->swizzle_recvdispls);
fail_free_swizzle_recvtypes:
  free(plan->swizzle_recvtypes);
fail_free_swizzle_senddispls:
  free(plan->swizzle_senddispls);
fail_free_swizzle_sendtypes:
  free(plan->swizzle_sendtypes);
fail_immed:
  return err;
}
