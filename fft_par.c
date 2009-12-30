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
  void *innersrc;
  void *innerdst;
  void *outersrc;
  void *outerdst;
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
  /* FFT Plans */
  fftw_plan innerplan;
  fftw_plan outerplan;
  int *ones;
  int *map; /* [procs][ndims] location of each processor */
  int *outerjobs; /* [procs][ndims] Number of outer jobs on each processor */
  int *nelems; /* [ndims] Number of elements in each direction */
  int *nprocs; /* [ndims] Number of processors in each direction */
  int ndims; /* How many dimensions there are */
  int procid;
  MPI_Comm comm;
};

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
int apply_twiddle_factors(fft_par_plan plan, int p)
{
  int procid;
  int err = MPI_Comm_rank(plan->comm, &procid);
  if(err != MPI_SUCCESS) goto fail_immed;

  const int dims = plan->ndims;
  const int *const procs = plan->nprocs;
  const int *const elems = plan->nelems;
  const int *const ojobs = plan->outerjobs + procid*dims;
  const int *const loc = plan->map + procid*dims;

  /* Multi-index over the work unit */
  int *wu = alloca(dims*sizeof(int));

  /* Lenght of a workunit */
  int wulen = 1;
  for(int d = 0; d < dims; ++d) { wulen *= procs[d]; }

  /* Root of unity in each direction */
  double complex *w0 = alloca(dims*sizeof(complex double));
  for(int d = 0; d < dims; ++d) w0[d] = cexp(-2*M_PI*I/elems[d]/procs[d]);

  /* Zero the multi-index */
  for(int d = 0; d < dims; ++d) { wu[d] = 0; }

  /* The processor index */
  const int *const pidx = plan->map + p*dims;

  /* The displacement of this processor's element within a workunit */
  int displ = 0;
  for(int d = 0; d < dims; ++d) {
    displ = displ*procs[d] + pidx[d];
  }

  /* The current target element we are applying the swizzle factor to */
  double complex *tgt = (double complex *)plan->outersrc + displ;

  int cont = 1; /* Continue iteration over work units */
  for(int d = 0; d < dims; ++d) { cont = cont && wu[d] < ojobs[d]; }
  while(cont) {
    /* Apply twiddle factors */
    for(int d = 0; d < dims; ++d) {
      *tgt *= cpow(w0[d], pidx[d]*(wu[d]*procs[d] + loc[d]));
    }

    /* Increment the work unit's outermost indices, and spill backwards */
    ++wu[dims - 1];
    for(int d = dims - 1; d >= 1; --d) {
      if(wu[d] >= ojobs[d]) {
        wu[d] = 0;
        ++wu[d-1];
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
    const int *map, const int *size, double *src, complex double *dst,
    int *errloc);

fft_par_plan fft_par_plan_r2c_1d(MPI_Comm comm, int const nelems,
    double * const src, double complex * const dst, int * const errloc)
{
  fft_par_plan plan = NULL;
  int size;
  int err = MPI_Comm_size(comm, &size);
  if(err != MPI_SUCCESS) goto fail_immed;

  /* Create a trivial map for the one dimensional case */
  int *const map = malloc(size*sizeof(int));
  for(int i = 0; i < size; ++i) { map[i] = i; }

  plan = fft_par_plan_r2c_nc(comm, 1, &size, map, &nelems,
      src, dst, &err);

  free(map);
fail_immed:
  if(errloc != NULL) *errloc = err;
  return plan;
}

fft_par_plan fft_par_plan_r2c(MPI_Comm comm, const int *const size,
    const int len, double *const src, double complex *const dst,
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
  plan = fft_par_plan_r2c_nc(comm, ndims, proc_dim, map, size,
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
    const int *const proc_dim, const int *const map, const int *const size,
    double * const src, double complex * const dst, int *errloc)
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
  plan->innersrc = fftw_malloc(netlen*sizeof(double));
  if(plan->innersrc == NULL) goto fail_free_nprocs;
  plan->innerdst = fftw_malloc(netlen*sizeof(double complex));
  if(plan->innerdst == NULL) goto fail_free_innersrc;
  plan->outersrc = fftw_malloc(ojobs*nprocs*sizeof(double complex));
  if(plan->outersrc == NULL) goto fail_free_innerdst;
  plan->outerdst = fftw_malloc(ojobs*nprocs*sizeof(double complex));
  if(plan->outerdst == NULL) goto fail_free_outersrc;

  /* Allocate an array of ones to use in MPI_Alltoallw as the send count. The
   * types encode all the information.
   */
  plan->ones = malloc(nprocs*sizeof(int));
  if(plan->ones == NULL) goto fail_free_outerdst;
  for(int i = 0; i < nprocs; ++i) plan->ones[i] = 1;

  /* Initialize the cyclic communication types */
  err = cycle_data_parameter_fill(plan);
  if(err != MPI_SUCCESS) goto fail_free_ones;

  /* Initialize the swizzle communication types */
  err = swizzle_data_parameter_fill(plan);
  if(err != MPI_SUCCESS) goto fail_free_cycle_params;

  plan->innerplan = fftw_plan_dft_r2c(ndims, size, plan->innersrc,
      plan->innerdst, FFTW_ESTIMATE);
  if(plan->innerplan == NULL) goto fail_free_swizzle_params;

  /* Initialize the outer plan */
  plan->outerplan = fftw_plan_many_dft(ndims, plan->nprocs, ojobs,
      plan->outersrc, NULL, 1, nprocs, plan->outerdst, NULL, 1, nprocs,
      FFTW_FORWARD, FFTW_ESTIMATE);
  if(plan->outerplan == NULL) goto fail_free_innerplan;

  if(errloc != NULL) *errloc = err;
  return plan;

  fftw_destroy_plan(plan->outerplan);
fail_free_innerplan:
  free(plan->innerplan);
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
fail_free_ones:
  free(plan->ones);
fail_free_outerdst:
  fftw_free(plan->outerdst);
fail_free_outersrc:
  fftw_free(plan->outersrc);
fail_free_innerdst:
  fftw_free(plan->innerdst);
fail_free_innersrc:
  fftw_free(plan->innersrc);
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
  }

  /* Free up memory buffers */
  fftw_destroy_plan(plan->outerplan);
  fftw_destroy_plan(plan->innerplan);
  free(plan->swizzle_recvdispls);
  free(plan->swizzle_recvtypes);
  free(plan->swizzle_senddispls);
  free(plan->swizzle_sendtypes);
  free(plan->cycle_recvdispls);
  free(plan->cycle_recvtypes);
  free(plan->cycle_senddispls);
  free(plan->cycle_sendtypes);
  free(plan->ones);
  fftw_free(plan->outerdst);
  fftw_free(plan->outersrc);
  fftw_free(plan->innerdst);
  fftw_free(plan->innersrc);
  free(plan->nprocs);
  free(plan->nelems);
  free(plan->map);
  free(plan->outerjobs);
  free(plan);

  return err;
}

int fft_par_execute(fft_par_plan plan)
{
  int err = MPI_SUCCESS;
  // Stage one: cyclic permutation.
  err = MPI_Alltoallw(plan->src, plan->ones, plan->cycle_senddispls,
      plan->cycle_sendtypes, plan->innersrc, plan->ones, plan->cycle_recvdispls,
      plan->cycle_recvtypes, plan->comm);
  if (err != MPI_SUCCESS) goto fail_immed;
  // Stage two: inner fft
  fftw_execute(plan->innerplan);
  fft_r2c_finish(plan->innerdst, plan->ndims, plan->nelems);
  // Stage three: swizzle data
  err = MPI_Alltoallw(plan->innerdst, plan->ones, plan->swizzle_senddispls,
     plan->swizzle_sendtypes, plan->outersrc, plan->ones,
     plan->swizzle_recvdispls, plan->swizzle_recvtypes, plan->comm);
  if(err != MPI_SUCCESS) goto fail_immed;
  //Stage four: twiddle factors
  int numprocs = 1;
  for(int d = 0; d < plan->ndims; ++d) {
    numprocs *= plan->nprocs[d];
  }
  for(int p = 0; p < numprocs; ++p) {
    err = apply_twiddle_factors(plan,p);
  }
  if(err != MPI_SUCCESS) goto fail_immed;
  // Stage five: outer fft
  fftw_execute(plan->outerplan);
  // Stage six: rearrange data back to destination
  // This is the reverse of the previous swizzling step
  err = MPI_Alltoallw(plan->outerdst, plan->ones, plan->swizzle_recvdispls,
      plan->swizzle_recvtypes, plan->dst, plan->ones, plan->swizzle_senddispls,
      plan->swizzle_sendtypes, plan->comm);
fail_immed:
  return err;
}

int cycle_data_parameter_fill(const fft_par_plan plan)
{
  /* We only read these fields of plan */
  const int *const elems = plan->nelems;
  const int *const procs = plan->nprocs;
  const int dims = plan->ndims;
  const int comm = plan->comm;
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
    plan->cycle_senddispls[r] = displ*ext;
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
     plan->cycle_recvdispls[s] = displ*ext;
  }
  int rc = -1; /* The last sendtype index with a type that needs to be freed */
  for(int r = 0; r < nprocs; ++r) {
    const int *const rloc = map + r*dims;
    /* initialize type */
    err = MPI_Type_dup(MPI_DOUBLE, plan->cycle_sendtypes + r);
    if(err != MPI_SUCCESS) goto fail_free_sendtypes;
    rc = r;
    int stride = ext;
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
    err = MPI_Type_dup(MPI_DOUBLE, plan->cycle_recvtypes + s);
    if(err != MPI_SUCCESS) goto fail_free_recvtypes;
    sc = s;
    int stride = ext;
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
    plan->swizzle_senddispls[r] = displ*ext;
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
    plan->swizzle_recvdispls[s] = displ*ext;
  }
  int rc = -1; /* The last sendtype index with a type that needs to be freed */
  for(int r = 0; r < nprocs; ++r) {
    /* initialize type */
    err = MPI_Type_dup(MPI_C_DOUBLE_COMPLEX, plan->swizzle_sendtypes + r);
    if(err != MPI_SUCCESS) goto fail_free_sendtypes;
    rc = r;
    int stride = ext;
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
    err = MPI_Type_vector(nwu, 1, wulen, MPI_C_DOUBLE_COMPLEX,
        plan->swizzle_recvtypes + s);
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
