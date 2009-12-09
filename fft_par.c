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
  /* Communication Types */
  MPI_Datatype *cycle_sendtypes;
  MPI_Datatype *cycle_recvtypes;
  /* FFT Plans */
  fftw_plan innerplan;
  int *ones;
  int *map; /* [number of procs][ndims] location of each processor */
  int *outerjobs; /* [number of procs] Number of outer jobs on each processor */
  int *nelems; /* [ndims] Number of elements in each direction */
  int *nprocs; /* [ndims] Number of processors in each direction */
  int ndims; /* How many dimensions there are */
  int len; /* the vector length of the data */
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
 *  \param plan The plan owning the outersrc array to apply twiddle factors to.
 *  \return the MPI error code, if any
 */
int apply_twiddle_factors(fft_par_plan plan)
{
  /* The work units are arranged across the processors cyclically:
   * 0: 0, p, 2p, ...
   * 1: 1, p+1, 2p+1, ..
   * 2: 2, p+2, 2p+2, ..
   * ...
   * p-1: p-1, p + p - 1, 2p + p - 1...
   *
   * Each work unit needs to be multiplied by a twiddle factor which is exactly
   * the root of unity of the number of elements in the entire fourier
   * transform to the power of the product of the work unit, to the power of
   * its index within the work unit.
   */
  int rank;
  int err = MPI_Comm_rank(plan->comm, &rank);
  if(err != MPI_SUCCESS) goto fail_immed;

  // int nprocs;
  // err = MPI_Comm_size(plan->comm, &nprocs);
  // if(err != MPI_SUCCESS) goto fail_immed;

  // const double complex w0 = cexp(-2*M_PI*I/(nprocs*plan->nelems));
  // for(int wu = 0; wu < plan->outerjobs[rank]; ++wu) {
  //   for(int i = 0; i < nprocs; ++i) {
  //     ((double complex *)plan->outersrc)[wu*nprocs + i] *=
  //       cpow(w0, i*(wu*nprocs + rank));
  //   }
  // }
fail_immed:
  return err;
}

fft_par_plan fft_par_plan_r2c_nc(MPI_Comm comm, int ndims, const int *proc_dim,
    const int *map, const int *size, int len, double *src, complex double *dst,
    int *errloc);

fft_par_plan fft_par_plan_r2c_1d(MPI_Comm comm, int const nelems,
    double * const src, double complex * const dst, int * const errloc)
{
  int size;
  int err = MPI_Comm_size(comm, &size);
  if(err != MPI_SUCCESS) goto fail_immed;

  /* Create a trivial map for the one dimensional case */
  int *const map = malloc(size*sizeof(int));
  for(int i = 0; i <= size; ++i) map[i] = i;

  fft_par_plan plan = fft_par_plan_r2c_nc(comm, 1, &size, map, &nelems, 1,
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
  fft_par_plan plan = fft_par_plan_r2c_nc(comm, ndims, proc_dim, map, size,
      len, src, dst, &err);

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
    const int len, double * const src, double complex * const dst, int *errloc)
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
  plan->len = len;

  /* The total number of data elements on each processor */
  int netlen = len; for(int i = 0; i < ndims; ++i) netlen *= size[i];

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
  plan->outersrc = fftw_malloc(ojobs*len*nprocs*sizeof(double complex));
  if(plan->outersrc == NULL) goto fail_free_innerdst;
  plan->outerdst = fftw_malloc(ojobs*len*nprocs*sizeof(double complex));
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

  plan->innerplan = fftw_plan_dft_r2c(ndims, size, plan->innersrc,
      plan->innerdst, FFTW_ESTIMATE);
  if(plan->innerplan == NULL) goto fail_free_swizzle_params;

  if(errloc != NULL) *errloc = err;
  return plan;

fail_free_swizzle_params:
//fail_free_cycle_params:
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

  int procs = 1; for(int d = 0; d < plan->ndims; ++d) procs *= plan->nprocs[d];

  /* Free up MPI resources */
  for(int i = 0; i < procs; ++i) {
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->cycle_sendtypes + i);
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->cycle_recvtypes + i);
  }

  /* Free up memory buffers */
  fftw_destroy_plan(plan->innerplan);
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
  // FIXME: As a debug aid, copy plan->innerdst directly to dst
  int netelems = plan->len;
  for(int d = 0; d < plan->ndims; ++d) netelems *= plan->nelems[d];
  for(int i = 0; i < netelems; ++i) {
    *((double complex *)plan->dst + i) = *((double complex *)plan->innerdst + i);
  }
  // // Stage three: swizzle data
  // err = MPI_Alltoallw(plan->innerdst, plan->ones, plan->swiz_sd, plan->swiz_st,
  //     plan->outersrc, plan->ones, plan->swiz_rd, plan->swiz_rt, plan->comm);
  // if(err != MPI_SUCCESS) goto fail_immed;
  // //Stage four: twiddle factors
  // err = apply_twiddle_factors(plan);
  // if(err != MPI_SUCCESS) goto fail_immed;
  // // Stage five: outer fft
  // int rank;
  // err = MPI_Comm_rank(plan->comm, &rank);
  // if(err != MPI_SUCCESS) goto fail_immed;
  // int nprocs;
  // err = MPI_Comm_size(plan->comm, &nprocs);
  // if(err != MPI_SUCCESS) goto fail_immed;
  // for(int w = 0; w < plan->outerjobs[rank]; ++w) {
  //   /* Execute the fft for each work unit. The use of fftw_execute_dft is
  //    * fraught with peril, because we have to ensure we have the proper
  //    * alignment for each block. The outersrc and outerdst buffers are
  //    * allocated to 8 or 16 byte alignment, depending on the SIMD requirements
  //    * of the platform.  The size of a double complex is 16 bytes, so each work
  //    * unit is thus also 8 or 16 byte aligned.
  //    */
  //   fftw_execute_dft(plan->outerplan,
  //       (double complex *)plan->outersrc + w*nprocs,
  //       (double complex *)plan->outerdst + w*nprocs);
  // }
  // // Stage six: rearrange data back to destination
  // // This is the reverse of the previous swizzling step
  // err = MPI_Alltoallw(plan->outerdst, plan->ones, plan->swiz_rd, plan->swiz_rt,
  //     plan->dst, plan->ones, plan->swiz_sd, plan->swiz_st, plan->comm);
  // TODO: Compute the FFT rather than just zeroing the destination */
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
  const int len = plan->len;
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
    plan->cycle_senddispls[r] = displ*ext*len;
  }
  // TODO: Delete this print statement
  for(int i = 0; i < nprocs; ++i) {
    if(i == procid) {
      for(int r = 0; r < nprocs; ++r) {
        printf("cyc_senddispls (%d", map[procid*dims]);
        for(int d = 1; d < dims; ++d) printf(",%d", map[procid*dims + d]);
        printf(")->(%d", map[r*dims]);
        for(int d = 1; d < dims; ++d) printf(",%d", map[r*dims + d]);
        printf("): %d\n", plan->cycle_senddispls[r]);
      }
      fflush(stdout);
    }
    MPI_Barrier(comm);
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
     plan->cycle_recvdispls[s] = displ*ext*len;
  }
  // TODO: Delete this print statement
  for(int i = 0; i < nprocs; ++i) {
     if(i == procid) {
       for(int s = 0; s < nprocs; ++s) {
         printf("cyc_recvdispls (%d", map[procid*dims]);
         for(int d = 1; d < dims; ++d) printf(",%d", map[procid*dims + d]);
         printf(")<-(%d", map[s*dims]);
         for(int d = 1; d < dims; ++d) printf(",%d", map[s*dims + d]);
         printf("): %d\n", plan->cycle_recvdispls[s]);
       }
       fflush(stdout);
     }
     MPI_Barrier(comm);
  }
  int rc = -1; /* The last sendtype index with a type that needs to be freed */
  for(int i = 0; i < nprocs; ++i) {
  if(i == procid) {
  for(int r = 0; r < nprocs; ++r) {
    const int *const rloc = map + r*dims;
    printf("cyc_sendcount: (%d", map[procid*dims]);
    for(int d = 1; d < dims; ++d) printf(",%d", map[procid*dims + d]);
    printf(")->(%d", map[r*dims]);
    for(int d = 1; d < dims; ++d) printf(",%d", map[r*dims + d]);
    printf("):");
    /* initialize type */
    err = MPI_Type_contiguous(len, MPI_DOUBLE, plan->cycle_sendtypes + r);
    if(err != MPI_SUCCESS) goto fail_free_sendtypes;
    rc = r;
    int stride = len*ext;
    /* Work downard along the dimensions, thus handling the fast indices first */
    for(int d = dims - 1; d >= 0; --d) {
      // Compute how many indices in this direction
      int count = elems[d]/procs[d];
      if(((rloc[d] - loc[d]*elems[d])%procs[d] + procs[d])%procs[d] <
          elems[d] % procs[d]) {
        ++count;
      }
      printf(" %d", count);
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
    printf("\n");
    err = MPI_Type_commit(plan->cycle_sendtypes + r);
    if(err != MPI_SUCCESS) goto fail_free_sendtypes;
  }
  fflush(stdout);
  }
  MPI_Barrier(comm);
  }

  int sc = -1;
  for(int i = 0; i < nprocs; ++i) {
  if(i == procid) {
  for(int s = 0; s < nprocs; ++s) {
    const int *const sloc = map + s*dims;
    printf("cyc_recvcount: (%d", map[procid*dims]);
    for(int d = 1; d < dims; ++d) printf(",%d", map[procid*dims + d]);
    printf(")<-(%d", map[s*dims]);
    for(int d = 1; d < dims; ++d) printf(",%d", map[s*dims + d]);
    printf("):");
    /* initialize receive type */
    err = MPI_Type_contiguous(len, MPI_DOUBLE, plan->cycle_recvtypes + s);
    if(err != MPI_SUCCESS) goto fail_free_recvtypes;
    sc = s;
    int stride = len*ext;
    /* Work down through indices, from fastest to slowest */
    for(int d = dims - 1; d >= 0; --d) {
      int count = elems[d]/procs[d];
      if(((loc[d] - sloc[d]*elems[d])%procs[d] + procs[d])%procs[d] <
          elems[d] % procs[d]) {
        ++count;
      }
      printf(" %d", count);
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
    printf("\n");
  }
  fflush(stdout);
  }
  MPI_Barrier(comm);
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
  // /* We only read these members of plan */
  // MPI_Comm comm = plan->comm;
  // const int * const ojobs = plan->outerjobs;
  //
  MPI_Aint lb, ext;
  int err = MPI_Type_get_extent(MPI_C_DOUBLE_COMPLEX, &lb, &ext);
  if(err != MPI_SUCCESS) goto fail_immed;

  // int procs;
  // err = MPI_Comm_size(comm, &procs);
  // if(err != MPI_SUCCESS) goto fail_immed;

  // int rank;
  // err = MPI_Comm_rank(comm, &rank);
  // if(err != MPI_SUCCESS) goto fail_immed;

  // int r = 0;
  // while(r < procs) {
  //   plan->swiz_sd[r] = r*ext;

  //   err = MPI_Type_vector(ojobs[r], 1, procs, MPI_C_DOUBLE_COMPLEX,
  //       plan->swiz_st + r);
  //   if(err != MPI_SUCCESS) goto fail_free_st;
  //   err = MPI_Type_commit(plan->swiz_st + r);
  //   if(err != MPI_SUCCESS) goto fail_free_st;

  //   ++r;
  // }

  // int s = 0;
  // while(s < procs) {
  //   plan->swiz_rd[s] = s*ext;

  //   err = MPI_Type_vector(ojobs[rank], 1, procs, MPI_C_DOUBLE_COMPLEX,
  //       plan->swiz_rt + s);
  //   if(err != MPI_SUCCESS) goto fail_free_rt;
  //   err = MPI_Type_commit(plan->swiz_rt + s);
  //   if(err != MPI_SUCCESS) goto fail_free_st;

  //   ++s;
  // }

  // return err;
  // fail_free_rt:
  // while(s >= 0) {
  //   if(err == MPI_SUCCESS) MPI_Type_free(plan->swiz_rt + s);

  //   --s;
  // }
  // fail_free_st:
  // while(r >= 0) {
  //   if(err == MPI_SUCCESS) MPI_Type_free(plan->swiz_st + r);

  //   --r;
  // }
fail_immed:
  return err;
}
