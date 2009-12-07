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
  int *outerjobs;
  int *cyc_sd;
  MPI_Datatype *cyc_st;
  int *cyc_rd;
  MPI_Datatype *cyc_rt;
  int *swiz_sd;
  MPI_Datatype *swiz_st;
  int *swiz_rd;
  MPI_Datatype *swiz_rt;
  int *ones; /* An nprocs-long array of ones */
  fftw_plan innerplan;
  fftw_plan outerplan;
  size_t nelems;
  int nprocs;
  int rank;
  MPI_Comm comm;
  MPI_Datatype srctype;
  MPI_Datatype srctype_pstrided;
};

//! Compute the send and receive offsets and types for a cyclic permutation.
/*! Compute the send and receive offsets and types for a given processor which
 *  will be used by <tt>MPI_Alltoallw</tt> to perform a cyclic data
 *  permutation.  The cyc_[sr][dt] members of the plan structure should be
 *  initialized to arrays of at least plan->nprocs length, and the nprocs,
 *  rank, and nelem members of the plan should be set to the values for the
 *  communication.
 *
 *  \param plan The plan to write the cyc_[sr][dt] members of.
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
 */
void apply_twiddle_factors(fft_par_plan plan)
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
  const double complex w0 = cexp(-2*M_PI*I/(plan->nprocs*plan->nelems));
  for(int wu = 0; wu < plan->outerjobs[plan->rank]; ++wu) {
    for(int i = 0; i < plan->nprocs; ++i) {
      ((double complex *)plan->outersrc)[wu*plan->nprocs + i] *=
        cpow(w0, i*(wu*plan->nprocs + plan->rank));
    }
  }
}

fft_par_plan fft_par_plan_r2c_1d(MPI_Comm comm, size_t const nelems,
    double * const src, double complex * const dst, int * const errloc)
{
  int err = MPI_SUCCESS;

  // Allocate the plan and initialize static values
  fft_par_plan plan = malloc(sizeof(struct fft_par_plan_s));
  if(plan == NULL) {
    goto fail_immed;
  }

  plan->nelems = nelems;
  plan->comm = comm;

  err = MPI_Comm_size(comm, &plan->nprocs);
  if(err != MPI_SUCCESS) goto fail_free_plan;

  err = MPI_Comm_rank(comm, &plan->rank);
  if(err != MPI_SUCCESS) goto fail_free_plan;

  plan->src = src;
  plan->dst = dst;

  /* How many jobs do we need to handle for the outer computation? The inner
   * computation is an nelem size fft carried out on each proc. The outer
   * computation is an nproc size fft carried out nelem times. We need to
   * spread the jobs out.
   */
  plan->outerjobs = malloc(sizeof(int)*plan->nprocs);
  if(plan->outerjobs == NULL) goto fail_free_plan;
  for(int i = 0; i < plan->nprocs; ++i) {
    plan->outerjobs[i] = nelems/plan->nprocs +
      (i < nelems % plan->nprocs ? 1 : 0);
  }

  plan->innersrc = fftw_malloc(sizeof(double)*nelems);
  if(plan->innersrc == NULL) goto fail_free_outerjobs;

  plan->innerdst = fftw_malloc(sizeof(double complex)*nelems);
  if(plan->innerdst == NULL) goto fail_free_innersrc;

  plan->outersrc = fftw_malloc(sizeof(double complex)*
      plan->outerjobs[plan->rank]*plan->nprocs);
  if(plan->outersrc == NULL) goto fail_free_innerdst;

  plan->outerdst = fftw_malloc(sizeof(double complex)*
      plan->outerjobs[plan->rank]*plan->nprocs);
  if(plan->outerdst == NULL) goto fail_free_outersrc;

  plan->innerplan = fftw_plan_dft_r2c_1d(nelems, plan->innersrc,
      plan->innerdst, FFTW_ESTIMATE);
  if(plan->innerplan == NULL) goto fail_free_outerdst;

  plan->outerplan = fftw_plan_dft_1d(plan->nprocs, plan->outersrc,
      plan->outerdst, FFTW_FORWARD, FFTW_ESTIMATE);
  if(plan->outerplan == NULL) goto fail_free_innerplan;

  plan->cyc_sd = malloc(sizeof(int)*plan->nprocs);
  if(plan->cyc_sd == NULL) goto fail_free_outerplan;

  plan->cyc_rd = malloc(sizeof(int)*plan->nprocs);
  if(plan->cyc_rd == NULL) goto fail_free_cyc_sd;

  plan->cyc_st = malloc(sizeof(MPI_Datatype)*plan->nprocs);
  if(plan->cyc_st == NULL) goto fail_free_cyc_rd;

  plan->cyc_rt = malloc(sizeof(MPI_Datatype)*plan->nprocs);
  if(plan->cyc_rt == NULL) goto fail_free_cyc_st;

  plan->ones = malloc(sizeof(int)*plan->nprocs);
  if(plan->ones == NULL) goto fail_free_cyc_rt;
  for(int i = 0; i < plan->nprocs; ++i) {
    plan->ones[i] = 1;
  }

  err = cycle_data_parameter_fill(plan);
  if(err != MPI_SUCCESS) goto fail_free_ones;

  plan->swiz_sd = malloc(sizeof(int)*plan->nprocs);
  if(plan->swiz_sd == NULL) goto fail_free_cyc_types;

  plan->swiz_rd = malloc(sizeof(int)*plan->nprocs);
  if(plan->swiz_rd == NULL) goto fail_free_swiz_sd;

  plan->swiz_st = malloc(sizeof(MPI_Datatype)*plan->nprocs);
  if(plan->swiz_st == NULL) goto fail_free_swiz_rd;

  plan->swiz_rt = malloc(sizeof(MPI_Datatype)*plan->nprocs);
  if(plan->swiz_rt == NULL) goto fail_free_swiz_st;

  err = swizzle_data_parameter_fill(plan);
  if(err != MPI_SUCCESS) goto fail_free_swiz_rt;

  if (errloc != NULL) *errloc = err;
  return plan;

  /* Error handling. */
fail_free_swiz_rt:
  free(plan->swiz_rt);
fail_free_swiz_st:
  free(plan->swiz_st);
fail_free_swiz_rd:
  free(plan->swiz_rd);
fail_free_swiz_sd:
  free(plan->swiz_sd);
fail_free_cyc_types:
  /* Free up the cyclic communication types */
  for(int i = plan->nprocs - 1; i >= 0; --i) {
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->cyc_st + i);
  }
  for(int i = plan->nprocs - 1; i >= 0; --i) {
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->cyc_rt + i);
  }
fail_free_ones:
  free(plan->ones);
fail_free_cyc_rt:
  free(plan->cyc_rt);
fail_free_cyc_st:
  free(plan->cyc_st);
fail_free_cyc_rd:
  free(plan->cyc_rd);
fail_free_cyc_sd:
  free(plan->cyc_sd);
fail_free_outerplan:
  fftw_destroy_plan(plan->outerplan);
fail_free_innerplan:
  fftw_destroy_plan(plan->innerplan);
fail_free_outerdst:
  fftw_free(plan->outerdst);
fail_free_outersrc:
  fftw_free(plan->outersrc);
fail_free_innerdst:
  fftw_free(plan->innerdst);
fail_free_innersrc:
  fftw_free(plan->innersrc);
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
  /* clean up MPI resources */
  int err = MPI_SUCCESS;
  for(int i = plan->nprocs - 1; i >= 0; --i)
  {
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->cyc_st + i);
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->cyc_rt + i);
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->swiz_st + i);
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->swiz_rt + i);
  }

  /* Free up plans */
  fftw_destroy_plan(plan->outerplan);
  fftw_destroy_plan(plan->innerplan);

  /* Free up memory buffers */
  free(plan->swiz_rt);
  free(plan->swiz_st);
  free(plan->swiz_rd);
  free(plan->swiz_sd);
  free(plan->ones);
  free(plan->cyc_rt);
  free(plan->cyc_st);
  free(plan->cyc_rd);
  free(plan->cyc_sd);
  fftw_free(plan->outerdst);
  fftw_free(plan->outersrc);
  fftw_free(plan->innerdst);
  fftw_free(plan->innersrc);
  free(plan->outerjobs);

  free(plan);

  return err;
}

int fft_par_execute(fft_par_plan plan)
{
  int err = MPI_SUCCESS;
  for(int i = 0; i < plan->nelems; ++i) ((double *)plan->innersrc)[i] = -1.0;
  // Stage one: cyclic permutation.
  err = MPI_Alltoallw(plan->src, plan->ones, plan->cyc_sd, plan->cyc_st,
      plan->innersrc, plan->ones, plan->cyc_rd, plan->cyc_rt, plan->comm);
  if (err != MPI_SUCCESS) goto fail_immed;
  // Stage two: inner fft
  fftw_execute(plan->innerplan);
  fft_r2c_1d_finish(plan->innerdst, plan->nelems);
  // Stage three: swizzle data
  err = MPI_Alltoallw(plan->innerdst, plan->ones, plan->swiz_sd, plan->swiz_st,
      plan->outersrc, plan->ones, plan->swiz_rd, plan->swiz_rt, plan->comm);
  if(err != MPI_SUCCESS) goto fail_immed;
  //Stage four: twiddle factors
  apply_twiddle_factors(plan);
  // Stage five: outer fft
  for(int w = 0; w < plan->outerjobs[plan->rank]; ++w) {
    /* Execute the fft for each work unit. The use of fftw_execute_dft is
     * fraught with peril, because we have to ensure we have the proper
     * alignment for each block. The outersrc and outerdst buffers are
     * allocated to 8 or 16 byte alignment, depending on the SIMD requirements
     * of the platform.  The size of a double complex is 16 bytes, so each work
     * unit is thus also 8 or 16 byte aligned.
     */
    fftw_execute_dft(plan->outerplan,
        (double complex *)plan->outersrc + w*plan->nprocs,
        (double complex *)plan->outerdst + w*plan->nprocs);
  }
  // Stage six: rearrange data back to destination
  // This is the reverse of the previous swizzling step
  err = MPI_Alltoallw(plan->outerdst, plan->ones, plan->swiz_rd, plan->swiz_rt,
      plan->dst, plan->ones, plan->swiz_sd, plan->swiz_st, plan->comm);
fail_immed:
  return err;
}

int cycle_data_parameter_fill(const fft_par_plan plan)
{
  /* We only read these fields of plan */
  const int p = plan->rank;
  const int nelems = plan->nelems;
  const int procs = plan->nprocs;

  MPI_Aint lb, ext;
  int err = MPI_Type_get_extent(MPI_DOUBLE, &lb, &ext);
  if(err != MPI_SUCCESS) goto fail_immed;
  // Compute sender offsets and types
  int r = 0;
  while(r < procs) {
    plan->cyc_sd[r] = (((r - p*nelems) % procs + procs) % procs)*ext;

    int sc = nelems/procs +
      (((r - p*nelems) % procs + procs) % procs < nelems % procs ? 1 : 0);

    err = MPI_Type_vector(sc, 1, procs, MPI_DOUBLE, plan->cyc_st + r);
    if(err != MPI_SUCCESS) goto fail_free_st;
    err = MPI_Type_commit(plan->cyc_st + r);
    if(err != MPI_SUCCESS) goto fail_free_st;
    ++r;
  }

  // Compute receiver count and types
  int s = 0;
  while(s < procs) {
    // Displacement is the sum of the receive counts from the preceding
    // processors.
    int rd = 0;
    for(int ss = 0; ss < s; ++ss) {
      rd += nelems/procs +
        (((p - ss*nelems) % procs + procs) % procs < nelems % procs ? 1 : 0);
    }
    // Multiple by the size of an MPI_DOUBLE, because MPI_Alltoallw takes its
    // displacement in bytes.
    plan->cyc_rd[s] = rd*ext;

    int rc = nelems/procs + (((p - s*nelems) % procs + procs) % procs <
        nelems % procs ? 1 : 0);

    err = MPI_Type_contiguous(rc, MPI_DOUBLE, plan->cyc_rt + s);
    if(err != MPI_SUCCESS) goto fail_free_rt;
    err = MPI_Type_commit(plan->cyc_rt + s);
    if(err != MPI_SUCCESS) goto fail_free_rt;
    ++s;
  }

  return err;

  /* This error handling code does nothing, because if we get here, err !=
   * MPI_SUCCESS. This is placed here so that if this function ever needs to
   * accomodate a new source of error, this is here and doesn't need to be
   * created long after I've forgotten how it should be done.
   */
fail_free_rt:
  while(s >= 0) {
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->cyc_rt + s);
  }
fail_free_st:
  while(r >= 0) {
    if(err == MPI_SUCCESS) err = MPI_Type_free(plan->cyc_st + r);
  }
fail_immed:
  return err;
}

int swizzle_data_parameter_fill(const fft_par_plan plan)
{
  /* We only read these members of plan */
  const int procs = plan->nprocs;
  const int rank = plan->rank;
  const int * const ojobs = plan->outerjobs;

  MPI_Aint lb, ext;
  int err = MPI_Type_get_extent(MPI_C_DOUBLE_COMPLEX, &lb, &ext);
  if(err != MPI_SUCCESS) goto fail_immed;

  int r = 0;
  while(r < procs) {
    plan->swiz_sd[r] = r*ext;

    err = MPI_Type_vector(ojobs[r], 1, procs, MPI_C_DOUBLE_COMPLEX,
        plan->swiz_st + r);
    if(err != MPI_SUCCESS) goto fail_free_st;
    err = MPI_Type_commit(plan->swiz_st + r);
    if(err != MPI_SUCCESS) goto fail_free_st;

    ++r;
  }

  int s = 0;
  while(s < procs) {
    plan->swiz_rd[s] = s*ext;

    err = MPI_Type_vector(ojobs[rank], 1, procs, MPI_C_DOUBLE_COMPLEX,
        plan->swiz_rt + s);
    if(err != MPI_SUCCESS) goto fail_free_rt;
    err = MPI_Type_commit(plan->swiz_rt + s);
    if(err != MPI_SUCCESS) goto fail_free_st;

    ++s;
  }

  return err;
fail_free_rt:
  while(s >= 0) {
    if(err == MPI_SUCCESS) MPI_Type_free(plan->swiz_rt + s);

    --s;
  }
fail_free_st:
  while(r >= 0) {
    if(err == MPI_SUCCESS) MPI_Type_free(plan->swiz_st + r);

    --r;
  }
fail_immed:
  return err;
}
