#define _ISOC99_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <error.h>
#include <mpi.h>

#include "ssfllg.h"

int main(int argc, char **argv)
{
  // Error handling scheme: this function has failed until proven otherwise.
  int ret = EXIT_FAILURE;

  if(MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    // Theoretically, an error at this point will abort the program, and this
    // code path is never followed. This is here for completeness.
    error(0,0,"unable to initialize MPI");
    goto skip_mpi_finalize;
  }

  // Install the MPI error handler that returns error codes, so we can perform
  // the usual process suicide ritual.
  if(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN)
      != MPI_SUCCESS) {
    // Again, theoretically, the previous error handler (MPI_Abort) gets called
    // instead of reaching this fail point.
    error(0,0,"unable to reset MPI error handler");
    goto fail;
  }

  int size, rank;
  if(MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS ||
      MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) {
    error(0,0,"unable to determine rank and size");
    goto fail;
  }

  errno = 0;
  if(printf("Processor %d of %d reporting.\n", rank, size) < 0) {
    error(0,errno,"unable to log to stdout");
  } else {
    errno = 0;
    if(fflush(stdout) != 0) {
      error(0,errno,"unable to flush stdout");
    }
  }

  // Error handling scheme changes here: now we have succeeded unless a
  // cleanup function chokes and dies.
  ret = EXIT_SUCCESS;
fail:
  if(MPI_Finalize() != MPI_SUCCESS) {
    error(0,0,"unable to finalize MPI\n");
    ret = EXIT_FAILURE;
  }
skip_mpi_finalize:
  return ret;
}
