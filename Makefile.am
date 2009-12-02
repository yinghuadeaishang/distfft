AUTOMAKE_OPTIONS = foreign

bin_PROGRAMS = ssfllg

CC = @MPICC@

ssfllg_SOURCES = ssfllg.c
ssfllg_CFLAGS = $(fftw3_CFLAGS)
ssfllg_LDFLAGS = $(AM_LDFLAGS) $(fftw3_LIBS)
