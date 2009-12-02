AUTOMAKE_OPTIONS = foreign

SUBDIRS = dSFMT tests

bin_PROGRAMS = ssfllg
noinst_LIBRARIES = libfft_ser.a

CC = $(MPICC)

ssfllg_SOURCES = ssfllg.c
ssfllg_CFLAGS = $(fftw3_CFLAGS)
ssfllg_LDFLAGS = $(AM_LDFLAGS) $(fftw3_LIBS)

libfft_ser_a_SOURCES = fft_ser.c
