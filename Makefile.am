AUTOMAKE_OPTIONS = foreign

SUBDIRS = dSFMT tests par_tests

bin_PROGRAMS = ssfllg
noinst_LIBRARIES = libfft_ser.a libfft_par.a

CC = $(MPICC)

ssfllg_SOURCES = ssfllg.c
ssfllg_CFLAGS = $(fftw3_CFLAGS)
ssfllg_LDFLAGS = $(AM_LDFLAGS)
ssfllg_LIBS = libfft_ser.a libfft_par.a $(fftw3_LIBS)

libfft_ser_a_SOURCES = fft_ser.c

libfft_par_a_SOURCES = fft_par.c
