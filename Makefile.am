AUTOMAKE_OPTIONS = foreign

SUBDIRS = dSFMT tests par_tests

bin_PROGRAMS = ssfllg fft_par_bench_2d
noinst_LIBRARIES = libfft_ser.a libfft_par.a

CC = $(MPICC)

ssfllg_SOURCES = ssfllg.c
ssfllg_CFLAGS = $(fftw3_CFLAGS)
ssfllg_LDFLAGS = $(AM_LDFLAGS)
ssfllg_LIBS = libfft_ser.a libfft_par.a $(fftw3_LIBS)

fft_par_bench_2d_SOURCES = fft_par_bench_2d.c
fft_par_bench_2d_LDADD = libfft_par.a libfft_ser.a ./dSFMT/libdsfmt.a $(fftw3_LIBS)

libfft_ser_a_SOURCES = fft_ser.c

libfft_par_a_SOURCES = fft_par.c
