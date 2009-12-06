check_PROGRAMS = fft_par_1d

TESTS_ENVIRONMENT = mpirun -np $(CPU_COUNT)
TESTS = fft_par_1d

CC = $(MPICC)

fft_par_1d_SOURCES = fft_par_1d.c
fft_par_1d_LDADD = ../libfft_ser.a ../libfft_par.a ../dSFMT/libdsfmt.a $(fftw3_LIBS)
