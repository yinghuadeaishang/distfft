check_PROGRAMS = fft_par_1d fft_par_2d

TESTS_ENVIRONMENT = mpirun -np $(CPU_COUNT)
TESTS = fft_par_1d fft_par_2d

CC = $(MPICC)

fft_par_1d_SOURCES = fft_par_1d.c
fft_par_1d_LDADD = ../libfft_par.a ../libfft_ser.a ../dSFMT/libdsfmt.a $(fftw3_LIBS)

fft_par_2d_SOURCES = fft_par_2d.c
fft_par_2d_LDADD = ../libfft_par.a ../libfft_ser.a ../dSFMT/libdsfmt.a $(fftw3_LIBS)
