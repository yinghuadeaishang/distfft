check_PROGRAMS = fft_r2c_finish_1d fft_r2c_finish_2d \
		 fft_r2c_finish_nd

fft_r2c_finish_1d_SOURCES = fft_r2c_finish_1d.c
fft_r2c_finish_1d_LDADD = ../dSFMT/libdsfmt.a ../libfft_ser.a
fft_r2c_finish_1d_LDFLAGS = $(AM_LDFLAGS) $(fftw3_LIBS)

fft_r2c_finish_2d_SOURCES = fft_r2c_finish_2d.c
fft_r2c_finish_2d_LDADD = ../dSFMT/libdsfmt.a ../libfft_ser.a
fft_r2c_finish_2d_LDFLAGS = $(AM_LDFLAGS) $(fftw3_LIBS)

fft_r2c_finish_nd_SOURCES = fft_r2c_finish_nd.c
fft_r2c_finish_nd_LDADD = ../dSFMT/libdsfmt.a ../libfft_ser.a
fft_r2c_finish_nd_LDFLAGS = $(AM_LDFLAGS) $(fftw3_LIBS)

TESTS = $(check_PROGRAMS)
