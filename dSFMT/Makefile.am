noinst_LIBRARIES = libdsfmt.a
check_PROGRAMS = dsfmt_test

libdsfmt_a_SOURCES = dSFMT.c
libdsfmt_a_CFLAGS = $(SIMD_FLAGS)

dsfmt_test_SOURCES = test.c
dsfmt_test_LDADD = libdsfmt.a

TESTS = compare_output.sh
