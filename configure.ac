AC_INIT([ssfllg], [0.0], [mcox22@uwo.ca])
AC_CONFIG_SRCDIR([ssfllg.c])
AM_INIT_AUTOMAKE

dnl Find compilers: find a C compiler, then figure out how to compile C99
AC_PROG_CC
AM_PROG_CC_C_O
AC_PROG_CC_C99
dnl Add the argument to compile C99 onto CFLAGS, so MPICC uses it
if test "x${ac_cv_prog_cc_c99}" = "xno"; then
  AC_MSG_ERROR([$CC is unable to compile C99.])
else
  CFLAGS="$ac_cv_prog_cc_c99 $CFLAGS"
fi
dnl Find MPICC
ACX_MPI([], AC_MSG_ERROR([cannot find an MPI compiler.]))

dnl Find Randlib for building libraries
AC_PROG_RANLIB

dnl Figure out the exponent for the Mersenne Twister
AC_ARG_VAR(MTEXP, [the exponent to use for the Mersenne Twister.])
dnl Verify it's an allowable exponent
AC_MSG_CHECKING([exponent for Mersenne Twister])
case $MTEXP in
  "")
    MTEXP=19937
    AC_MSG_RESULT([defaulting to ${MTEXP}.])
    ;;
  521 | 1279 | 2203 | 4253 | 11213 | 19937 | 444937 | 44497 | 86243 | \
      132049 | 216091)
    AC_MSG_RESULT([${MTEXP}.])
    ;;
  *)
    AC_MSG_ERROR([$MTEXP is not a supported exponent.])
    ;;
esac
AC_DEFINE_UNQUOTED([MTEXP], [$MTEXP], [Exponent to use for Mersenne Twister])

dnl Check for SIMD
AX_EXT

dnl Count how many CPUs we have
AX_COUNT_CPUS
AC_SUBST([CPU_COUNT])

dnl Check for packages
PKG_PROG_PKG_CONFIG
PKG_CHECK_MODULES(fftw3, [fftw3 >= 3], [], AC_MSG_ERROR([cannot find fftw3.]))

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile dSFMT/Makefile tests/Makefile par_tests/Makefile])
AC_CONFIG_FILES([dSFMT/compare_output.sh], [chmod +x dSFMT/compare_output.sh])
AC_OUTPUT
