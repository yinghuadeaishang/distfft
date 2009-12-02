AC_INIT([ssfllg], [0.0], [mcox22@uwo.ca])
AC_CONFIG_SRCDIR([ssfllg.c])
AM_INIT_AUTOMAKE

dnl Find compilers: find a C compiler, then figure out how to compile C99
AC_PROG_CC
AC_PROG_CC_C99
dnl Add the argument to compile C99 onto CFLAGS, so MPICC uses it
if test "x${ac_cv_prog_cc_c99}" = "xno"; then
  AC_MSG_ERROR([$CC is unable to compile C99.])
else
  CFLAGS="$ac_cv_prog_cc_c99 $CFLAGS"
fi
dnl Find MPICC
ACX_MPI([], AC_MSG_ERROR([cannot find an MPI compiler.]))

dnl Check for packages
PKG_PROG_PKG_CONFIG
PKG_CHECK_MODULES(fftw3, [fftw3 >= 3], [], AC_MSG_ERROR([cannot find fftw3.]))

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
