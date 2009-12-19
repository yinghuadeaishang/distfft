# This CMakeLists.txt for building dSFMT was created by Matthew W. Cox, who can
# be contacted at matt@mattcox.ca or mcox22@uwo.ca.

# I hereby place the contents of this file into the public domain.
cmake_minimum_required(VERSION 2.8)

include(CheckCCompilerFlag)
include(TestBigEndian)

# Set a cache variable for the mersenne twister exponent. Set a default value
# which the user can override. Check the value to make sure it's valid.
set(MTEXP "19937" CACHE STRING "The exponent to use for the Mersenne Twister.")
set(VALID_EXPONENTS 521;1279;2203;4253;11213;19937;44497;86243;132049;216091)
set(VALID_EXP FALSE)
foreach(EXP ${VALID_EXPONENTS})
  if(MTEXP EQUAL ${EXP})
    set(VALID_EXP TRUE)
    break()
  endif(MTEXP EQUAL ${EXP})
endforeach(EXP ${VALID_EXPONENTS})
if(NOT VALID_EXP)
  message(FATAL_ERROR "${MTEXP} is not an exponent supported by dSFMT.")
else(NOT VALID_EXP)
  message(STATUS "Configured ${MTEXP} as the exponent for dSFMT.")
endif(NOT VALID_EXP)

add_library(dsfmt STATIC dSFMT.c)
add_executable(dsfmt_test test.c)
target_link_libraries(dsfmt_test dsfmt)

# Check for endianness flags
test_big_endian(DSFMT_BIG_ENDIAN)

# Check for SIMD flags
check_c_compiler_flag("-msse2" HAVE_SSE2)
if(HAVE_SSE2)
  message(STATUS "Compiling dSFMT with SSE2")
  set_property(TARGET dsfmt APPEND PROPERTY COMPILE_FLAGS "-msse2")
endif(HAVE_SSE2)

configure_file(config.h.in config.h)
