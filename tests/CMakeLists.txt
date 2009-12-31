cmake_minimum_required(VERSION 2.8)

include_directories(${SSFLLG_SOURCE_DIR} ${FFTW_INCLUDE_DIRS}
  ${MPI_INCLUDE_PATH})
link_directories(${FFTW_LIBRARY_DIRS})

add_executable(fft_r2c_finish_1d fft_r2c_finish_1d.c)
target_link_libraries(fft_r2c_finish_1d fft_ser dsfmt ${FFTW_LIBRARIES})

add_executable(fft_r2c_finish_2d fft_r2c_finish_2d.c)
target_link_libraries(fft_r2c_finish_2d fft_ser dsfmt ${FFTW_LIBRARIES})

add_executable(fft_r2c_finish_nd fft_r2c_finish_nd.c)
target_link_libraries(fft_r2c_finish_nd fft_ser dsfmt ${FFTW_LIBRARIES})

add_executable(fft_par_1d fft_par_1d.c)
target_link_libraries(fft_par_1d fft_par fft_ser dsfmt ${FFTW_LIBRARIES}
  ${MPI_LIBRARIES})

add_executable(fft_par_rev_1d fft_par_rev_1d.c)
target_link_libraries(fft_par_rev_1d fft_par fft_ser dsfmt ${FFTW_LIBRARIES}
  ${MPI_LIBRARIES})

add_executable(fft_par_2d fft_par_2d.c)
target_link_libraries(fft_par_2d fft_par fft_ser dsfmt ${FFTW_LIBRARIES}
  ${MPI_LIBRARIES})

add_executable(fft_par_bench_2d fft_par_bench_2d.c)
target_link_libraries(fft_par_bench_2d fft_par fft_ser dsfmt
  ${FFTW_LIBRARIES} ${MPI_LIBRARIES})

add_test(NAME fft_r2c_finish_1d COMMAND $<TARGET_FILE:fft_r2c_finish_1d>)
add_test(NAME fft_r2c_finish_2d COMMAND $<TARGET_FILE:fft_r2c_finish_2d>)
add_test(NAME fft_r2c_finish_nd COMMAND $<TARGET_FILE:fft_r2c_finish_nd>)
# FIXME: determine the number of processors properly
add_test(NAME fft_par_1d COMMAND
  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
  $<TARGET_FILE:fft_par_1d> ${MPIEXEC_POSTFLAGS})
add_test(NAME fft_par_rev_1d COMMAND
  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
  $<TARGET_FILE:fft_par_rev_1d> ${MPIEXEC_POSTFLAGS})
add_test(NAME fft_par_2d COMMAND
  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
  $<TARGET_FILE:fft_par_2d> ${MPIEXEC_POSTFLAGS})
