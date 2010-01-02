cmake_minimum_required(VERSION 2.8)

include_directories(${SSFLLG_SOURCE_DIR} ${FFTW_INCLUDE_DIRS}
  ${MPI_INCLUDE_PATH})
link_directories(${FFTW_LIBRARY_DIRS})

add_executable(fft_r2c_finish_1d fft_r2c_finish_1d.c)
target_link_libraries(fft_r2c_finish_1d fft_ser dsfmt ${FFTW_LIBRARIES})

add_executable(fft_r2c_finish_1d_vec fft_r2c_finish_1d_vec.c)
target_link_libraries(fft_r2c_finish_1d_vec fft_ser dsfmt ${FFTW_LIBRARIES})

add_executable(fft_r2c_finish_2d fft_r2c_finish_2d.c)
target_link_libraries(fft_r2c_finish_2d fft_ser dsfmt ${FFTW_LIBRARIES})

add_executable(fft_r2c_finish_2d_vec fft_r2c_finish_2d_vec.c)
target_link_libraries(fft_r2c_finish_2d_vec fft_ser dsfmt ${FFTW_LIBRARIES})

add_executable(fft_r2c_finish_nd fft_r2c_finish_nd.c)
target_link_libraries(fft_r2c_finish_nd fft_ser dsfmt ${FFTW_LIBRARIES})

add_executable(fft_r2c_finish_nd_vec fft_r2c_finish_nd_vec.c)
target_link_libraries(fft_r2c_finish_nd_vec fft_ser dsfmt ${FFTW_LIBRARIES})

add_executable(fft_par_1d fft_par_1d.c)
target_link_libraries(fft_par_1d fft_par fft_ser dsfmt ${FFTW_LIBRARIES}
  ${MPI_LIBRARIES})

add_executable(fft_par_rev_1d fft_par_rev_1d.c)
target_link_libraries(fft_par_rev_1d fft_par fft_ser dsfmt ${FFTW_LIBRARIES}
  ${MPI_LIBRARIES})

add_executable(fft_par_vec_1d fft_par_vec_1d.c)
target_link_libraries(fft_par_vec_1d fft_par fft_ser dsfmt ${FFTW_LIBRARIES}
  ${MPI_LIBRARIES})

add_executable(fft_par_2d fft_par_2d.c)
target_link_libraries(fft_par_2d fft_par fft_ser dsfmt ${FFTW_LIBRARIES}
  ${MPI_LIBRARIES})

add_executable(fft_par_rev_2d fft_par_rev_2d.c)
target_link_libraries(fft_par_rev_2d fft_par fft_ser dsfmt ${FFTW_LIBRARIES}
  ${MPI_LIBRARIES})

add_executable(fft_par_revvec_2d fft_par_revvec_2d.c)
target_link_libraries(fft_par_revvec_2d fft_par fft_ser dsfmt ${FFTW_LIBRARIES}
  ${MPI_LIBRARIES})

add_executable(fft_par_vec_2d fft_par_vec_2d.c)
target_link_libraries(fft_par_vec_2d fft_par fft_ser dsfmt ${FFTW_LIBRARIES}
  ${MPI_LIBRARIES})

add_executable(fft_par_bench_2d fft_par_bench_2d.c)
target_link_libraries(fft_par_bench_2d fft_par fft_ser dsfmt
  ${FFTW_LIBRARIES} ${MPI_LIBRARIES})

add_test(NAME fft_r2c_finish_1d COMMAND $<TARGET_FILE:fft_r2c_finish_1d>)
add_test(NAME fft_r2c_finish_1d_vec
  COMMAND $<TARGET_FILE:fft_r2c_finish_1d_vec>)
add_test(NAME fft_r2c_finish_2d COMMAND $<TARGET_FILE:fft_r2c_finish_2d>)
add_test(NAME fft_r2c_finish_2d_vec
  COMMAND $<TARGET_FILE:fft_r2c_finish_2d_vec>)
add_test(NAME fft_r2c_finish_nd COMMAND $<TARGET_FILE:fft_r2c_finish_nd>)
add_test(NAME fft_r2c_finish_nd_vec
  COMMAND $<TARGET_FILE:fft_r2c_finish_nd_vec>)
# FIXME: determine the number of processors properly
add_test(NAME fft_par_1d COMMAND
  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
  $<TARGET_FILE:fft_par_1d> ${MPIEXEC_POSTFLAGS})
add_test(NAME fft_par_rev_1d COMMAND
  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
  $<TARGET_FILE:fft_par_rev_1d> ${MPIEXEC_POSTFLAGS})
add_test(NAME fft_par_vec_1d COMMAND
  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
  $<TARGET_FILE:fft_par_vec_1d> ${MPIEXEC_POSTFLAGS})
add_test(NAME fft_par_2d COMMAND
  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
  $<TARGET_FILE:fft_par_2d> ${MPIEXEC_POSTFLAGS})
add_test(NAME fft_par_rev_2d COMMAND
  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
  $<TARGET_FILE:fft_par_rev_2d> ${MPIEXEC_POSTFLAGS})
add_test(NAME fft_par_revvec_2d COMMAND
  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
  $<TARGET_FILE:fft_par_revvec_2d> ${MPIEXEC_POSTFLAGS})
add_test(NAME fft_par_vec_2d COMMAND
  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
  $<TARGET_FILE:fft_par_vec_2d> ${MPIEXEC_POSTFLAGS})
