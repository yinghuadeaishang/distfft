#/usr/bin/env python3.1

import fbuild
import fbuild.builders.c
import fbuild.config.c as c
import fbuild.config.c.openssl

cflags = {'-std=c99', '-Wall', '-Werror', '-pedantic'}

class popt_h(c.Header):
    poptGetContext = c.function_test('poptContext', 'const char *', 'int',
            'const char **', 'const struct poptOption *', 'int')

def build(ctx):
    # figure out our platform with MPI
    mpi_stc = fbuild.builders.c.guess_static(ctx, exe='mpicc', flags=cflags)
    mpi_stc.build_exe('ssfllg', ['ssfllg.c'])
