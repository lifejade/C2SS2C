// flint_wrap.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif


void multiply_mod_matrix_flint(const unsigned long long *a,
                         const unsigned long long *b,
                         unsigned long long       *result,
                         const unsigned long long  n_a,
                         const unsigned long long  n_b,
                         const unsigned long long  n_c,
                         const unsigned long long p);

#ifdef __cplusplus
}
#endif