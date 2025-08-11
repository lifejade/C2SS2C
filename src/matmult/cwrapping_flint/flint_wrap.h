// flint_wrap.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void multiply_mod_matrix_flint(const unsigned long long *a,
                         const unsigned long long *b,
                         unsigned long long       *result,
                         const unsigned long long  n,
                         const char               *p_str);

#ifdef __cplusplus
}
#endif