// wrap.h
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


void multiply_mod_matrix_blas(const double *a,
                         const double *b,
                         unsigned long long *result,
                         const unsigned long long  n_a,
                         const unsigned long long  n_b,
                         const unsigned long long  n_c,
                         const unsigned long long p);

                    

void multiply_mod_matrix_blas2(const double *a,
                         const double *b,
                         unsigned long long *result,
                         const unsigned long long  n_a,
                         const unsigned long long  n_b,
                         const unsigned long long  n_c,
                         const unsigned long long p,
                         const unsigned long long bred);



void multiply_mod_matrix_blas_Inplace(const double *a,
                         const double *b,
                         double *result,
                         const unsigned int  n_a,
                         const unsigned int  n_b,
                         const unsigned int  n_c,
                         const unsigned int level);

void multiply_mod_matrix_blas_Inplace2(const double *a,
                         const double *b,
                         double *result,
                         const unsigned int  n_a,
                         const unsigned int  n_b,
                         const unsigned int  n_c);

void multiply_mod_matrix_blas_Inplace_Stride(const double *a,
                         const double *b,
                         double *result,
                         const unsigned int  n_a,
                         const unsigned int  n_b,
                         const unsigned int  n_c,
                         const unsigned int level,
                         const unsigned int lda,const unsigned int ldb,const unsigned int ldc);


                   typedef double _Complex dcomplex;
      
void computeCombinedMat_blas(
    int n,
    int startIdx,
    int count,
    const dcomplex* roots,
    dcomplex div,
    int isInverse,
    dcomplex* out
);

#ifdef __cplusplus
}
#endif