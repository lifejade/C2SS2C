// flint_wrap.c
#include <flint/flint.h>
#include <flint/nmod_mat.h>
#include <stdlib.h>
#include <errno.h>
#include <stdint.h>
#include <gmp.h>
#include <cblas.h>

#include <omp.h>   
#include <stdio.h>
#include <time.h>


void multiply_mod_matrix_flint(const unsigned long long *a,
                         const unsigned long long *b,
                         unsigned long long       *result,
                         const unsigned long long  n_a,
                         const unsigned long long  n_b,
                         const unsigned long long  n_c,
                         const unsigned long long p)
{
    /* 1. 모듈러스 파싱 (64‑bit) */
    flint_set_num_threads(omp_get_max_threads());
    /* 2. 행렬 초기화 */
    nmod_mat_t A, B, C;
    nmod_mat_init(A, n_a, n_b, (mp_limb_t) p);
    nmod_mat_init(B, n_b, n_c, (mp_limb_t) p);
    nmod_mat_init(C, n_a, n_c, (mp_limb_t) p);

    /* 3. Go/NTL 형식(일차원 row‑major) → FLINT 행렬로 복사 */
    for (ulong i = 0; i < n_a; i++)
        for (ulong j = 0; j < n_b; j++) {
            mp_limb_t valA = (mp_limb_t)(a[i * n_b + j] % p);
            nmod_mat_set_entry(A, i, j, valA);
        }

    for (ulong i = 0; i < n_b; i++)
        for (ulong j = 0; j < n_c; j++) {
            mp_limb_t valB = (mp_limb_t)(b[i * n_c + j] % p);
            nmod_mat_set_entry(B, i, j, valB);
        }
    /* 4. 곱셈:  C = A * B  (FLINT이 내부적으로 클래식/Strassen 선택) */
   
    nmod_mat_mul(C, A, B);
    
    /* 5. 결과를 원래 배열 포맷으로 복사 */
    for (ulong i = 0; i < n_a; i++)
        for (ulong j = 0; j < n_c; j++)
            result[i * n_c + j] = nmod_mat_entry(C, i, j);

    /* 6. 정리 */
    nmod_mat_clear(A);
    nmod_mat_clear(B);
    nmod_mat_clear(C);
}




void multiply_mod_matrix_blas(const double *a,
                         const double *b,
                         unsigned long long *result,
                         const unsigned long long  n_a,
                         const unsigned long long  n_b,
                         const unsigned long long  n_c,
                         const unsigned long long p)
{
    double *res = (double*)malloc(sizeof(double) * n_a*n_c);
    cblas_dgemm(
            CblasRowMajor,   // 메모리 저장 방식 (row-major)
            CblasNoTrans,CblasNoTrans,    // A를 전치하지 않음
            n_a, n_c,n_b,            // 행렬 A의 크기 (m x n)
            1,           // 스케일 값 alpha
            a, n_b,            // 행렬 A와 leading dimension (n)
            b, n_c,            // 벡터 x와 stride
            0,            // 스케일 값 beta
            res, n_c             // 결과 벡터 y와 stride
    );

    for(int i = 0; i< n_a * n_c; i++){
        result[i] = (ulong)(res[i]) % p;
    }
}
