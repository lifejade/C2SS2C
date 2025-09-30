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
    free(res);
}


static inline __attribute__((always_inline))
unsigned long long barrett_u64(unsigned long long x,
                               unsigned long long p,
                               unsigned long long bred)
{
    unsigned __int128 t = (unsigned __int128)x * (unsigned __int128)bred;
    unsigned long long q = (unsigned long long)(t >> 64);
    unsigned long long r = x - q * p;
    // 분기 없는 조건 감산: r -= (r>=p)? p : 0
    r -= (unsigned long long)-(r >= p) & p;
    return r;
}

void multiply_mod_matrix_blas2(const double *restrict a,
                               const double *restrict b,
                               unsigned long long *restrict result,
                               const unsigned long long n_a,
                               const unsigned long long n_b,
                               const unsigned long long n_c,
                               const unsigned long long p,
                               const unsigned long long bred)
{
    double *res = (double*)aligned_alloc(64, sizeof(double) * n_a * n_c);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)n_a, (int)n_c, (int)n_b,
                1.0, a, (int)n_b,
                     b, (int)n_c,
                1.0, res, (int)n_c);

    const size_t N = (size_t)n_a * (size_t)n_c;

    for (size_t i = 0; i < N; ++i) {
        unsigned long long u = (unsigned long long)res[i];
        result[i] = barrett_u64(u, p, bred);
    }

    free(res);
}

void mul64(unsigned long long x, unsigned long long y, unsigned long long *hi) {
	const unsigned long long mask32 = 1<<32 - 1;
	unsigned long long x0 = x & mask32;
	unsigned long long x1 = x >> 32;
	unsigned long long y0 = y & mask32;
	unsigned long long y1 = y >> 32;
	unsigned long long w0 = x0 * y0;
	unsigned long long t = x1*y0 + w0>>32;
	unsigned long long w1 = t & mask32;
	unsigned long long w2 = t >> 32;
    w1 += x0 * y1;

	(*hi) = x1*y1 + w2 + w1>>32;
}