// wrap.c
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
#include <complex.h>
#include <string.h>
typedef double _Complex dcomplex;


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


void multiply_mod_matrix_blas_Inplace(const double *a,
                         const double *b,
                         double *result,
                         const unsigned int  n_a,
                         const unsigned int  n_b,
                         const unsigned int  n_c,
                         const unsigned int level)
{
    for(int i =0;i<level;i++){
        cblas_dgemm(
                CblasRowMajor,   // 메모리 저장 방식 (row-major)
                CblasNoTrans,CblasNoTrans,    // A를 전치하지 않음
                n_a, n_c,n_b,            // 행렬 A의 크기 (m x n)
                1,           // 스케일 값 alpha
                &(a[i*n_a*n_b]), n_b,            // 행렬 A와 leading dimension (n)
                &(b[i*n_b*n_c]), n_c,            // 벡터 x와 stride
                0,            // 스케일 값 beta
                &(result[i*n_a*n_c]), n_c             // 결과 벡터 y와 stride
        );
    }
}
void multiply_mod_matrix_blas_Inplace2(const double *a,
                         const double *b,
                         double *result,
                         const unsigned int  n_a,
                         const unsigned int  n_b,
                         const unsigned int  n_c)
{
    cblas_dgemm(
        CblasRowMajor,   // 메모리 저장 방식 (row-major)
        CblasNoTrans,CblasNoTrans,    // A를 전치하지 않음
        n_a, n_c,n_b,            // 행렬 A의 크기 (m x n)
        1,           // 스케일 값 alpha
        &(a[0]), n_b,            // 행렬 A와 leading dimension (n)
        &(b[0]), n_c,            // 벡터 x와 stride
        0,            // 스케일 값 beta
        &(result[0]), n_c             // 결과 벡터 y와 stride
    );
}




void multiply_mod_matrix_blas_Inplace_Stride(const double *a,
                         const double *b,
                         double *result,
                         const unsigned int  n_a,
                         const unsigned int  n_b,
                         const unsigned int  n_c,
                         const unsigned int level,
                         const unsigned int lda,const unsigned int ldb,const unsigned int ldc)
{
    for(int i =0;i<level;i++){
        cblas_dgemm(
                CblasRowMajor,   // 메모리 저장 방식 (row-major)
                CblasNoTrans,CblasNoTrans,    // A를 전치하지 않음
                n_a, n_c,n_b,            // 행렬 A의 크기 (m x n)
                1,           // 스케일 값 alpha
                &(a[i*n_a*n_b]), lda,            // 행렬 A와 leading dimension (n)
                &(b[i*n_b*n_c]), ldb,            // 벡터 x와 stride
                0,            // 스케일 값 beta
                &(result[i*n_a*n_c]), ldc             // 결과 벡터 y와 stride
        );
    }
}


static inline int pow5_update(int pow5v, int m) {
    // pow5v = (pow5v * 5) & ((m << 2) - 1)
    return (pow5v * 5) & ((m << 2) - 1);
}

static void set_identity_colmajor(int n, double _Complex* out) {
    // out[r + c*n]
    memset(out, 0, (size_t)n * (size_t)n * sizeof(double _Complex));
    for (int i = 0; i < n; i++) {
        out[i + i*n] = 1.0 + 0.0*I;
    }
}

static void applySFStepMatBLAS(int n, int idx, const double _Complex* roots, double _Complex* A) {
    // A is column-major (n x n). Row r across columns is strided by n:
    // elements: A[r + c*n], c=0..n-1
    const int m     = 1 << (idx + 1);
    const int halfM = m >> 1;
    const int gap   = n / m;

    double _Complex* top     = (double _Complex*)malloc((size_t)n * sizeof(double _Complex));
    double _Complex* bot     = (double _Complex*)malloc((size_t)n * sizeof(double _Complex));
    double _Complex* topCopy = (double _Complex*)malloc((size_t)n * sizeof(double _Complex));
    if (!top || !bot || !topCopy) { free(top); free(bot); free(topCopy); return; }

    for (int base = 0; base < n; base += m) {
        int pow5v = 1;
        for (int j = 0; j < halfM; j++) {
            const int k = pow5v * gap;
            const double _Complex w = roots[k];

            const int rTop = base + j;
            const int rBot = base + j + halfM;

            // top = row(rTop), bot = row(rBot)
            // row start address is &A[rTop], stride = n across columns
            cblas_zcopy(n, (const void*)(&A[rTop]), n, (void*)top, 1);
            cblas_zcopy(n, (const void*)(&A[rBot]), n, (void*)bot, 1);

            memcpy(topCopy, top, (size_t)n * sizeof(double _Complex));

            // bot = w * bot
            cblas_zscal(n, (const void*)(&w), (void*)bot, 1);

            // top = top + bot
            {
                const double _Complex one = 1.0 + 0.0*I;
                cblas_zaxpy(n, (const void*)(&one), (const void*)bot, 1, (void*)top, 1);
            }

            // topCopy = topCopy - bot
            {
                const double _Complex minusOne = -1.0 + 0.0*I;
                cblas_zaxpy(n, (const void*)(&minusOne), (const void*)bot, 1, (void*)topCopy, 1);
            }

            // write back
            cblas_zcopy(n, (const void*)top, 1, (void*)(&A[rTop]), n);
            cblas_zcopy(n, (const void*)topCopy, 1, (void*)(&A[rBot]), n);

            pow5v = pow5_update(pow5v, m);
        }
    }

    free(top);
    free(bot);
    free(topCopy);
}

static void applySFIStepMatBLAS(int n, int idx, const double _Complex* roots, double _Complex div, double _Complex* A) {
    const int m     = n >> idx;
    const int halfM = m >> 1;
    const int gap   = n / m;

    const double _Complex invDiv = 1.0 / div;

    double _Complex* a    = (double _Complex*)malloc((size_t)n * sizeof(double _Complex));
    double _Complex* b    = (double _Complex*)malloc((size_t)n * sizeof(double _Complex));
    double _Complex* sum  = (double _Complex*)malloc((size_t)n * sizeof(double _Complex));
    double _Complex* diff = (double _Complex*)malloc((size_t)n * sizeof(double _Complex));
    if (!a || !b || !sum || !diff) { free(a); free(b); free(sum); free(diff); return; }

    for (int base = 0; base < n; base += m) {
        int pow5v = 1;
        for (int j = 0; j < halfM; j++) {
            const int k = pow5v * gap;
            const double _Complex wInv = conj(roots[k]);

            const int rTop = base + j;
            const int rBot = base + j + halfM;

            cblas_zcopy(n, (const void*)(&A[rTop]), n, (void*)a, 1);
            cblas_zcopy(n, (const void*)(&A[rBot]), n, (void*)b, 1);

            memcpy(sum, a, (size_t)n * sizeof(double _Complex));
            memcpy(diff, a, (size_t)n * sizeof(double _Complex));

            // sum = a + b
            {
                const double _Complex one = 1.0 + 0.0*I;
                cblas_zaxpy(n, (const void*)(&one), (const void*)b, 1, (void*)sum, 1);
            }
            // diff = a - b
            {
                const double _Complex minusOne = -1.0 + 0.0*I;
                cblas_zaxpy(n, (const void*)(&minusOne), (const void*)b, 1, (void*)diff, 1);
            }

            // sum = sum/div
            cblas_zscal(n, (const void*)(&invDiv), (void*)sum, 1);

            // diff = (wInv/div) * diff
            {
                const double _Complex alpha = wInv * invDiv;
                cblas_zscal(n, (const void*)(&alpha), (void*)diff, 1);
            }

            cblas_zcopy(n, (const void*)sum, 1, (void*)(&A[rTop]), n);
            cblas_zcopy(n, (const void*)diff, 1, (void*)(&A[rBot]), n);

            pow5v = pow5_update(pow5v, m);
        }
    }

    free(a);
    free(b);
    free(sum);
    free(diff);
}

void computeCombinedMat_blas(
    int n,
    int startIdx,
    int count,
    const dcomplex* roots,
    dcomplex div,
    int isInverse,
    dcomplex* out
) {
    set_identity_colmajor(n, out);

    if (isInverse) {
        for (int l = 0; l < count; l++) {
            applySFIStepMatBLAS(n, startIdx + l, roots, div, out);
        }
    } else {
        for (int l = 0; l < count; l++) {
            applySFStepMatBLAS(n, startIdx + l, roots, out);
        }
    }
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