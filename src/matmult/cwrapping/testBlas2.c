//go:build ignore
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <time.h>
#include <immintrin.h>


static inline unsigned long long cvttsd_u64_sse2(__m128d x) {
    const __m128d two63 = _mm_set_sd(9223372036854775808.0); // 2^63
    double d = _mm_cvtsd_f64(x);
    if (d <= 0.0) return 0ULL;
    if (d < 9223372036854775808.0) {
        return (unsigned long long)_mm_cvttsd_si64(x);
    } else if (d < 18446744073709551616.0) {
        __m128d y = _mm_sub_sd(x, two63);
        long long t = _mm_cvttsd_si64(y);
        return (1ULL<<63) + (unsigned long long)t;
    } else {
        return ~0ULL; // saturate; 정책에 맞게 조정 가능
    }
}

__attribute__((target("avx512f")))
static unsigned long long f64_to_u64_trunc_avx512(double v){
    __m128d x = _mm_load_sd(&v);
    return _mm_cvttsd_u64(x); // AVX-512 경로
}

__attribute__((target("sse2")))
static unsigned long long f64_to_u64_trunc_sse2_path(double v){
    __m128d x = _mm_load_sd(&v);
    return cvttsd_u64_sse2(x); // 휴대성 경로
}

static inline unsigned long long f64_to_u64_trunc(double v){
    if (__builtin_cpu_supports("avx512f"))
        return f64_to_u64_trunc_avx512(v);
    return f64_to_u64_trunc_sse2_path(v);
}

void f64_to_ui64_sse2_trunc(const double* src, unsigned long long* dst, size_t n){
    for (size_t i = 0; i < n; ++i) dst[i] = f64_to_u64_trunc(src[i]);
}
int main() {
    srand((unsigned int)time(NULL));
    int m = 1 << 16;
    int k = 5;
    int n = 1;
    

    // unsigned long long p = 786433;

    double *A = (double*)malloc(sizeof(double) * k * m);
    
    for(int i = 0;i<k*m;i++){
        A[i] = 1.0f;
    }

    // 벡터 x
    double *x = (double*)malloc(sizeof(double) * n*k);

    // 결과 벡터 y
    double *y = (double*)malloc(sizeof(double) * n*m);

    for(int i = 0;i<n*k;i++){
        x[i] = 0.6;
        y[i] = 0.5;
    }



    // y = alpha*A*x + beta*y
    double alpha = 1.0;
    double beta = 1;

    int total = 1;

    // CBLAS 호출 (행렬-벡터 곱: dgemv)
        clock_t start = clock();
            cblas_dgemm(
            CblasRowMajor,   // 메모리 저장 방식 (row-major)
            CblasNoTrans,CblasNoTrans,    // A를 전치하지 않음
            m, n,k,            // 행렬 A의 크기 (m x n)
            alpha,           // 스케일 값 alpha
            A, k,            // 행렬 A와 leading dimension (n)
            x, n,            // 벡터 x와 stride
            beta,            // 스케일 값 beta
            y, n             // 결과 벡터 y와 stride
        );
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("%d x %d x %d : %fms\n", m,k,n,cpu_time_used);
    printf("%d, %f\n", (int)(y[0]), y[0]);
    fflush(stdout); // ← 중요
    
   
    unsigned long long *y2 = (unsigned long long*)malloc(sizeof(unsigned long long) * n*m);
    
    start = clock();
    f64_to_ui64_sse2_trunc(y,y2, n*m);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("rounding : %fms\n", cpu_time_used);
    printf("%d, %f\n", y2[0], y[0]);


    // // 결과 출력
    // printf("y = [");
    // for (int i = 0; i < m; i++) {
    //     printf(" %f", y[i]);
    // }
    // printf(" ]\n");

    return 0;
}