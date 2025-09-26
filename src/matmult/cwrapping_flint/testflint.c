//go:build ignore
#include <flint/flint.h>
#include <flint/nmod_mat.h>
#include <stdlib.h>
#include <errno.h>
#include <stdint.h>
#include <gmp.h>

#include <omp.h>   
#include <stdio.h>
#include <time.h>


int main(void)
{
    flint_set_num_threads(1);
    
    int n_a = 1 << 13;
    int n_b = 1 << 13;
    int n_c = 1 << 13;
    //48bit
    unsigned long long p = 242957446545409;
    
    unsigned long long *a = (unsigned long long *)malloc(sizeof(unsigned long long) * n_a*n_b);
    unsigned long long *b = (unsigned long long *)malloc(sizeof(unsigned long long) * n_b*n_c);
    unsigned long long *c = (unsigned long long *)malloc(sizeof(unsigned long long) * n_a*n_c);

    // unsigned long long *a2 = (unsigned long long *)malloc(sizeof(unsigned long long) * n_a*n_b);
    // unsigned long long *b2 = (unsigned long long *)malloc(sizeof(unsigned long long) * n_b*n_c);
    // unsigned long long *c2 = (unsigned long long *)malloc(sizeof(unsigned long long) * n_a*n_c);

    unsigned long long *result = (unsigned long long *)malloc(sizeof(unsigned long long) * n_a*n_c);
    
    
    for(int i = 0;i<n_a*n_b;i++){
        a[i] = rand() % p;
        b[i] = rand() % p;
        c[i] = rand() % p;

        // a2[i] = rand() % p2;
        // b2[i] = rand() % p2;
        // c2[i] = rand() % p2;
    }

    /* 2. 행렬 초기화 */
    nmod_mat_t A, B, C, A2, B2, C2;
    nmod_mat_init(A, n_a, n_b, (mp_limb_t) p);
    nmod_mat_init(B, n_b, n_c, (mp_limb_t) p);
    nmod_mat_init(C, n_a, n_c, (mp_limb_t) p);
    // nmod_mat_init(A2, n_a, n_b, (mp_limb_t) p2);
    // nmod_mat_init(B2, n_b, n_c, (mp_limb_t) p2);
    // nmod_mat_init(C2, n_a, n_c, (mp_limb_t) p2);
    
    
    
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
    clock_t start = clock_gettime(CLOCK_MONOTONIC);

    // ////////////////////////
    // for (ulong i = 0; i < n_a; i++)
    //     for (ulong j = 0; j < n_b; j++) {
    //         mp_limb_t valA = (mp_limb_t)(a2[i * n_b + j] % p2);
    //         nmod_mat_set_entry(A2, i, j, valA);
    //     }

    // for (ulong i = 0; i < n_b; i++)
    //     for (ulong j = 0; j < n_c; j++) {
    //         mp_limb_t valB = (mp_limb_t)(b2[i * n_c + j] % p2);
    //         nmod_mat_set_entry(B2, i, j, valB);
    //     }
    /* 4. 곱셈:  C = A * B  (FLINT이 내부적으로 클래식/Strassen 선택) */
    nmod_mat_mul(C, A, B);
    // nmod_mat_mul(C2, A2, B2);



    // for (ulong i = 0; i < n_a; i++)
    //     for (ulong j = 0; j < n_c; j++)
    //         result[i * n_c + j] = nmod_mat_entry(C2, i, j);

    
    clock_t end = clock_gettime(CLOCK_MONOTONIC);
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Elapse : %f\n ", cpu_time_used);
        for (ulong i = 0; i < n_a; i++)
        for (ulong j = 0; j < n_c; j++)
            result[i * n_c + j] = nmod_mat_entry(C, i, j);
        /* 5. 결과를 원래 배열 포맷으로 복사 */

        // 7) 정리
    nmod_mat_clear(A);
    nmod_mat_clear(B);
    nmod_mat_clear(C);
    // nmod_mat_clear(A2);
    // nmod_mat_clear(B2);
    // nmod_mat_clear(C2);

    free(a); free(b); free(c);
    // free(a2); free(b2); free(c2);
    free(result);
}