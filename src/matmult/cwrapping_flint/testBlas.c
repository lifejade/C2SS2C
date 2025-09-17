//go:build ignore
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <time.h>

int main() {
    srand((unsigned int)time(NULL));
    int m = 1 << 13; // 행 개수
    int n = 1 << 13; // 열 개수
    int k= 1<<13;

    unsigned long long p = 786433;

    double *A = (double*)malloc(sizeof(double) * n * m);
    
    for(int i = 0;i<n*m;i++){
        A[i] = (double)(rand() % p);
    }

    // 벡터 x
    double *x = (double*)malloc(sizeof(double) * n*m);

    // 결과 벡터 y
    double *y = (double*)malloc(sizeof(double) * n*m);

    for(int i = 0;i<n*m;i++){
        x[i] = (double)(rand() % p);
        y[i] = 0;
    }



    // y = alpha*A*x + beta*y
    double alpha = 1.0;
    double beta = 0.0;

    int total = 3;

    // CBLAS 호출 (행렬-벡터 곱: dgemv)
        clock_t start = clock();
    for(int i=0;i<total;i++){
        cblas_dgemm(
            CblasRowMajor,   // 메모리 저장 방식 (row-major)
            CblasNoTrans,CblasNoTrans,    // A를 전치하지 않음
            m, n,k,            // 행렬 A의 크기 (m x n)
            alpha,           // 스케일 값 alpha
            A, n,            // 행렬 A와 leading dimension (n)
            x, n,            // 벡터 x와 stride
            beta,            // 스케일 값 beta
            y, n             // 결과 벡터 y와 stride
        );
    }
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Elapse : %f\n ", cpu_time_used);


    // // 결과 출력
    // printf("y = [");
    // for (int i = 0; i < m; i++) {
    //     printf(" %f", y[i]);
    // }
    // printf(" ]\n");

    return 0;
}