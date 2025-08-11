#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// a, b: 입력 행렬 (flattened 1D 배열)
// result: 출력 행렬 (flattened 1D 배열)
// n: 행렬 크기 (n x n)
// p: 소수 (mod p)
void multiply_mod_matrix(const unsigned long long* a, const unsigned long long* b, unsigned long long* result, const unsigned long long n, const char* p_str);

#ifdef __cplusplus
}
#endif