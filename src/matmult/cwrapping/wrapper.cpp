#include "wrapper.hpp"
#include <NTL/mat_ZZ_p.h>

using namespace NTL;

extern "C" {
    void multiply_mod_matrix(const unsigned long long* a, const unsigned long long* b, unsigned long long* result, const unsigned long long n, const char* p_str) {
        ZZ p;
        p = to_ZZ(p_str);
        ZZ_p::init(to_ZZ(p));

        Mat<ZZ_p> A, B, C;
        A.SetDims(n, n);
        B.SetDims(n, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = a[i * n + j];
                B[i][j] = b[i * n + j];
            }
        }

        C = A * B;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i * n + j] = static_cast<unsigned long long>(conv<long>(rep(C[i][j])));
            }
        }
    }
}
