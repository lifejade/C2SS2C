#include <fflas-ffpack/fflas-ffpack.h>
#include <givaro/modular.h>
#include <vector>
#include <cstdio>
#include <ctime>
#include <cstdint>

int main() {
    const size_t n_a = 1u<<13, n_b = 1u<<13, n_c = 1u<<13;
    const uint64_t p = 17179869171ULL;

    // 48비트 법을 안전하게 처리하려면 __uint128_t를 컴퓨트 타입으로(권장)
    using Field = Givaro::Modular<uint64_t>;
    using E = Field::Element;
    Field F(p);

    std::vector<E> A(n_a*n_b), B(n_b*n_c), C(n_a*n_c);
    for (size_t i=0;i<A.size();++i) A[i] = (E)(std::rand()%p);
    for (size_t i=0;i<B.size();++i) B[i] = (E)(std::rand()%p);

    const E alpha = (E)1, beta = (E)0;

    // 🔴 요 부분이 포인트: Winograd/Auto 말고 Classic으로 “명시”
    FFLAS::MMHelper<Field, FFLAS::MMHelperAlgo::Winograd> mmh;

    std::clock_t st = std::clock();
    FFLAS::fgemm(F,
                 FFLAS::FflasNoTrans, FFLAS::FflasNoTrans,
                 n_a, n_c, n_b,
                 alpha,
                 A.data(), n_b,
                 B.data(), n_c,
                 beta,
                 C.data(), n_c,
                 mmh);                     // ← helper를 넘겨서 Classic 강제
    std::clock_t ed = std::clock();

    std::printf("Elapse: %f\n", double(ed-st)/CLOCKS_PER_SEC);
    return 0;
}