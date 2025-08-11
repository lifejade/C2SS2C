package test

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"testing"
)

// FindNTTPrime returns the first prime p of bitLen bits such that
// p ≡ 1 (mod modulus).  If it fails to find one after 'trials' attempts,
// it returns nil.
//
//	bitLen  : 원하는 소수의 비트 길이 (예: 26, 52 …)
//	modulus : 2 * NthRoot  (예: NthRoot=131072 → modulus=262144)
//	trials  : 시도 횟수 (충분히 크면 while-true 처럼 사용 가능)
func FindNTTPrime(bitLen int, modulus uint64, trials int) *big.Int {
	mod := new(big.Int).SetUint64(modulus)
	one := big.NewInt(1)

	for i := 0; trials <= 0 || i < trials; i++ {
		// 1) 무작위 k 선택 → cand = k*modulus + 1
		k, _ := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), uint(bitLen))) // < 2^bitLen
		cand := new(big.Int).Mul(k, mod)
		cand.Add(cand, one)

		// 2) 정확한 비트 길이 맞추기
		if cand.BitLen() != bitLen {
			continue
		}
		// 3) Miller–Rabin 프라이멀리티 테스트
		if cand.ProbablyPrime(24) { // 2⁻⁴² 오차. 필요시 반복 횟수 조정
			return cand
		}
	}
	return nil // 실패
}

func FindNTTPrime2(bitLen int, modulus uint64, trials, count int) []*big.Int {
	mod := new(big.Int).SetUint64(modulus)
	one := big.NewInt(1)

	result := make([]*big.Int, count)
	k, _ := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), uint(bitLen))) // < 2^bitLen
	for i, idx := 0, 0; trials <= 0 || i < trials; i++ {
		// 1) 무작위 k 선택 → cand = k*modulus + 1

		cand := new(big.Int).Mul(k, mod)
		cand.Add(cand, one)
		k.Add(k, one)
		// 2) 정확한 비트 길이 맞추기
		if cand.BitLen() != bitLen {
			continue
		}
		// 3) Miller–Rabin 프라이멀리티 테스트
		if cand.ProbablyPrime(24) { // 2⁻⁴² 오차. 필요시 반복 횟수 조정
			result[idx] = cand
			idx++
		}
		if idx >= count {
			break
		}

	}
	return result
}

func Test_Prime(t *testing.T) {
	const (
		logQ       = 48
		logN       = 11 // N = 2¹¹ = 2048
		mod1InvDeg = 0  // Mod1InvDegree
	)
	nthRootLog := logN + mod1InvDeg          // = 15
	modulus := uint64(1) << (nthRootLog + 1) // 2*NthRoot = 2^(15+1)

	k := 8
	Q := make([]uint64, 0, k)
	for len(Q) < k {
		if p := FindNTTPrime(logQ, modulus, 0); p != nil {
			Q = append(Q, p.Uint64())
		}
	}
	for i := range Q {
		fmt.Print(Q[i], ",")
	}
}

func Test_Prime2(t *testing.T) {
	const (
		logQ       = 32 //
		logN       = 11 // N = 2¹¹ = 2048
		mod1InvDeg = 4  // Mod1InvDegree
	)
	nthRootLog := logN + mod1InvDeg          // = 15
	modulus := uint64(1) << (nthRootLog + 1) // 2*NthRoot = 2^(15+1)
	fmt.Println(modulus)
	// 26-bit 소수 2개 뽑기
	Q := FindNTTPrime2(26, modulus, 0, 4)
	fmt.Println(Q)
}
