package test

import (
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/shirou/gopsutil/v3/process"

	"github.com/lifejade/mm/src/matmult"
	"github.com/lifejade/mm/src/transpose"
	"github.com/lifejade/mm/src/util"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/schemes/ckks"
	"github.com/tuneinsight/lattigo/v5/utils/bignum"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
)

// 행렬 크기 확인 (직사각형/공백 방지)
func dims(m [][]complex128) (r, c int, ok bool) {
	r = len(m)
	if r == 0 {
		return 0, 0, false
	}
	c = len(m[0])
	if c == 0 {
		return 0, 0, false
	}
	for i := 1; i < r; i++ {
		if len(m[i]) != c {
			return 0, 0, false
		}
	}
	return r, c, true
}

// 단위 행렬 생성 (n x n)
func identity(n int) [][]complex128 {
	I := make([][]complex128, n)
	for i := range I {
		I[i] = make([]complex128, n)
		I[i][i] = complex(1, 0)
	}
	return I
}

// 행렬 곱 (r x k) * (k x c) = (r x c)
func mul(a, b [][]complex128) [][]complex128 {
	ar, ac, okA := dims(a)
	br, bc, okB := dims(b)
	if !okA || !okB || ac != br {
		return nil
	}
	out := make([][]complex128, ar)
	for i := 0; i < ar; i++ {
		out[i] = make([]complex128, bc)
		for j := 0; j < bc; j++ {
			var s complex128
			for k := 0; k < ac; k++ {
				s += a[i][k] * b[k][j]
			}
			out[i][j] = s
		}
	}
	return out
}

// 두 복소수 행렬이 tol 이내로 같은지 (엔트리별 복소수 노름 비교)
func closeMat(a, b [][]complex128, tol float64) bool {
	ar, ac, okA := dims(a)
	br, bc, okB := dims(b)
	if !okA || !okB || ar != br || ac != bc {
		return false
	}
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			if cmplx.Abs(a[i][j]-b[i][j]) > tol {
				return false
			}
		}
	}
	return true
}

// 역행렬 관계 확인: A*B ≈ I 그리고 B*A ≈ I
func AreInverses(A, B [][]complex128, tol float64) bool {
	n, m, okA := dims(A)
	n2, m2, okB := dims(B)
	if !okA || !okB || n != m || n2 != m2 || n != n2 {
		return false // 정사각 + 같은 크기 필요
	}
	I := identity(n)
	AB := mul(A, B)
	BA := mul(B, A)
	if AB == nil || BA == nil {
		return false
	}

	for i := range AB {
		for j := range AB[i] {
			fmt.Print((AB[i][j]), " ")
		}
		fmt.Println()
	}
	fmt.Println()

	for i := range BA {
		for j := range BA[i] {
			fmt.Print((BA[i][j]), " ")
		}
		fmt.Println()
	}
	return closeMat(BA, I, tol)
}

func Test_BootBasic(t *testing.T) {
	logN := 15
	galLen := 1
	// sparses := []int{5}
	// lenCL := []int{2}

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            logN,
		LogQ:            []int{48, 40, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}

	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	fmt.Println("galLen : ", galLen)

	// generate keys - Rotating key
	galEls := make([]uint64, galLen)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())
	rtk = make([]*rlwe.GaloisKey, len(galEls))
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		i := i
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
		}()
	}
	wg.Wait()

	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	_ = evaluator
	fmt.Println("ckks parameter init end")
	CoeffsToSlotsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicEncode,
		Format:       hefloat.RepackImagAsReal, // Returns the real and imaginary part into separate ciphertexts
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel(),
		Levels:       []int{1, 1}, //qiCoeffsToSlots
		LogBSGSRatio: 0,
		BitReversed:  false,
	}

	// Parameters of the homomorphic modular reduction x mod 1
	Mod1ParametersLiteral := hefloat.Mod1ParametersLiteral{
		LevelStart:      params.MaxLevel() - 2,
		LogScale:        48,                  // Matches qiEvalMod
		Mod1Type:        hefloat.CosDiscrete, // Multi-interval Chebyshev interpolation
		Mod1Degree:      63,                  // Depth 6
		DoubleAngle:     3,                   // Depth 3
		K:               31,                  // With EphemeralSecretWeight = 32 and 2^{15} slots, ensures < 2^{-138.7} failure probability
		LogMessageRatio: 8,                   // q/|m| = 2^10
		Mod1InvDegree:   0,                   // Depth 0
	}

	// SlotsToCoeffs parameters (homomorphic decoding)
	SlotsToCoeffsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicDecode,
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel() - 11,
		Levels:       []int{1, 1}, // qiSlotsToCoeffs
		LogBSGSRatio: 0,
		BitReversed:  false,
	}

	// Custom bootstrapping.Parameters.
	// All fields are public and can be manually instantiated.
	btpParams := bootstrapping.Parameters{
		ResidualParameters:      params,
		BootstrappingParameters: params,
		SlotsToCoeffsParameters: SlotsToCoeffsParameters,
		Mod1ParametersLiteral:   Mod1ParametersLiteral,
		CoeffsToSlotsParameters: CoeffsToSlotsParameters,
		EphemeralSecretWeight:   0, // > 128bit secure for LogN=16 and LogQP = 115.
		CircuitOrder:            bootstrapping.Custom,
	}
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)
	btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	if err != nil {
		panic(err)
	}
	value := make([]float64, params.N())
	for i := range value {
		value[i] = 0.0001 * float64(i)
	}
	plaintext := hefloat.NewPlaintext(params, params.MaxLevel())
	plaintext.IsBatched = false
	encoder.Encode(value, plaintext)
	ct, _ := encryptor.EncryptNew(plaintext)
	// runtime.GOMAXPROCS(1)
	starttime := time.Now()
	result1, result2, _ := btp.CoeffsToSlots(ct)
	elapse := time.Since(starttime)
	fmt.Println()
	fmt.Println("#############################################################")
	fmt.Println("Original Total Elapse", elapse)

	dept := decryptor.DecryptNew(result1)
	dept.IsBatched = true
	encoder.Decode(dept, value)
	fmt.Println(value[:100])

	dept = decryptor.DecryptNew(result2)
	encoder.Decode(dept, value)
	fmt.Println(value[:100])

}

func Test_Inverse(t *testing.T) {

	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            11,
		LogQ:            []int{48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}
	//parameter init
	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}
	n := 1 << params.LogMaxSlots()

	roots := ckks.GetRootsBigComplex(n<<2, params.EncodingPrecision())
	roots_complex := make([]complex128, 4*n)

	for i := range roots_complex {
		roots_complex[i] = roots[i].Complex128()
	}
	fmt.Println()

	pow5 := make([]int, (n<<1)+1)
	pow5[0] = 1
	for i := 1; i < (n<<1)+1; i++ {
		pow5[i] = pow5[i-1] * 5
		pow5[i] &= (n << 2) - 1
	}

	SF := make([][]complex128, n)

	for i := range SF {
		SF[i] = make([]complex128, n)
		for j := range SF[i] {
			idx := (pow5[i] * j) & ((n << 2) - 1)
			SF[i][j] = roots_complex[idx]
		}
	}

	SFI := make([][]complex128, n)
	for i := range SFI {
		SFI[i] = make([]complex128, n)
	}
	for i := range SFI {
		for j := range SFI[i] {
			idx := (pow5[i] * j) & ((n << 2) - 1)
			SFI[j][i] = cmplx.Conj(roots_complex[idx]) / complex((float64(n)), 0)
		}
	}
	fmt.Println()

	fmt.Println("A와 B는 서로 역행렬인가?", AreInverses(SF, SFI, 0.0001))

}

func Test_MMFloat(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            5,
		LogQ:            []int{48, 40, 40, 40, 40, 40, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}
	//parameter init
	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}

	// //====================================
	// //=== 2) BOOTSTRAPPING PARAMETERS ===
	// //====================================

	// // CoeffsToSlots parameters (homomorphic encoding)
	// CoeffsToSlotsParameters := hefloat.DFTMatrixLiteral{
	// 	Type:         hefloat.HomomorphicEncode,
	// 	Format:       hefloat.RepackImagAsReal, // Returns the real and imaginary part into separate ciphertexts
	// 	LogSlots:     params.LogMaxSlots(),
	// 	LevelStart:   params.MaxLevel(),
	// 	Levels:       []int{1, 1}, //qiCoeffsToSlots
	// 	LogBSGSRatio: 0,
	// }

	// // Parameters of the homomorphic modular reduction x mod 1
	// Mod1ParametersLiteral := hefloat.Mod1ParametersLiteral{
	// 	LevelStart:      params.MaxLevel() - 2,
	// 	LogScale:        48,                  // Matches qiEvalMod
	// 	Mod1Type:        hefloat.CosDiscrete, // Multi-interval Chebyshev interpolation
	// 	Mod1Degree:      24,                  // Depth 5
	// 	DoubleAngle:     3,                   // Depth 3
	// 	K:               8,                   // With EphemeralSecretWeight = 32 and 2^{15} slots, ensures < 2^{-138.7} failure probability
	// 	LogMessageRatio: 8,                   // q/|m| = 2^10
	// 	Mod1InvDegree:   0,                   // Depth 0
	// }

	// // SlotsToCoeffs parameters (homomorphic decoding)
	// SlotsToCoeffsParameters := hefloat.DFTMatrixLiteral{
	// 	Type:         hefloat.HomomorphicDecode,
	// 	LogSlots:     params.LogMaxSlots(),
	// 	LevelStart:   params.MaxLevel() - 10,
	// 	Levels:       []int{1, 1}, // qiSlotsToCoeffs
	// 	LogBSGSRatio: 0,
	// }

	// // Custom bootstrapping.Parameters.
	// // All fields are public and can be manually instantiated.
	// btpParams := bootstrapping.Parameters{
	// 	ResidualParameters:      params,
	// 	BootstrappingParameters: params,
	// 	SlotsToCoeffsParameters: SlotsToCoeffsParameters,
	// 	Mod1ParametersLiteral:   Mod1ParametersLiteral,
	// 	CoeffsToSlotsParameters: CoeffsToSlotsParameters,
	// 	EphemeralSecretWeight:   32, // > 128bit secure for LogN=16 and LogQP = 115.
	// 	CircuitOrder:            bootstrapping.Custom,
	// }

	fmt.Println("ckks parameter init end")

	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	n := 1 << params.LogMaxSlots()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, 2*n)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	starttime := time.Now()
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
		}()
	}
	wg.Wait()
	elapse := time.Since(starttime)
	fmt.Println(elapse)
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)

	fmt.Println("generate Evaluator end")

	_, _, _, _ = encoder, encryptor, decryptor, evaluator

	matscale := float64(1 << 15)
	mat_00 := make([][][]uint64, len(params.Q()))
	for q := range mat_00 {
		mat_00[q] = make([][]uint64, 2*n)
		for i := range mat_00[q] {
			mat_00[q][i] = make([]uint64, 2*n)
			for j := range mat_00[q][i] {
				mat_00[q][i][j] = uint64(2 * matscale)
			}
		}
	}

	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, 2)
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	cts := make([]*rlwe.Ciphertext, 2*n)
	for i := range cts {
		cts[i] = ct.CopyNew()
	}

	ctT := transpose.Transpose(cts, params, evaluator, encoder, 2*n)
	util.DebugCTS(ctT, params, encoder, decryptor)
	res00 := matmult.PPMM_Flint_CRT(ctT, mat_00, params, 2*n)
	for i := range res00 {
		evaluator.Mul(res00[i], 1/matscale, res00[i])
		evaluator.Rescale(res00[i], res00[i])
	}

	util.DebugCTS(res00, params, encoder, decryptor)
	result0 := transpose.Transpose(res00, params, evaluator, encoder, 2*n)
	util.DebugCTS(result0, params, encoder, decryptor)

	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")

	// resvalue := make([]complex128, n)
	// for i := range result0 {
	// 	result0[i].IsBatched = true
	// 	dept := decryptor.DecryptNew(result0[i])
	// 	encoder.Decode(dept, resvalue)

	// 	fmt.Println(resvalue[0:10])
	// }
}

func Test_C2S(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            5,
		LogQ:            []int{50, 50, 50},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}
	//parameter init
	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}

	// //====================================
	// //=== 2) BOOTSTRAPPING PARAMETERS ===
	// //====================================

	// // CoeffsToSlots parameters (homomorphic encoding)
	// CoeffsToSlotsParameters := hefloat.DFTMatrixLiteral{
	// 	Type:         hefloat.HomomorphicEncode,
	// 	Format:       hefloat.RepackImagAsReal, // Returns the real and imaginary part into separate ciphertexts
	// 	LogSlots:     params.LogMaxSlots(),
	// 	LevelStart:   params.MaxLevel(),
	// 	Levels:       []int{1, 1}, //qiCoeffsToSlots
	// 	LogBSGSRatio: 0,
	// }

	// // Parameters of the homomorphic modular reduction x mod 1
	// Mod1ParametersLiteral := hefloat.Mod1ParametersLiteral{
	// 	LevelStart:      params.MaxLevel() - 2,
	// 	LogScale:        48,                  // Matches qiEvalMod
	// 	Mod1Type:        hefloat.CosDiscrete, // Multi-interval Chebyshev interpolation
	// 	Mod1Degree:      24,                  // Depth 5
	// 	DoubleAngle:     3,                   // Depth 3
	// 	K:               8,                   // With EphemeralSecretWeight = 32 and 2^{15} slots, ensures < 2^{-138.7} failure probability
	// 	LogMessageRatio: 8,                   // q/|m| = 2^10
	// 	Mod1InvDegree:   0,                   // Depth 0
	// }

	// // SlotsToCoeffs parameters (homomorphic decoding)
	// SlotsToCoeffsParameters := hefloat.DFTMatrixLiteral{
	// 	Type:         hefloat.HomomorphicDecode,
	// 	LogSlots:     params.LogMaxSlots(),
	// 	LevelStart:   params.MaxLevel() - 10,
	// 	Levels:       []int{1, 1}, // qiSlotsToCoeffs
	// 	LogBSGSRatio: 0,
	// }

	// // Custom bootstrapping.Parameters.
	// // All fields are public and can be manually instantiated.
	// btpParams := bootstrapping.Parameters{
	// 	ResidualParameters:      params,
	// 	BootstrappingParameters: params,
	// 	SlotsToCoeffsParameters: SlotsToCoeffsParameters,
	// 	Mod1ParametersLiteral:   Mod1ParametersLiteral,
	// 	CoeffsToSlotsParameters: CoeffsToSlotsParameters,
	// 	EphemeralSecretWeight:   32, // > 128bit secure for LogN=16 and LogQP = 115.
	// 	CircuitOrder:            bootstrapping.Custom,
	// }

	fmt.Println("ckks parameter init end")

	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	n := 1 << params.LogMaxSlots()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, 2*n)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	starttime := time.Now()
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
		}()
	}
	wg.Wait()
	elapse := time.Since(starttime)
	fmt.Println(elapse)
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	// btpevk, _, _ := btpParams.GenEvaluationKeys(sk)

	// btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	// if err != nil {
	// 	panic(err)
	// }
	fmt.Println("generate Evaluator end")

	_, SFI := matmult.GenSFMat(params)
	scale := float64(1 << 40)
	mat0, mat1, mat2, mat3 := matmult.GenC2SMat(SFI, scale, params)
	// mat_00 := make([][][]uint64, len(params.Q()))
	// mat_01 := make([][][]uint64, len(params.Q()))
	// scale := float64(1 << 40)
	// for q := range mat_00 {
	// 	mat_00[q] = make([][]uint64, 2*n)
	// 	mat_01[q] = make([][]uint64, 2*n)
	// 	for i := range 2 * n {
	// 		mat_00[q][i] = make([]uint64, 2*n)
	// 		mat_01[q][i] = make([]uint64, 2*n)
	// 		for j := range 2 * n {
	// 			if i < n && j < n {
	// 				if real(SFI[i][j]) >= 0 {
	// 					mat_00[q][i][j] = uint64(real(SFI[i][j]) * scale)
	// 				} else {
	// 					mat_00[q][i][j] = uint64(int(params.Q()[q]) + int(real(SFI[i][j])*scale))
	// 				}

	// 			} else {
	// 				mat_00[q][i][j] = 0
	// 			}
	// 			if i >= n && j >= n {
	// 				if imag(SFI[i%n][j%n]) >= 0 {
	// 					mat_01[q][i][j] = uint64(imag(SFI[i%n][j%n]) * scale)
	// 				} else {
	// 					mat_01[q][i][j] = uint64(int(params.Q()[q]) + int(imag(SFI[i%n][j%n])*scale))
	// 				}

	// 			} else {
	// 				mat_01[q][i][j] = 0
	// 			}
	// 		}
	// 	}
	// }
	// fmt.Println(params.Q()[0])

	// for i := range mat_00[0] {
	// 	for j := range mat_00[0][i] {
	// 		fmt.Print(mat_00[0][i][j], " ")
	// 	}
	// 	fmt.Println()
	// }
	// for i := range SFI {
	// 	for j := range SFI[i] {
	// 		fmt.Print(imag(SFI[i][j]), " ")
	// 	}
	// 	fmt.Println()
	// }

	// mat_10 := make([][][]uint64, 2*n)
	// mat_11 := make([][][]uint64, 2*n)
	// for q := range len(params.Q()) {
	// 	mat_10[q] = make([][]uint64, 2*n)
	// 	mat_11[q] = make([][]uint64, 2*n)
	// 	for i := range 2 * n {
	// 		mat_10[q][i] = make([]uint64, 2*n)
	// 		mat_11[q][i] = make([]uint64, 2*n)
	// 		for j := range 2 * n {
	// 			if i < n && j < n {
	// 				if real(SFI[i][j]) >= 0 {
	// 					mat_10[q][i][j] = uint64(int(params.Q()[q]) - int(real(SFI[i][j])*scale))
	// 				} else {
	// 					mat_10[q][i][j] = uint64(-real(SFI[i][j]) * scale)
	// 				}
	// 			}
	// 			if i >= n && j >= n {
	// 				if imag(SFI[i%n][j%n]) >= 0 {
	// 					mat_11[q][i][j] = uint64(imag(SFI[i%n][j%n]) * scale)
	// 				} else {
	// 					mat_11[q][i][j] = uint64(int(params.Q()[q]) + int(imag(SFI[i%n][j%n])*scale))
	// 				}

	// 			}

	// 		}
	// 	}
	// }

	// _, _, _, _ = encoder, encryptor, decryptor, evaluator

	// value := make([]float64, 2*n)
	// for i := range value {
	// 	value[i] = 0.001 * float64(i)
	// }

	// pt := hefloat.NewPlaintext(params, 1)
	// pt.IsBatched = false

	// encoder.Encode(value, pt)
	// ct, _ := encryptor.EncryptNew(pt)
	// cts := make([]*rlwe.Ciphertext, 2*n)
	// for i := range cts {
	// 	cts[i] = ct.CopyNew()
	// }

	// starttime = time.Now()
	// starttime_ := time.Now()
	// ctT := transpose.Transpose(cts, params, evaluator, encoder, 2*n)
	// elapse_ := time.Since(starttime_)
	// fmt.Println("transpose time per once : ", elapse_)

	// ctT2 := make([]*rlwe.Ciphertext, 2*n)
	// for i := range ctT2 {
	// 	if i < n {
	// 		ctT2[i], _ = evaluator.MulNew(ctT[i+n], -1)
	// 	} else {
	// 		ctT2[i] = ctT[i-n].CopyNew()
	// 	}
	// }

	// res00 := matmult.PPMM_Flint_CRT(ctT, mat_00, params, 2*n)
	// res01 := matmult.PPMM_Flint_CRT(ctT2, mat_01, params, 2*n)
	// util.DebugCTS(res00, params, encoder, decryptor)
	// util.DebugCTS(res01, params, encoder, decryptor)
	// res0 := make([]*rlwe.Ciphertext, 2*n)
	// for i := range res0 {
	// 	res0[i], _ = evaluator.AddNew(res00[i], res01[i])
	// 	evaluator.Mul(res0[i], 1.0/(scale), res0[i])
	// 	evaluator.Rescale(res0[i], res0[i])

	// }
	// util.DebugCTS(res0, params, encoder, decryptor)

	// res10 := matmult.PPMM_Flint_CRT(ctT2, mat_10, params, 2*n)
	// res11 := matmult.PPMM_Flint_CRT(ctT, mat_11, params, 2*n)
	// res1 := make([]*rlwe.Ciphertext, 2*n)
	// for i := range res1 {
	// 	res1[i], _ = evaluator.AddNew(res10[i], res11[i])
	// 	evaluator.Mul(res1[i], 1.0/scale, res1[i])
	// 	evaluator.Rescale(res1[i], res1[i])
	// }

	// result0 := transpose.Transpose(res0, params, evaluator, encoder, 2*n)
	// result1 := transpose.Transpose(res1, params, evaluator, encoder, 2*n)
	// _ = result1

	// elapse = time.Since(starttime)
	// fmt.Println("total time : ", elapse)

	// fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")
	// fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")

	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, 1)
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	cts := make([]*rlwe.Ciphertext, 2*n)
	for i := range cts {
		cts[i] = ct.CopyNew()
	}
	result0, result1 := matmult.C2S_OnceMul(cts, params, evaluator, encoder, mat0, mat1, mat2, mat3, scale)
	util.DebugCTS(result0, params, encoder, decryptor)
	resvalue := make([]complex128, n)
	for i := range result0 {
		result0[i].IsBatched = true
		dept := decryptor.DecryptNew(result0[i])
		encoder.Decode(dept, resvalue)

		fmt.Println(resvalue)
	}
	fmt.Println()
	fmt.Println()
	fmt.Println()
	for i := range result1 {
		result1[i].IsBatched = true
		dept := decryptor.DecryptNew(result1[i])
		encoder.Decode(dept, resvalue)

		fmt.Println(resvalue)
	}

}

func Test_S2C(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            5,
		LogQ:            []int{50, 50, 50},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}
	//parameter init
	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}

	fmt.Println("ckks parameter init end")

	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	n := 1 << params.LogMaxSlots()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, 2*n)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	starttime := time.Now()
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
		}()
	}
	wg.Wait()
	elapse := time.Since(starttime)
	fmt.Println(elapse)
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	// btpevk, _, _ := btpParams.GenEvaluationKeys(sk)

	// btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	// if err != nil {
	// 	panic(err)
	// }
	fmt.Println("generate Evaluator end")
	SF, _ := matmult.GenSFMat(params)
	scale := float64(1 << 40)
	mat0, mat1, mat2, mat3 := matmult.GenS2CMat(SF, scale, params)

	_, _, _, _ = encoder, encryptor, decryptor, evaluator

	value := make([]float64, n)
	for i := range value {
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = true

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	cts := make([]*rlwe.Ciphertext, 2*n)
	for i := range cts {
		cts[i] = ct.CopyNew()
	}
	cts2 := make([]*rlwe.Ciphertext, 2*n)
	for i := range cts {
		cts2[i] = ct.CopyNew()
	}
	starttime = time.Now()
	result := matmult.S2C_OnceMul(cts, cts2, params, evaluator, encoder, mat0, mat1, mat2, mat3, scale)
	elapse = time.Since(starttime)
	fmt.Println("s2c time : ", elapse)

	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")

	util.DebugCTS(result, params, encoder, decryptor)
	resvalue := make([]complex128, 2*n)
	for i := range result {
		result[i].IsBatched = false
		dept := decryptor.DecryptNew(result[i])
		encoder.Decode(dept, resvalue)

		fmt.Println(resvalue)
	}
}

func Test_Boot(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(1) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{48, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}
	//parameter init
	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}

	//====================================
	//=== 2) BOOTSTRAPPING PARAMETERS ===
	//====================================

	// CoeffsToSlots parameters (homomorphic encoding)
	CoeffsToSlotsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicEncode,
		Format:       hefloat.RepackImagAsReal, // Returns the real and imaginary part into separate ciphertexts
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel(),
		Levels:       []int{1, 1, 1}, //qiCoeffsToSlots
		LogBSGSRatio: 0,
		BitReversed:  false,
	}

	// Parameters of the homomorphic modular reduction x mod 1
	Mod1ParametersLiteral := hefloat.Mod1ParametersLiteral{
		LevelStart:      params.MaxLevel() - 3,
		LogScale:        48,                  // Matches qiEvalMod
		Mod1Type:        hefloat.CosDiscrete, // Multi-interval Chebyshev interpolation
		Mod1Degree:      24,                  // Depth 5
		DoubleAngle:     3,                   // Depth 3
		K:               8,                   // With EphemeralSecretWeight = 32 and 2^{15} slots, ensures < 2^{-138.7} failure probability
		LogMessageRatio: 8,                   // q/|m| = 2^10
		Mod1InvDegree:   0,                   // Depth 0
	}

	// SlotsToCoeffs parameters (homomorphic decoding)
	SlotsToCoeffsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicDecode,
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel() - 11,
		Levels:       []int{1, 1, 1}, // qiSlotsToCoeffs
		LogBSGSRatio: 0,
		BitReversed:  false,
	}

	// Custom bootstrapping.Parameters.
	// All fields are public and can be manually instantiated.
	btpParams := bootstrapping.Parameters{
		ResidualParameters:      params,
		BootstrappingParameters: params,
		SlotsToCoeffsParameters: SlotsToCoeffsParameters,
		Mod1ParametersLiteral:   Mod1ParametersLiteral,
		CoeffsToSlotsParameters: CoeffsToSlotsParameters,
		EphemeralSecretWeight:   32, // > 128bit secure for LogN=16 and LogQP = 115.
		CircuitOrder:            bootstrapping.Custom,
	}

	fmt.Println("ckks parameter init end")
	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	n := 1 << params.LogMaxSlots()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, 2)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
		}()
	}
	wg.Wait()
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)
	_ = decryptor
	_ = evaluator
	btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	if err != nil {
		panic(err)
	}
	fmt.Println("generate Evaluator end")
	printMemUsage()
	fmt.Println(btp.CoeffsToSlotsParameters.BitReversed)

	// roots := ckks.GetRootsBigComplex(n<<2, params.EncodingPrecision())
	// roots_complex := make([]complex128, 4*n)

	// for i := range roots_complex {
	// 	roots_complex[i] = roots[i].Complex128()
	// }
	// fmt.Println()

	// pow5 := make([]int, (n<<1)+1)
	// pow5[0] = 1
	// for i := 1; i < (n<<1)+1; i++ {
	// 	pow5[i] = pow5[i-1] * 5
	// 	pow5[i] &= (n << 2) - 1
	// }

	// SF := make([][]complex128, n)

	// for i := range SF {
	// 	SF[i] = make([]complex128, n)
	// 	for j := range SF[i] {
	// 		idx := (pow5[i] * j) & ((n << 2) - 1)
	// 		SF[i][j] = roots_complex[idx]
	// 	}
	// }

	// SFI := make([][]complex128, n)
	// for i := range SFI {
	// 	SFI[i] = make([]complex128, n)
	// }
	// for i := range SFI {
	// 	for j := range SFI[i] {
	// 		idx := (pow5[i] * j) & ((n << 2) - 1)
	// 		SFI[j][i] = cmplx.Conj(roots_complex[idx]) / complex((float64(n)), 0)
	// 	}
	// }
	// scale := float64(1 << 40)
	// mat0, mat1, mat2, mat3 := matmult.GenC2SMat(SFI, scale, params)
	// mat0_, mat1_, mat2_, mat3_ := matmult.GenS2CMat(SF, scale, params)

	value := make([]float64, n)
	for i := range value {
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = true

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	// cts := make([]*rlwe.Ciphertext, 2*n)
	// for i := range cts {
	// 	cts[i] = ct.CopyNew()
	// }

	// fmt.Println("gen cts & mats")
	// printMemUsage()

	// starttime_ := time.Now()
	// cts1_, cts2_ := matmult.C2S_OnceMul(cts, params, evaluator, encoder, mat0, mat1, mat2, mat3, scale)
	// elapse_ := time.Since(starttime_)
	// fmt.Println("cts time : ", elapse_)
	// fmt.Println("ckeck cts")
	// printMemUsage()

	// starttime_ = time.Now()
	// for i := range cts1_ {
	// 	cts1_[i], _ = btp.EvalMod(cts1_[i])
	// }
	// for i := range cts2_ {
	// 	cts2_[i], _ = btp.EvalMod(cts2_[i])
	// }
	// elapse_ = time.Since(starttime_)
	// fmt.Println("eval time : ", elapse_)

	// starttime_ = time.Now()
	// res := matmult.S2C_OnceMul(cts1_, cts2_, params, evaluator, encoder, mat0_, mat1_, mat2_, mat3_, scale)
	// elapse_ = time.Since(starttime_)
	// fmt.Println("stc time : ", elapse_)
	// fmt.Println("ckeck stc")
	// printMemUsage()
	// _ = res

	// res = nil
	// cts1_ = nil
	// cts2_ = nil
	// mat0, mat1, mat2, mat3, mat0_, mat1_, mat2_, mat3_ = nil, nil, nil, nil, nil, nil, nil, nil
	// printMemUsage()

	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")
	starttime_ := time.Now()
	ct_, ct2_, _ := btp.DFTEvaluator.CoeffsToSlotsNew(ct, btp.C2SDFTMatrix)
	elapse_ := time.Since(starttime_)
	fmt.Println("cts time(origin) : ", (elapse_ * (1 << 16)).Seconds())
	starttime_ = time.Now()
	ct_, _ = btp.EvalMod(ct_)
	ct2_, _ = btp.EvalMod(ct2_)
	elapse_ = time.Since(starttime_)
	fmt.Println("eval time(origin) : ", (elapse_ * (1 << 16)).Seconds())

	// evaluator.DropLevel(ct, 2)
	starttime_ = time.Now()
	ct, _ = btp.DFTEvaluator.SlotsToCoeffsNew(ct_, ct2_, btp.S2CDFTMatrix)
	elapse_ = time.Since(starttime_)
	fmt.Println("stc time(origin) : ", (elapse_ * (1 << 16)).Seconds())
}

func Test_S2CModEval_Origin(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            10,
		LogQ:            []int{48, 40, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52, 52, 52},
		LogDefaultScale: 40,
	}
	//parameter init
	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}

	fmt.Println("ckks parameter init end")

	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	N := 1 << params.LogN()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, N)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	starttime := time.Now()
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		i := i
		gal := galEls[i]
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(gal, sk)
		}()
	}
	wg.Wait()
	elapse := time.Since(starttime)
	fmt.Println(elapse)
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	fmt.Println("generate Evaluator end")

	CoeffsToSlotsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicEncode,
		Format:       hefloat.RepackImagAsReal, // Returns the real and imaginary part into separate ciphertexts
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel(),
		Levels:       []int{1, 1}, //qiCoeffsToSlots
		LogBSGSRatio: 0,
	}

	// Parameters of the homomorphic modular reduction x mod 1
	Mod1ParametersLiteral := hefloat.Mod1ParametersLiteral{
		LevelStart:      params.MaxLevel() - 2,
		LogScale:        48,                  // Matches qiEvalMod
		Mod1Type:        hefloat.CosDiscrete, // Multi-interval Chebyshev interpolation
		Mod1Degree:      30,                  // Depth 5
		DoubleAngle:     3,                   // Depth 3
		K:               8,                   // With EphemeralSecretWeight = 32 and 2^{15} slots, ensures < 2^{-138.7} failure probability
		LogMessageRatio: 8,                   // q/|m| = 2^10
		Mod1InvDegree:   0,                   // Depth 0
	}

	// SlotsToCoeffs parameters (homomorphic decoding)
	SlotsToCoeffsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicDecode,
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel() - 10,
		Levels:       []int{1, 1}, // qiSlotsToCoeffs
		LogBSGSRatio: 0,
	}

	// Custom bootstrapping.Parameters.
	// All fields are public and can be manually instantiated.
	btpParams := bootstrapping.Parameters{
		ResidualParameters:      params,
		BootstrappingParameters: params,
		SlotsToCoeffsParameters: SlotsToCoeffsParameters,
		Mod1ParametersLiteral:   Mod1ParametersLiteral,
		CoeffsToSlotsParameters: CoeffsToSlotsParameters,
		EphemeralSecretWeight:   32, // > 128bit secure for LogN=16 and LogQP = 115.
		CircuitOrder:            bootstrapping.Custom,
	}
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)
	_ = decryptor
	_ = evaluator
	btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	if err != nil {
		panic(err)
	}
	_ = btp
	fmt.Println("generate btp Evaluator end")

	fmt.Println("ckks log degree : ", params.LogN())

	// mat0si := make([][][]float64, len(SFI))
	// ringQ := params.RingQ()
	value := make([]float64, N)
	for j := range value {
		value[j] = 0.0001 * float64(j)
	}
	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	cts := make([]*rlwe.Ciphertext, N)
	for i := range cts {
		encoder.Encode(value, pt)
		cts[i], _ = encryptor.EncryptNew(pt)
		// ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
		// ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
	}

	for i := range cts {
		cts[i], _ = btp.ModUp(cts[i])
		cts[i], _, err = btp.CoeffsToSlots(cts[i])
		// if err != nil {
		// 	panic(err)
		// }
		cts[i], _ = btp.EvalMod(cts[i])
		// cts[i], _ = btp.SlotsToCoeffs(eval, nil)
	}

	maxerr := 0.0
	idx := 0
	resvalue := make([]complex128, N/2)
	sc, _ := btp.CoeffsToSlotsParameters.Scaling.Float64()
	_ = sc
	for i := range cts {
		cts[i].IsBatched = true
		dept := decryptor.DecryptNew(cts[i])
		encoder.Decode(dept, resvalue)
		if i < 100 {
			fmt.Println(resvalue[:100])
		}

		for j := range resvalue {
			val := math.Abs(value[bitReverse(j, 9)] - real(resvalue[j]))
			if val > maxerr {
				idx = j
				maxerr = val
			}
		}
	}

	fmt.Println(-math.Log2(maxerr))
	fmt.Println(maxerr)
	fmt.Println(idx, " ", value[bitReverse(idx, 9)], " ", resvalue[idx])
}
func bitReverse(i, m int) int {
	rev := 0
	for j := 0; j < m; j++ {
		rev = (rev << 1) | (i & 1)
		i >>= 1
	}
	return rev
}

func Test_C2SModEval(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            10,
		LogQ:            []int{48, 40, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52, 52, 52},
		LogDefaultScale: 40,
	}
	//parameter init
	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}

	fmt.Println("ckks parameter init end")

	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	N := 1 << params.LogN()
	n := N / 2
	sparseN := N

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, N)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	starttime := time.Now()
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		i := i
		gal := galEls[i]
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(gal, sk)
		}()
	}
	wg.Wait()
	elapse := time.Since(starttime)
	fmt.Println(elapse)
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	fmt.Println("generate Evaluator end")

	CoeffsToSlotsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicEncode,
		Format:       hefloat.RepackImagAsReal, // Returns the real and imaginary part into separate ciphertexts
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel(),
		Levels:       []int{1, 1}, //qiCoeffsToSlots
		LogBSGSRatio: 0,
		BitReversed:  false,
	}

	// Parameters of the homomorphic modular reduction x mod 1
	Mod1ParametersLiteral := hefloat.Mod1ParametersLiteral{
		LevelStart:      params.MaxLevel() - 2,
		LogScale:        48,                  // Matches qiEvalMod
		Mod1Type:        hefloat.CosDiscrete, // Multi-interval Chebyshev interpolation
		Mod1Degree:      30,                  // Depth 6
		DoubleAngle:     3,                   // Depth 3
		K:               8,                   // With EphemeralSecretWeight = 32 and 2^{15} slots, ensures < 2^{-138.7} failure probability
		LogMessageRatio: 8,                   // q/|m| = 2^10
		Mod1InvDegree:   0,                   // Depth 0
	}

	// SlotsToCoeffs parameters (homomorphic decoding)
	SlotsToCoeffsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicDecode,
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel() - 10,
		Levels:       []int{1, 1}, // qiSlotsToCoeffs
		LogBSGSRatio: 0,
		BitReversed:  false,
	}

	// Custom bootstrapping.Parameters.
	// All fields are public and can be manually instantiated.
	btpParams := bootstrapping.Parameters{
		ResidualParameters:      params,
		BootstrappingParameters: params,
		SlotsToCoeffsParameters: SlotsToCoeffsParameters,
		Mod1ParametersLiteral:   Mod1ParametersLiteral,
		CoeffsToSlotsParameters: CoeffsToSlotsParameters,
		EphemeralSecretWeight:   32, // > 128bit secure for LogN=16 and LogQP = 115.
		CircuitOrder:            bootstrapping.Custom,
	}
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)
	_ = decryptor
	_ = evaluator
	btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	if err != nil {
		panic(err)
	}
	_ = btp
	fmt.Println("generate btp Evaluator end")

	fmt.Println("ckks log degree : ", params.LogN())
	fmt.Println(params.EncodingPrecision())
	CL_arr := []int{5, 4}
	_, SFI := matmult.GenSFMat_CL(params, CL_arr, CL_arr)
	scaling_ := bignum.Pow(btp.CoeffsToSlotsParameters.Scaling, new(big.Float).Quo(new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(1), new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(2)))
	for i := range SFI {
		for j := range SFI[i] {
			for k := range SFI[i][j] {
				val := new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(real(SFI[i][j][k]))
				val = val.Mul(val, scaling_)
				val2 := new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(imag(SFI[i][j][k]))
				val2 = val2.Mul(val2, scaling_)
				v1, _ := val.Float64()
				v2, _ := val2.Float64()
				SFI[i][j][k] = complex(v1, v2)
			}
		}
	}
	C2Scaling, _ := btp.CoeffsToSlotsParameters.Scaling.Float64()
	// C2Scaling = 1.0
	fmt.Println(C2Scaling)
	scale := float64(1 << 40)
	mat0 := make([][][]float64, len(SFI))
	mat0i := make([][][]float64, len(SFI))
	// mat0si := make([][][]float64, len(SFI))
	ringQ := params.RingQ()
	value := make([]float64, N)
	for j := range value {
		value[j] = 0.0001 * float64(j)
	}
	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	cts := make([]*rlwe.Ciphertext, N)
	for i := range cts {
		encoder.Encode(value, pt)
		cts[i], _ = encryptor.EncryptNew(pt)
		// cts[i], _ = btp.ModUp(cts[i])
		ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
		// cts[i], _ = ModUp(btp, cts[i])
	}
	fmt.Println(cts[0].LogScale())

	P := []uint64{3422539, 3370361, 3231143, 3545881, 3577031, 3832931, 4064197, 3617099, 3651497, 3711319, 3439693, 3502001, 3555509, 3552013, 4031179, 4115407, 3167453, 3365393, 3291143, 3204973, 4182419, 3495781, 3315883, 3403391, 3529153, 3390899, 3453773, 3705469, 3180337, 4091993, 3503221, 3598949, 3822277, 3277853, 3547249, 3278053, 3696257, 3849409, 3725257, 3239449, 3730721, 3393619, 3361363, 3732997, 3661573, 3158971, 3516031, 3737039, 3882649, 3614969, 3518491, 3169759, 3326417, 4165333, 3853097, 3845357, 3721603, 3494831, 3255467, 3442987, 3381641, 4188433, 3960053, 3825473, 3269713, 3373781, 3403843, 4177609, 3265337, 3382231, 3342137, 3330179, 3272629, 3725357, 3667453, 3960049, 3435323, 3664249, 3632423, 3515269, 3784733, 3377657, 4064143, 3702119, 3835367, 3564937, 3507397, 3345877, 4169129, 3206783, 3397769, 4145293, 3773477, 3229319, 3161617, 3517427, 3456743, 3687163, 3389423, 3553541}
	PLevel := 35
	P = P[:PLevel+1]
	ringP, _ := ring.NewRing(N, P)
	be := matmult.NewBasisExtender(ringQ, ringP, []matmult.Key{{params.MaxLevel(), PLevel}, {0, PLevel}}, []matmult.Key{{PLevel, params.MaxLevel()}})

	fmt.Println("mat gen start")
	inter_it := n
	for l := range SFI {
		inter := inter_it >> CL_arr[l]
		llen := (1 << CL_arr[l])
		mat0[l] = make([][]float64, n/llen)
		mat0i[l] = make([][]float64, n/llen)

		// mat0si[l] = make([][]float64, n/llen)
		for t := range n / llen {
			mat0[l][t] = make([]float64, len(P)*llen*llen)
			mat0i[l][t] = make([]float64, len(P)*llen*llen)
			stpoint := inter_it*int(t/inter) + (t % inter)
			for q := range len(P) {
				for i := range llen {
					for j := range llen {
						idx := q*llen*llen + i*llen + j
						if real(SFI[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
							mat0[l][t][idx] = float64(int64(real(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q]))
						} else {
							mat0[l][t][idx] = float64(int64(P[q]) - (int64(-real(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q])))
						}
						if imag(SFI[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
							mat0i[l][t][idx] = float64(int64(imag(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q]))
						} else {
							mat0i[l][t][idx] = float64(int64(P[q]) - (int64(-imag(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q])))
						}
					}
				}
			}
		}

		inter_it = inter
	}
	fmt.Println("mat gen end")

	inputPolys := make([][]ring.Poly, 2)
	for i := range inputPolys {
		inputPolys[i] = make([]ring.Poly, sparseN)
		for j := range inputPolys[i] {
			inputPolys[i][j] = ringP.NewPoly()
		}
	}

	inputPolysC := make([][]ring.Poly, 2)
	for i := range inputPolysC {
		inputPolysC[i] = make([]ring.Poly, sparseN)
		for j := range inputPolysC[i] {
			inputPolysC[i][j] = ringP.NewPoly()
		}
	}
	result := make([]*rlwe.Ciphertext, N)
	result2 := make([]*rlwe.Ciphertext, N)
	pttemp := hefloat.NewPlaintext(params, params.MaxLevel())
	pttemp.IsBatched = false
	encoder.Encode(value, pttemp)
	fmt.Println("start ct res pre allocate")
	for i := range N {
		cttemp, _ := encryptor.EncryptNew(pttemp)
		result[i] = cttemp.CopyNew()
		result2[i] = cttemp.CopyNew()
	}
	fmt.Println("prealloc end")
	fmt.Println(result[0].LogScale())

	ppmmbuffer1 := make([]float64, len(P)*N*N)
	ppmmbuffer2 := make([]float64, len(P)*N*N)

	resPolys00 := make([][]ring.Poly, 2)
	resPolys00i := make([][]ring.Poly, 2)
	resPolys01 := make([][]ring.Poly, 2)
	resPolys01i := make([][]ring.Poly, 2)
	resPolys10 := make([][]ring.Poly, 2)
	resPolys10i := make([][]ring.Poly, 2)
	resPolys11 := make([][]ring.Poly, 2)
	resPolys11i := make([][]ring.Poly, 2)
	for i := range 2 {
		resPolys00[i] = make([]ring.Poly, N)
		resPolys00i[i] = make([]ring.Poly, N)
		resPolys01[i] = make([]ring.Poly, N)
		resPolys01i[i] = make([]ring.Poly, N)
		resPolys10[i] = make([]ring.Poly, N)
		resPolys10i[i] = make([]ring.Poly, N)
		resPolys11[i] = make([]ring.Poly, N)
		resPolys11i[i] = make([]ring.Poly, N)
		for j := range N {
			resPolys00[i][j] = ringP.NewPoly()
			resPolys00i[i][j] = ringP.NewPoly()
			resPolys01[i][j] = ringP.NewPoly()
			resPolys01i[i][j] = ringP.NewPoly()
			resPolys10[i][j] = ringP.NewPoly()
			resPolys10i[i][j] = ringP.NewPoly()
			resPolys11[i][j] = ringP.NewPoly()
			resPolys11i[i][j] = ringP.NewPoly()
		}
	}
	work := make([]*rlwe.Ciphertext, N)
	for i := range work {
		work[i] = encryptor.EncryptZeroNew(params.MaxLevel())
	}
	aux := make([]*rlwe.Ciphertext, N)
	for i := range aux {
		aux[i] = encryptor.EncryptZeroNew(params.MaxLevel())
	}

	fmt.Println("start c2s")
	starttime = time.Now()
	printMemUsage()

	// transpose.Transpose3(cts, params, evaluator, ringQ.AtLevel(cts[0].Level()), N, N, work, aux, cts)

	fmt.Println("mod switch start")

	for i := range sparseN {
		for d := range 2 {
			be.ModSwitchQtoP_Old(cts[i].Level(), PLevel, cts[i].Value[d], inputPolys[d][i])
			// be.ModUpQtoP(PLevel, cts[i].Value[d], inputPolys[d][i])
		}
		if i < n {
			ringQ.AtLevel(cts[0].Level()).Neg(cts[i+n].Value[0], work[0].Value[0])
			ringQ.AtLevel(cts[0].Level()).Neg(cts[i+n].Value[1], work[0].Value[1])
		} else {
			work[0] = cts[i-n]
		}
		for d := range 2 {
			be.ModSwitchQtoP_Old(cts[i].Level(), PLevel, work[0].Value[d], inputPolysC[d][i])
			// be.ModUpQtoP(PLevel, work[0].Value[d], inputPolysC[d][i])
		}
	}

	fmt.Println(inputPolysC[0][0].Coeffs[0])
	fmt.Println("mod switch end")
	fmt.Println("ppmm start")
	inter_it = n
	for l := range len(SFI) {
		inter := inter_it >> CL_arr[l]
		llen := (1 << CL_arr[l])
		if l == 0 {
			for t := range n / llen {
				//00 ~ 01
				// fmt.Println(inputPolys[0][0].Coeffs[0][:100])
				stpoint := (t % inter) + inter_it*int(t/inter)

				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00i, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01i, ppmmbuffer1, ppmmbuffer2)

				//10~11
				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10i, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11i, ppmmbuffer1, ppmmbuffer2)

			}

		} else if l == len(SFI)-1 {
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(resPolys00, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys00i, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00i, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys01, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys01i, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01i, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.SubManyRing(ringP, resPolys00, resPolys00i, resPolys00)
			matmult.AddManyRing(ringP, resPolys01, resPolys01i, resPolys01i)

			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(resPolys10, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys10i, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10i, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys11, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys11i, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11i, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.SubManyRing(ringP, resPolys10i, resPolys10, resPolys10)
			matmult.AddManyRing(ringP, resPolys11, resPolys11i, resPolys11i)

		}
		inter_it = inter
	}
	matmult.AddManyRing(ringP, resPolys00, resPolys01i, resPolys00)
	matmult.AddManyRing(ringP, resPolys10, resPolys11i, resPolys10)

	rev := matmult.BitReversePermutationMatrix(n)
	matrev := make([]float64, len(P)*N*N)
	for p := range len(P) {
		for i := range N {
			for j := range N {
				if (i < n && j < n) || (i >= n && j >= n) {
					matrev[p*N*N+i*N+j] = (real(rev[i%n][j%n]))
				}
			}
		}
	}

	matmult.PPMM_Blas_CRT_Inplace(resPolys00, matrev, N, N, N, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
	matmult.PPMM_Blas_CRT_Inplace(resPolys10, matrev, N, N, N, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
	fmt.Println(resPolys00[0][0])
	for i := range N {
		for idx := range 2 {
			be.ModSwitchPtoQ_Old(PLevel, params.MaxLevel(), resPolys00[idx][i], result[i].Value[idx])
			be.ModSwitchPtoQ_Old(PLevel, params.MaxLevel(), resPolys10[idx][i], result2[i].Value[idx])
			// ringQ.Mul
		}
		if i == 0 {
			fmt.Println(result[0].Value[0].Coeffs)
		}

		q := rlwe.NewScale(params.Q()[result[i].Level()])
		util.Mul_ScaleExact(evaluator, result[i], 1/(scale), result[i], q)
		util.Rescale_NonNTT(evaluator, result[i], result[i])
		util.Mul_ScaleExact(evaluator, result2[i], 1.0/(scale), result2[i], q)
		util.Rescale_NonNTT(evaluator, result2[i], result2[i])

		q = rlwe.NewScale(params.Q()[result[i].Level()])
		util.Mul_ScaleExact(evaluator, result[i], 1/(scale), result[i], q)
		util.Rescale_NonNTT(evaluator, result[i], result[i])
		util.Mul_ScaleExact(evaluator, result2[i], 1/(scale), result2[i], q)
		util.Rescale_NonNTT(evaluator, result2[i], result2[i])

		// scaling, _ := btp.CoeffsToSlotsParameters.Scaling.Float64()
		// q = rlwe.NewScale(params.Q()[result[i].Level()])
		// util.Mul_ScaleExact(evaluator, result[i], scaling, result[i], q)
		// util.Rescale_NonNTT(evaluator, result[i], result[i])
	}
	fmt.Println(result[0].Level())

	transpose.Transpose3(result, params, evaluator, ringQ.AtLevel(result[0].Level()), N, sparseN, work, aux, result)
	transpose.Transpose3(result2, params, evaluator, ringQ.AtLevel(result2[0].Level()), N, sparseN, work, aux, result2)
	elapse = time.Since(starttime)
	fmt.Println(elapse)

	for i := range result {
		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[0], result[i].Value[0])
		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[1], result[i].Value[1])
		// result[i], err = btp.EvalMod(result[i])
		// if err != nil {
		// 	panic(err)
		// }

		// ringQ.AtLevel(result2[i].Level()).NTT(result2[i].Value[0], result2[i].Value[0])
		// ringQ.AtLevel(result2[i].Level()).NTT(result2[i].Value[1], result2[i].Value[1])
	}
	fmt.Println(result[0].Value[0].Coeffs)
	resvalue := make([]complex128, n)
	maxerr := 0.0
	idx := 0
	C2Scaling, _ = btp.CoeffsToSlotsParameters.Scaling.Float64()
	for i := range result {
		result[i].IsBatched = true
		dept := decryptor.DecryptNew(result[i])
		encoder.Decode(dept, resvalue)

		if i < 100 {
			fmt.Println(resvalue[:100])
		}

		for j := range resvalue {
			val := math.Abs(0.0001*float64(i) - real(resvalue[j])/C2Scaling)
			if val > maxerr {
				idx = j
				maxerr = val
			}
		}
	}

	fmt.Println(-math.Log2(maxerr))
	fmt.Println(maxerr)
	fmt.Println(idx, " ", 0.0001*float64(idx), " ", resvalue[idx])
}

func Test_CheckC2SPrec(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            10,
		LogQ:            []int{48, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52, 52, 52},
		LogDefaultScale: 40,
	}
	//parameter init
	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}

	fmt.Println("ckks parameter init end")

	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	N := 1 << params.LogN()
	n := N / 2
	sparseN := N

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, N)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	starttime := time.Now()
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		i := i
		gal := galEls[i]
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(gal, sk)
		}()
	}
	wg.Wait()
	elapse := time.Since(starttime)
	fmt.Println(elapse)
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	fmt.Println("generate Evaluator end")

	CoeffsToSlotsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicEncode,
		Format:       hefloat.RepackImagAsReal, // Returns the real and imaginary part into separate ciphertexts
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel(),
		Levels:       []int{1, 1}, //qiCoeffsToSlots
		LogBSGSRatio: 0,
		BitReversed:  false,
	}

	// Parameters of the homomorphic modular reduction x mod 1
	Mod1ParametersLiteral := hefloat.Mod1ParametersLiteral{
		LevelStart:      params.MaxLevel() - 2,
		LogScale:        48,                  // Matches qiEvalMod
		Mod1Type:        hefloat.CosDiscrete, // Multi-interval Chebyshev interpolation
		Mod1Degree:      63,                  // Depth 6
		DoubleAngle:     3,                   // Depth 3
		K:               31,                  // With EphemeralSecretWeight = 32 and 2^{15} slots, ensures < 2^{-138.7} failure probability
		LogMessageRatio: 8,                   // q/|m| = 2^10
		Mod1InvDegree:   0,                   // Depth 0
	}

	// SlotsToCoeffs parameters (homomorphic decoding)
	SlotsToCoeffsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicDecode,
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel() - 11,
		Levels:       []int{1, 1}, // qiSlotsToCoeffs
		LogBSGSRatio: 0,
		BitReversed:  true,
	}

	// Custom bootstrapping.Parameters.
	// All fields are public and can be manually instantiated.
	btpParams := bootstrapping.Parameters{
		ResidualParameters:      params,
		BootstrappingParameters: params,
		SlotsToCoeffsParameters: SlotsToCoeffsParameters,
		Mod1ParametersLiteral:   Mod1ParametersLiteral,
		CoeffsToSlotsParameters: CoeffsToSlotsParameters,
		EphemeralSecretWeight:   0, // > 128bit secure for LogN=16 and LogQP = 115.
		CircuitOrder:            bootstrapping.Custom,
	}
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)
	_ = decryptor
	_ = evaluator
	btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	if err != nil {
		panic(err)
	}
	_ = btp
	fmt.Println("generate btp Evaluator end")

	fmt.Println("ckks log degree : ", params.LogN())
	CL_arr := []int{4, 5}
	_, SFI := matmult.GenSFMat_CL(params, CL_arr, CL_arr)
	C2Scaling, _ := btp.CoeffsToSlotsParameters.Scaling.Float64()
	C2Scaling = 1
	scaling_ := big.NewFloat(C2Scaling)
	scaling_.Quo(scaling_, new(big.Float).SetFloat64(float64(N)))
	scaling_ = bignum.Pow(scaling_, new(big.Float).Quo(new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(1), new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(2)))
	fmt.Println(scaling_)
	for i := range SFI {
		for j := range SFI[i] {
			for k := range SFI[i][j] {
				val := new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(real(SFI[i][j][k]))
				val = val.Mul(val, scaling_)
				val2 := new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(imag(SFI[i][j][k]))
				val2 = val2.Mul(val2, scaling_)
				v1, _ := val.Float64()
				v2, _ := val2.Float64()
				SFI[i][j][k] = complex(v1, v2)
			}
		}
	}

	// C2Scaling = 1.0
	fmt.Println(C2Scaling)
	scale := float64(1 << 40)
	mat0 := make([][][]float64, len(SFI))
	mat0i := make([][][]float64, len(SFI))
	// mat0si := make([][][]float64, len(SFI))
	ringQ := params.RingQ()
	value := make([]float64, N)
	for j := range value {
		value[j] = sampling.RandFloat64(-1, 1)
	}
	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	cts := make([]*rlwe.Ciphertext, N)
	for i := range cts {
		encoder.Encode(value, pt)
		cts[i], _ = encryptor.EncryptNew(pt)
		// cts[i], _ = btp.ModUp(cts[i])
		ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
		// cts[i], _ = ModUp(btp, cts[i])
	}
	fmt.Println(cts[0].LogScale())

	P := []uint64{3422539, 3370361, 3231143, 3545881, 3577031, 3832931, 4064197, 3617099, 3651497, 3711319, 3439693, 3502001, 3555509, 3552013, 4031179, 4115407, 3167453, 3365393, 3291143, 3204973, 4182419, 3495781, 3315883, 3403391, 3529153, 3390899, 3453773, 3705469, 3180337, 4091993, 3503221, 3598949, 3822277, 3277853, 3547249, 3278053, 3696257, 3849409, 3725257, 3239449, 3730721, 3393619, 3361363, 3732997, 3661573, 3158971, 3516031, 3737039, 3882649, 3614969, 3518491, 3169759, 3326417, 4165333, 3853097, 3845357, 3721603, 3494831, 3255467, 3442987, 3381641, 4188433, 3960053, 3825473, 3269713, 3373781, 3403843, 4177609, 3265337, 3382231, 3342137, 3330179, 3272629, 3725357, 3667453, 3960049, 3435323, 3664249, 3632423, 3515269, 3784733, 3377657, 4064143, 3702119, 3835367, 3564937, 3507397, 3345877, 4169129, 3206783, 3397769, 4145293, 3773477, 3229319, 3161617, 3517427, 3456743, 3687163, 3389423, 3553541}
	PLevel := 35
	P = P[:PLevel+1]
	ringP, _ := ring.NewRing(N, P)
	be := matmult.NewBasisExtender(ringQ, ringP, []matmult.Key{{params.MaxLevel(), PLevel}, {0, PLevel}}, []matmult.Key{{PLevel, params.MaxLevel()}})

	fmt.Println("mat gen start")
	inter_it := n
	for l := range SFI {
		inter := inter_it >> CL_arr[l]
		llen := (1 << CL_arr[l])
		mat0[l] = make([][]float64, n/llen)
		mat0i[l] = make([][]float64, n/llen)

		// mat0si[l] = make([][]float64, n/llen)
		for t := range n / llen {
			mat0[l][t] = make([]float64, len(P)*llen*llen)
			mat0i[l][t] = make([]float64, len(P)*llen*llen)
			stpoint := inter_it*int(t/inter) + (t % inter)
			for q := range len(P) {
				for i := range llen {
					for j := range llen {
						idx := q*llen*llen + i*llen + j
						if real(SFI[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
							mat0[l][t][idx] = float64(int64(real(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q]))
						} else {
							mat0[l][t][idx] = float64(int64(P[q]) - (int64(-real(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q])))
						}
						if imag(SFI[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
							mat0i[l][t][idx] = float64(int64(imag(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q]))
						} else {
							mat0i[l][t][idx] = float64(int64(P[q]) - (int64(-imag(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q])))
						}
					}
				}
			}
		}

		inter_it = inter
	}
	fmt.Println("mat gen end")

	inputPolys := make([][]ring.Poly, 2)
	for i := range inputPolys {
		inputPolys[i] = make([]ring.Poly, sparseN)
		for j := range inputPolys[i] {
			inputPolys[i][j] = ringP.NewPoly()
		}
	}

	inputPolysC := make([][]ring.Poly, 2)
	for i := range inputPolysC {
		inputPolysC[i] = make([]ring.Poly, sparseN)
		for j := range inputPolysC[i] {
			inputPolysC[i][j] = ringP.NewPoly()
		}
	}
	result := make([]*rlwe.Ciphertext, N)
	result2 := make([]*rlwe.Ciphertext, N)
	pttemp := hefloat.NewPlaintext(params, params.MaxLevel())
	pttemp.IsBatched = false
	encoder.Encode(value, pttemp)
	fmt.Println("start ct res pre allocate")
	for i := range N {
		cttemp, _ := encryptor.EncryptNew(pttemp)
		result[i] = cttemp.CopyNew()
		result2[i] = cttemp.CopyNew()
	}
	fmt.Println("prealloc end")
	fmt.Println(result[0].LogScale())

	ppmmbuffer1 := make([]float64, len(P)*N*N)
	ppmmbuffer2 := make([]float64, len(P)*N*N)

	resPolys00 := make([][]ring.Poly, 2)
	resPolys00i := make([][]ring.Poly, 2)
	resPolys01 := make([][]ring.Poly, 2)
	resPolys01i := make([][]ring.Poly, 2)
	resPolys10 := make([][]ring.Poly, 2)
	resPolys10i := make([][]ring.Poly, 2)
	resPolys11 := make([][]ring.Poly, 2)
	resPolys11i := make([][]ring.Poly, 2)
	for i := range 2 {
		resPolys00[i] = make([]ring.Poly, N)
		resPolys00i[i] = make([]ring.Poly, N)
		resPolys01[i] = make([]ring.Poly, N)
		resPolys01i[i] = make([]ring.Poly, N)
		resPolys10[i] = make([]ring.Poly, N)
		resPolys10i[i] = make([]ring.Poly, N)
		resPolys11[i] = make([]ring.Poly, N)
		resPolys11i[i] = make([]ring.Poly, N)
		for j := range N {
			resPolys00[i][j] = ringP.NewPoly()
			resPolys00i[i][j] = ringP.NewPoly()
			resPolys01[i][j] = ringP.NewPoly()
			resPolys01i[i][j] = ringP.NewPoly()
			resPolys10[i][j] = ringP.NewPoly()
			resPolys10i[i][j] = ringP.NewPoly()
			resPolys11[i][j] = ringP.NewPoly()
			resPolys11i[i][j] = ringP.NewPoly()
		}
	}
	work := make([]*rlwe.Ciphertext, N)
	for i := range work {
		work[i] = encryptor.EncryptZeroNew(params.MaxLevel())
	}
	aux := make([]*rlwe.Ciphertext, N)
	for i := range aux {
		aux[i] = encryptor.EncryptZeroNew(params.MaxLevel())
	}

	fmt.Println("start c2s")
	starttime = time.Now()
	printMemUsage()

	transpose.Transpose3(cts, params, evaluator, ringQ.AtLevel(cts[0].Level()), N, N, work, aux, cts)

	fmt.Println("mod switch start")

	for i := range sparseN {
		for d := range 2 {
			be.ModSwitchQtoP_Old(cts[i].Level(), PLevel, cts[i].Value[d], inputPolys[d][i])
			// be.ModUpQtoP(PLevel, cts[i].Value[d], inputPolys[d][i])
		}
		if i < n {
			ringQ.AtLevel(cts[0].Level()).Neg(cts[i+n].Value[0], work[0].Value[0])
			ringQ.AtLevel(cts[0].Level()).Neg(cts[i+n].Value[1], work[0].Value[1])
		} else {
			work[0] = cts[i-n]
		}
		for d := range 2 {
			be.ModSwitchQtoP_Old(cts[i].Level(), PLevel, work[0].Value[d], inputPolysC[d][i])
			// be.ModUpQtoP(PLevel, work[0].Value[d], inputPolysC[d][i])
		}
	}

	fmt.Println(inputPolysC[0][0].Coeffs[0])
	fmt.Println("mod switch end")
	fmt.Println("ppmm start")
	inter_it = n
	for l := range len(SFI) {
		inter := inter_it >> CL_arr[l]
		llen := (1 << CL_arr[l])
		if l == 0 {
			for t := range n / llen {
				//00 ~ 01
				// fmt.Println(inputPolys[0][0].Coeffs[0][:100])
				stpoint := (t % inter) + inter_it*int(t/inter)

				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00i, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01i, ppmmbuffer1, ppmmbuffer2)

				//10~11
				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10i, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11i, ppmmbuffer1, ppmmbuffer2)

			}

		} else if l == len(SFI)-1 {
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(resPolys00, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys00i, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00i, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys01, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys01i, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01i, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.SubManyRing(ringP, resPolys00, resPolys00i, resPolys00)
			matmult.AddManyRing(ringP, resPolys01, resPolys01i, resPolys01i)

			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(resPolys10, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys10i, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10i, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys11, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys11i, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11i, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.SubManyRing(ringP, resPolys10i, resPolys10, resPolys10)
			matmult.AddManyRing(ringP, resPolys11, resPolys11i, resPolys11i)

		}
		inter_it = inter
	}
	matmult.AddManyRing(ringP, resPolys00, resPolys01i, resPolys00)
	matmult.AddManyRing(ringP, resPolys10, resPolys11i, resPolys10)

	rev := matmult.BitReversePermutationMatrix(n)
	matrev := make([]float64, len(P)*N*N)
	for p := range len(P) {
		for i := range N {
			for j := range N {
				if (i < n && j < n) || (i >= n && j >= n) {
					matrev[p*N*N+i*N+j] = (real(rev[i%n][j%n]))
				}
			}
		}
	}

	matmult.PPMM_Blas_CRT_Inplace(resPolys00, matrev, N, N, N, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
	matmult.PPMM_Blas_CRT_Inplace(resPolys10, matrev, N, N, N, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
	fmt.Println(resPolys00[0][0])
	for i := range N {
		for idx := range 2 {
			be.ModSwitchPtoQ_Old(PLevel, params.MaxLevel(), resPolys00[idx][i], result[i].Value[idx])
			be.ModSwitchPtoQ_Old(PLevel, params.MaxLevel(), resPolys10[idx][i], result2[i].Value[idx])
			// ringQ.Mul
		}
		if i == 0 {
			fmt.Println(result[0].Value[0].Coeffs)
		}

		q := rlwe.NewScale(params.Q()[result[i].Level()])
		util.Mul_ScaleExact(evaluator, result[i], 1.0/(scale), result[i], q)
		util.Rescale_NonNTT(evaluator, result[i], result[i])
		util.Mul_ScaleExact(evaluator, result2[i], 1.0/(scale), result2[i], q)
		util.Rescale_NonNTT(evaluator, result2[i], result2[i])

		q = rlwe.NewScale(params.Q()[result[i].Level()])
		util.Mul_ScaleExact(evaluator, result[i], 1.0/(scale), result[i], q)
		util.Rescale_NonNTT(evaluator, result[i], result[i])
		util.Mul_ScaleExact(evaluator, result2[i], 1.0/(scale), result2[i], q)
		util.Rescale_NonNTT(evaluator, result2[i], result2[i])

		// scaling, _ := btp.CoeffsToSlotsParameters.Scaling.Float64()
		// q = rlwe.NewScale(params.Q()[result[i].Level()])
		// util.Mul_ScaleExact(evaluator, result[i], scaling, result[i], q)
		// util.Rescale_NonNTT(evaluator, result[i], result[i])
	}
	fmt.Println(result[0].Level())

	transpose.Transpose3(result, params, evaluator, ringQ.AtLevel(result[0].Level()), N, sparseN, work, aux, result)
	transpose.Transpose3(result2, params, evaluator, ringQ.AtLevel(result2[0].Level()), N, sparseN, work, aux, result2)

	elapse = time.Since(starttime)
	fmt.Println(elapse)

	for i := range result {

		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[0], result[i].Value[0])
		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[1], result[i].Value[1])

		conj, _ := evaluator.ConjugateNew(result[i])
		evaluator.Add(result[i], conj, result[i])
		// result[i], err = btp.EvalMod(result[i])
		if err != nil {
			panic(err)
		}

		// ringQ.AtLevel(result2[i].Level()).NTT(result2[i].Value[0], result2[i].Value[0])
		// ringQ.AtLevel(result2[i].Level()).NTT(result2[i].Value[1], result2[i].Value[1])
	}
	fmt.Println(result[0].Value[0].Coeffs)
	resvalue := make([]complex128, n)
	maxerr := 0.0
	idx := 0
	for i := range result {
		result[i].IsBatched = true
		dept := decryptor.DecryptNew(result[i])
		encoder.Decode(dept, resvalue)

		if i < 10 {
			fmt.Println(resvalue[:10])
		}
		for j := range resvalue {
			val := math.Abs(value[j] - real(resvalue[j]))
			if val > maxerr {
				idx = j
				maxerr = val
			}
		}
	}

	fmt.Println(-math.Log2(maxerr))
	fmt.Println(maxerr)
	fmt.Println(idx, " ", value[idx], " ", resvalue[idx])
	pt = hefloat.NewPlaintext(params, 0)
	pt.IsBatched = false

	cts = make([]*rlwe.Ciphertext, N)
	for i := range cts {
		encoder.Encode(value, pt)
		cts[i], _ = encryptor.EncryptNew(pt)
		// ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
		// ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
	}

	for i := range cts {
		cts[i], _ = btp.ModUp(cts[i])
		cts[i], _, err = btp.CoeffsToSlots(cts[i])
		cts[i], _ = btp.EvalMod(cts[i])
	}
	maxerr = 0.0
	idx = 0
	resvalue = make([]complex128, N/2)
	for i := range cts {
		cts[i].IsBatched = true
		dept := decryptor.DecryptNew(cts[i])
		encoder.Decode(dept, resvalue)
		if i < 10 {
			fmt.Println(resvalue[:10])
		}

		for j := range resvalue {
			val := math.Abs(value[bitReverse(j, 9)] - real(resvalue[j]))
			if val > maxerr {
				idx = j
				maxerr = val
			}
		}
	}

	fmt.Println(-math.Log2(maxerr))
	fmt.Println(maxerr)
	fmt.Println(idx, " ", value[bitReverse(idx, 9)], " ", resvalue[idx])

}

func Test_CheckC2SModEvalPrec(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            10,
		LogQ:            []int{48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52, 52, 52},
		LogDefaultScale: 40,
		Xs:              ring.Ternary{H: 192},
	}

	//parameter init
	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}

	fmt.Println("ckks parameter init end")

	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	N := 1 << params.LogN()
	n := N / 2
	sparseN := N

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, N)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	starttime := time.Now()
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		i := i
		gal := galEls[i]
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(gal, sk)
		}()
	}
	wg.Wait()
	elapse := time.Since(starttime)
	fmt.Println(elapse)
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	fmt.Println("generate Evaluator end")

	CoeffsToSlotsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicEncode,
		Format:       hefloat.RepackImagAsReal, // Returns the real and imaginary part into separate ciphertexts
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel(),
		Levels:       []int{1, 1}, //qiCoeffsToSlots
		LogBSGSRatio: 0,
		BitReversed:  false,
	}

	// Parameters of the homomorphic modular reduction x mod 1
	Mod1ParametersLiteral := hefloat.Mod1ParametersLiteral{
		LevelStart:      params.MaxLevel() - 3,
		LogScale:        48,                  // Matches qiEvalMod
		Mod1Type:        hefloat.CosDiscrete, // Multi-interval Chebyshev interpolation
		Mod1Degree:      63,                  // Depth 6
		DoubleAngle:     3,                   // Depth 3
		K:               31,                  // With EphemeralSecretWeight = 32 and 2^{15} slots, ensures < 2^{-138.7} failure probability
		LogMessageRatio: 8,                   // q/|m| = 2^10
		Mod1InvDegree:   0,                   // Depth 0
	}

	// SlotsToCoeffs parameters (homomorphic decoding)
	SlotsToCoeffsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicDecode,
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel() - 10,
		Levels:       []int{1, 1}, // qiSlotsToCoeffs
		LogBSGSRatio: 0,
		BitReversed:  false,
	}

	// Custom bootstrapping.Parameters.
	// All fields are public and can be manually instantiated.
	btpParams := bootstrapping.Parameters{
		ResidualParameters:      params,
		BootstrappingParameters: params,
		SlotsToCoeffsParameters: SlotsToCoeffsParameters,
		Mod1ParametersLiteral:   Mod1ParametersLiteral,
		CoeffsToSlotsParameters: CoeffsToSlotsParameters,
		EphemeralSecretWeight:   0, // > 128bit secure for LogN=16 and LogQP = 115.
		CircuitOrder:            bootstrapping.Custom,
	}
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)
	_ = decryptor
	_ = evaluator
	btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	if err != nil {
		panic(err)
	}
	_ = btp
	fmt.Println("generate btp Evaluator end")

	fmt.Println("ckks log degree : ", params.LogN())
	CL_arr := []int{4, 5}
	_, SFI := matmult.GenSFMat_CL(params, CL_arr, CL_arr)
	C2Scaling, _ := btp.CoeffsToSlotsParameters.Scaling.Float64()
	scaling_ := big.NewFloat(C2Scaling)
	scaling_.Quo(scaling_, new(big.Float).SetFloat64(float64(N)))
	scaling_ = bignum.Pow(scaling_, new(big.Float).Quo(new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(1), new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(2)))
	fmt.Println(scaling_)
	for i := range SFI {
		for j := range SFI[i] {
			for k := range SFI[i][j] {
				val := new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(real(SFI[i][j][k]))
				val = val.Mul(val, scaling_)
				val2 := new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(imag(SFI[i][j][k]))
				val2 = val2.Mul(val2, scaling_)
				v1, _ := val.Float64()
				v2, _ := val2.Float64()
				SFI[i][j][k] = complex(v1, v2)
			}
		}
	}

	// C2Scaling = 1.0
	fmt.Println(C2Scaling)
	scale := float64(1 << 40)
	mat0 := make([][][]float64, len(SFI))
	mat0i := make([][][]float64, len(SFI))
	// mat0si := make([][][]float64, len(SFI))
	ringQ := params.RingQ()
	value := make([]float64, N)
	for j := range value {
		value[j] = 0.0001 * float64(j)
	}
	pt := hefloat.NewPlaintext(params, 0)
	pt.IsBatched = false

	encoder.Encode(value, pt)
	cts := make([]*rlwe.Ciphertext, N)
	for i := range cts {
		encoder.Encode(value, pt)
		cts[i], _ = encryptor.EncryptNew(pt)
		// cts[i], _ = btp.ModUp(cts[i])

		ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
		// cts[i], _ = ModUp(btp, cts[i])
	}
	fmt.Println(cts[0].LogScale())

	P := []uint64{3422539, 3370361, 3231143, 3545881, 3577031, 3832931, 4064197, 3617099, 3651497, 3711319, 3439693, 3502001, 3555509, 3552013, 4031179, 4115407, 3167453, 3365393, 3291143, 3204973, 4182419, 3495781, 3315883, 3403391, 3529153, 3390899, 3453773, 3705469, 3180337, 4091993, 3503221, 3598949, 3822277, 3277853, 3547249, 3278053, 3696257, 3849409, 3725257, 3239449, 3730721, 3393619, 3361363, 3732997, 3661573, 3158971, 3516031, 3737039, 3882649, 3614969, 3518491, 3169759, 3326417, 4165333, 3853097, 3845357, 3721603, 3494831, 3255467, 3442987, 3381641, 4188433, 3960053, 3825473, 3269713, 3373781, 3403843, 4177609, 3265337, 3382231, 3342137, 3330179, 3272629, 3725357, 3667453, 3960049, 3435323, 3664249, 3632423, 3515269, 3784733, 3377657, 4064143, 3702119, 3835367, 3564937, 3507397, 3345877, 4169129, 3206783, 3397769, 4145293, 3773477, 3229319, 3161617, 3517427, 3456743, 3687163, 3389423, 3553541}
	PLevel := 32
	P = P[:PLevel+1]
	P = append(P, 43)
	PLevel += 1
	ringP, _ := ring.NewRing(N, P)
	be := matmult.NewBasisExtender(ringQ, ringP, []matmult.Key{{params.MaxLevel(), PLevel}, {0, PLevel}}, []matmult.Key{{PLevel, params.MaxLevel()}})

	fmt.Println("mat gen start")
	inter_it := n
	for l := range SFI {
		inter := inter_it >> CL_arr[l]
		llen := (1 << CL_arr[l])
		mat0[l] = make([][]float64, n/llen)
		mat0i[l] = make([][]float64, n/llen)

		// mat0si[l] = make([][]float64, n/llen)
		for t := range n / llen {
			mat0[l][t] = make([]float64, len(P)*llen*llen)
			mat0i[l][t] = make([]float64, len(P)*llen*llen)
			stpoint := inter_it*int(t/inter) + (t % inter)
			for q := range len(P) {
				for i := range llen {
					for j := range llen {
						idx := q*llen*llen + i*llen + j
						if real(SFI[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
							mat0[l][t][idx] = float64(int64(real(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q]))
						} else {
							mat0[l][t][idx] = float64(int64(P[q]) - (int64(-real(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q])))
						}
						if imag(SFI[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
							mat0i[l][t][idx] = float64(int64(imag(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q]))
						} else {
							mat0i[l][t][idx] = float64(int64(P[q]) - (int64(-imag(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q])))
						}
					}
				}
			}
		}

		inter_it = inter
	}
	fmt.Println("mat gen end")

	inputPolys := make([][]ring.Poly, 2)
	for i := range inputPolys {
		inputPolys[i] = make([]ring.Poly, sparseN)
		for j := range inputPolys[i] {
			inputPolys[i][j] = ringP.NewPoly()
		}
	}

	inputPolysC := make([][]ring.Poly, 2)
	for i := range inputPolysC {
		inputPolysC[i] = make([]ring.Poly, sparseN)
		for j := range inputPolysC[i] {
			inputPolysC[i][j] = ringP.NewPoly()
		}
	}
	result := make([]*rlwe.Ciphertext, N)
	result2 := make([]*rlwe.Ciphertext, N)
	pttemp := hefloat.NewPlaintext(params, params.MaxLevel())
	pttemp.IsBatched = false
	encoder.Encode(value, pttemp)
	fmt.Println("start ct res pre allocate")
	for i := range N {
		cttemp, _ := encryptor.EncryptNew(pttemp)
		result[i] = cttemp.CopyNew()
		result2[i] = cttemp.CopyNew()
	}
	fmt.Println("prealloc end")
	fmt.Println(result[0].LogScale())

	ppmmbuffer1 := make([]float64, len(P)*N*N)
	ppmmbuffer2 := make([]float64, len(P)*N*N)

	resPolys00 := make([][]ring.Poly, 2)
	resPolys00i := make([][]ring.Poly, 2)
	resPolys01 := make([][]ring.Poly, 2)
	resPolys01i := make([][]ring.Poly, 2)
	resPolys10 := make([][]ring.Poly, 2)
	resPolys10i := make([][]ring.Poly, 2)
	resPolys11 := make([][]ring.Poly, 2)
	resPolys11i := make([][]ring.Poly, 2)
	for i := range 2 {
		resPolys00[i] = make([]ring.Poly, N)
		resPolys00i[i] = make([]ring.Poly, N)
		resPolys01[i] = make([]ring.Poly, N)
		resPolys01i[i] = make([]ring.Poly, N)
		resPolys10[i] = make([]ring.Poly, N)
		resPolys10i[i] = make([]ring.Poly, N)
		resPolys11[i] = make([]ring.Poly, N)
		resPolys11i[i] = make([]ring.Poly, N)
		for j := range N {
			resPolys00[i][j] = ringP.NewPoly()
			resPolys00i[i][j] = ringP.NewPoly()
			resPolys01[i][j] = ringP.NewPoly()
			resPolys01i[i][j] = ringP.NewPoly()
			resPolys10[i][j] = ringP.NewPoly()
			resPolys10i[i][j] = ringP.NewPoly()
			resPolys11[i][j] = ringP.NewPoly()
			resPolys11i[i][j] = ringP.NewPoly()
		}
	}
	work := make([]*rlwe.Ciphertext, N)
	for i := range work {
		work[i] = encryptor.EncryptZeroNew(params.MaxLevel())
	}
	aux := make([]*rlwe.Ciphertext, N)
	for i := range aux {
		aux[i] = encryptor.EncryptZeroNew(params.MaxLevel())
	}

	fmt.Println("start c2s")
	starttime = time.Now()
	printMemUsage()

	transpose.Transpose3(cts, params, evaluator, ringQ.AtLevel(cts[0].Level()), N, N, work, aux, cts)

	fmt.Println("mod switch start")

	for i := range sparseN {
		for d := range 2 {
			// be.ModSwitchQtoP_Old(cts[i].Level(), PLevel, cts[i].Value[d], inputPolys[d][i])
			be.ModUpQtoP(PLevel, cts[i].Value[d], inputPolys[d][i])
		}
		if i < n {
			ringQ.AtLevel(cts[0].Level()).Neg(cts[i+n].Value[0], work[0].Value[0])
			ringQ.AtLevel(cts[0].Level()).Neg(cts[i+n].Value[1], work[0].Value[1])
		} else {
			work[0] = cts[i-n]
		}
		for d := range 2 {
			// be.ModSwitchQtoP_Old(cts[i].Level(), PLevel, work[0].Value[d], inputPolysC[d][i])
			be.ModUpQtoP(PLevel, work[0].Value[d], inputPolysC[d][i])
		}
	}

	fmt.Println(inputPolysC[0][0].Coeffs[0])
	fmt.Println("mod switch end")
	fmt.Println("ppmm start")
	inter_it = n
	for l := range len(SFI) {
		inter := inter_it >> CL_arr[l]
		llen := (1 << CL_arr[l])
		if l == 0 {
			for t := range n / llen {
				//00 ~ 01
				// fmt.Println(inputPolys[0][0].Coeffs[0][:100])
				stpoint := (t % inter) + inter_it*int(t/inter)

				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00i, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01i, ppmmbuffer1, ppmmbuffer2)

				//10~11
				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10i, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11i, ppmmbuffer1, ppmmbuffer2)

			}

		} else if l == len(SFI)-1 {
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(resPolys00, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys00i, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00i, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys01, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys01i, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01i, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.SubManyRing(ringP, resPolys00, resPolys00i, resPolys00)
			matmult.AddManyRing(ringP, resPolys01, resPolys01i, resPolys01i)

			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(resPolys10, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys10i, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10i, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys11, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(resPolys11i, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11i, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.SubManyRing(ringP, resPolys10i, resPolys10, resPolys10)
			matmult.AddManyRing(ringP, resPolys11, resPolys11i, resPolys11i)

		}
		inter_it = inter
	}
	matmult.AddManyRing(ringP, resPolys00, resPolys01i, resPolys00)
	matmult.AddManyRing(ringP, resPolys10, resPolys11i, resPolys10)

	rev := matmult.BitReversePermutationMatrix(n)
	matrev := make([]float64, len(P)*N*N)
	for p := range len(P) {
		for i := range N {
			for j := range N {
				if (i < n && j < n) || (i >= n && j >= n) {
					matrev[p*N*N+i*N+j] = (real(rev[i%n][j%n]))
				}
			}
		}
	}

	BigP := ringP.ModulusAtLevel[PLevel]
	bigQ := ringQ.ModulusAtLevel[params.MaxLevel()]
	Div := new(big.Float).SetPrec(params.EncodingPrecision()).Quo(new(big.Float).SetPrec(params.EncodingPrecision()).SetInt(BigP), new(big.Float).SetPrec(params.EncodingPrecision()).SetInt(bigQ))
	fmt.Println(Div.Float64())
	div, _ := Div.Float64()
	matmult.PPMM_Blas_CRT_Inplace(resPolys00, matrev, N, N, N, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
	matmult.PPMM_Blas_CRT_Inplace(resPolys10, matrev, N, N, N, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
	fmt.Println(resPolys00[0][0])
	for i := range N {
		for idx := range 2 {
			be.ModSwitchPtoQ_Old(PLevel, params.MaxLevel(), resPolys00[idx][i], result[i].Value[idx])
			be.ModSwitchPtoQ_Old(PLevel, params.MaxLevel(), resPolys10[idx][i], result2[i].Value[idx])
			// ringQ.Mul
		}
		if i == 0 {
			fmt.Println(result[0].Value[0].Coeffs)
		}

		q := rlwe.NewScale(params.Q()[result[i].Level()])
		util.Mul_ScaleExact(evaluator, result[i], 1.0/(scale), result[i], q)
		util.Rescale_NonNTT(evaluator, result[i], result[i])
		util.Mul_ScaleExact(evaluator, result2[i], 1.0/(scale), result2[i], q)
		util.Rescale_NonNTT(evaluator, result2[i], result2[i])

		q = rlwe.NewScale(params.Q()[result[i].Level()])
		util.Mul_ScaleExact(evaluator, result[i], 1.0/(scale), result[i], q)
		util.Rescale_NonNTT(evaluator, result[i], result[i])
		util.Mul_ScaleExact(evaluator, result2[i], 1.0/(scale), result2[i], q)
		util.Rescale_NonNTT(evaluator, result2[i], result2[i])

		q = rlwe.NewScale(params.Q()[result[i].Level()])
		util.Mul_ScaleExact(evaluator, result[i], div, result[i], q)
		util.Rescale_NonNTT(evaluator, result[i], result[i])
		// scaling, _ := btp.CoeffsToSlotsParameters.Scaling.Float64()
		// q = rlwe.NewScale(params.Q()[result[i].Level()])
		// util.Mul_ScaleExact(evaluator, result[i], scaling, result[i], q)
		// util.Rescale_NonNTT(evaluator, result[i], result[i])
	}
	fmt.Println(result[0].Level())

	transpose.Transpose3(result, params, evaluator, ringQ.AtLevel(result[0].Level()), N, sparseN, work, aux, result)
	transpose.Transpose3(result2, params, evaluator, ringQ.AtLevel(result2[0].Level()), N, sparseN, work, aux, result2)

	elapse = time.Since(starttime)
	fmt.Println(elapse)

	for i := range result {

		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[0], result[i].Value[0])
		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[1], result[i].Value[1])

		conj, _ := evaluator.ConjugateNew(result[i])
		evaluator.Add(result[i], conj, result[i])
		result[i], err = btp.EvalMod(result[i])
		if err != nil {
			panic(err)
		}

		// ringQ.AtLevel(result2[i].Level()).NTT(result2[i].Value[0], result2[i].Value[0])
		// ringQ.AtLevel(result2[i].Level()).NTT(result2[i].Value[1], result2[i].Value[1])
	}
	fmt.Println(result[0].Value[0].Coeffs)
	resvalue := make([][]complex128, N)
	maxerr := 0.0
	idx1 := 0
	idx2 := 0
	for i := range result {
		result[i].IsBatched = true
		dept := decryptor.DecryptNew(result[i])
		resvalue[i] = make([]complex128, n)
		encoder.Decode(dept, resvalue[i])

		if i < 10 {
			fmt.Println(resvalue[i][:10])
		}
		for j := range resvalue[i] {
			val := math.Abs(value[j] - real(resvalue[i][j]))
			if val > maxerr {
				idx1 = i
				idx2 = j
				maxerr = val
				// fmt.Println(maxerr)
			}
		}
	}

	fmt.Println(-math.Log2(maxerr))
	fmt.Println(maxerr)
	fmt.Println(idx2, " ", value[idx2], " ", resvalue[idx1][idx2])
	pt = hefloat.NewPlaintext(params, 0)
	pt.IsBatched = false

	cts = make([]*rlwe.Ciphertext, N)
	for i := range cts {
		encoder.Encode(value, pt)
		cts[i], _ = encryptor.EncryptNew(pt)
		// ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
		// ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
	}

	for i := range cts {
		cts[i], _ = btp.ModUp(cts[i])
		cts[i], _, err = btp.CoeffsToSlots(cts[i])
		// if err != nil {
		// 	panic(err)
		// }
		cts[i], _ = btp.EvalMod(cts[i])
		// cts[i], _ = btp.SlotsToCoeffs(eval, nil)
	}
	maxerr = 0.0
	idx1 = 0
	idx2 = 0
	for i := range cts {
		cts[i].IsBatched = true
		dept := decryptor.DecryptNew(cts[i])
		encoder.Decode(dept, resvalue[i])
		if i < 10 {
			fmt.Println(resvalue[i][:10])
		}

		for j := range resvalue[i] {
			val := math.Abs(value[bitReverse(j, 9)] - real(resvalue[i][j]))
			if val > maxerr {
				idx1 = i
				idx2 = j
				maxerr = val
			}
		}
	}
	fmt.Println(-math.Log2(maxerr))
	fmt.Println(maxerr)
	fmt.Println(idx2, " ", value[idx2], " ", resvalue[idx1][idx2])
	pt = hefloat.NewPlaintext(params, 0)
	pt.IsBatched = false

}

func Test_CheckC2SPrams(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            5,
		LogQ:            []int{48, 40, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52, 52, 52},
		LogDefaultScale: 40,
	}
	//parameter init
	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}

	fmt.Println("ckks parameter init end")

	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	N := 1 << params.LogN()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, N)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	starttime := time.Now()
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		i := i
		gal := galEls[i]
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(gal, sk)
		}()
	}
	wg.Wait()
	elapse := time.Since(starttime)
	fmt.Println(elapse)
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	_, _ = encryptor, encoder
	fmt.Println("generate Evaluator end")

	CoeffsToSlotsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicEncode,
		Format:       hefloat.RepackImagAsReal, // Returns the real and imaginary part into separate ciphertexts
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel(),
		Levels:       []int{1, 1}, //qiCoeffsToSlots
		LogBSGSRatio: 0,
		BitReversed:  false,
	}

	// Parameters of the homomorphic modular reduction x mod 1
	Mod1ParametersLiteral := hefloat.Mod1ParametersLiteral{
		LevelStart:      params.MaxLevel() - 2,
		LogScale:        48,                  // Matches qiEvalMod
		Mod1Type:        hefloat.CosDiscrete, // Multi-interval Chebyshev interpolation
		Mod1Degree:      30,                  // Depth 6
		DoubleAngle:     3,                   // Depth 3
		K:               8,                   // With EphemeralSecretWeight = 32 and 2^{15} slots, ensures < 2^{-138.7} failure probability
		LogMessageRatio: 8,                   // q/|m| = 2^10
		Mod1InvDegree:   0,                   // Depth 0
	}

	// SlotsToCoeffs parameters (homomorphic decoding)
	SlotsToCoeffsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicDecode,
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel() - 10,
		Levels:       []int{1, 1}, // qiSlotsToCoeffs
		LogBSGSRatio: 0,
		BitReversed:  false,
	}

	// Custom bootstrapping.Parameters.
	// All fields are public and can be manually instantiated.
	btpParams := bootstrapping.Parameters{
		ResidualParameters:      params,
		BootstrappingParameters: params,
		SlotsToCoeffsParameters: SlotsToCoeffsParameters,
		Mod1ParametersLiteral:   Mod1ParametersLiteral,
		CoeffsToSlotsParameters: CoeffsToSlotsParameters,
		EphemeralSecretWeight:   32, // > 128bit secure for LogN=16 and LogQP = 115.
		CircuitOrder:            bootstrapping.Custom,
	}
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)
	_ = decryptor
	_ = evaluator
	btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	if err != nil {
		panic(err)
	}
	_ = btp
	fmt.Println("generate btp Evaluator end")

	fmt.Println("ckks log degree : ", params.LogN())
	fmt.Println(params.EncodingPrecision())
	CL_arr := []int{2, 2}
	_, SFI := matmult.GenSFMat_CL(params, CL_arr, CL_arr)
	C2SScale, _ := btp.CoeffsToSlotsParameters.Scaling.Float64()
	scaling_ := new(big.Float).SetFloat64(C2SScale)
	scaling_.Quo(scaling_, new(big.Float).SetFloat64(float64(N)))
	scaling_ = bignum.Pow(scaling_, new(big.Float).Quo(new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(1), new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(2)))
	fmt.Println(scaling_)
	for i := range SFI {
		for j := range SFI[i] {
			for k := range SFI[i][j] {
				val := new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(real(SFI[i][j][k]))
				val = val.Mul(val, scaling_)
				val2 := new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(imag(SFI[i][j][k]))
				val2 = val2.Mul(val2, scaling_)
				v1, _ := val.Float64()
				v2, _ := val2.Float64()
				SFI[i][j][k] = complex(v1, v2)
			}
		}
	}

	for i := range SFI {
		for j := range SFI[i] {
			for k := range SFI[i][j] {
				fmt.Print(SFI[i][j][k], " ")
			}
			fmt.Println()
		}
		fmt.Println()
	}
	fmt.Println()
	diagonal := btp.CoeffsToSlotsParameters.GenMatrices(params.LogN(), 53)
	fmt.Println(len(diagonal))
	fmt.Println(len(diagonal[0]))
	for i := range diagonal {
		for j := range diagonal[i] {
			for k := range diagonal[i][j] {
				fmt.Print((*diagonal[i][j][k]), " ")
			}
			fmt.Println()
		}
		fmt.Println()
	}
}
func Test_TransposePrec(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            10,
		LogQ:            []int{48, 40, 40, 40, 40},
		LogP:            []int{52, 52, 52},
		LogDefaultScale: 40,
	}
	//parameter init
	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}

	fmt.Println("ckks parameter init end")

	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	N := 1 << params.LogN()
	sparseN := N >> 1

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, sparseN)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	starttime := time.Now()
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		i := i
		gal := galEls[i]
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(gal, sk)
		}()
	}
	wg.Wait()
	elapse := time.Since(starttime)
	fmt.Println(elapse)
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	fmt.Println("generate Evaluator end")

	work := make([]*rlwe.Ciphertext, sparseN)
	for i := range work {
		work[i] = encryptor.EncryptZeroNew(params.MaxLevel())
	}
	aux := make([]*rlwe.Ciphertext, sparseN)
	for i := range aux {
		aux[i] = encryptor.EncryptZeroNew(params.MaxLevel())
	}

	fmt.Println("start c2s")
	starttime = time.Now()
	printMemUsage()

	value := make([]float64, N)
	for j := range value {
		value[j] = 0.0001 * float64(j)
	}
	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false
	ringQ := params.RingQ()
	encoder.Encode(value, pt)
	cts := make([]*rlwe.Ciphertext, sparseN)
	for i := range cts {
		encoder.Encode(value, pt)
		cts[i], _ = encryptor.EncryptNew(pt)
		// cts[i], _ = btp.ModUp(cts[i])
		ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
		// cts[i], _ = ModUp(btp, cts[i])
	}
	fmt.Println(cts[0].LogScale())

	transpose.Transpose3(cts, params, evaluator, ringQ.AtLevel(cts[0].Level()), N, sparseN, work, aux, cts)
	fmt.Println(len(cts))
	resvalue := make([]float64, N)
	maxerr := 0.0
	for i := range cts {
		ringQ.AtLevel(cts[i].Level()).NTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.AtLevel(cts[i].Level()).NTT(cts[i].Value[1], cts[i].Value[1])
		ptres := decryptor.DecryptNew(cts[i])
		encoder.Decode(ptres, resvalue)
		// fmt.Println(resvalue)
		for j := range resvalue {
			if j%(N/sparseN) != 0 {
				continue
			}
			val := math.Abs(0.0001*float64(i)*float64(N/sparseN) - resvalue[j])
			if maxerr < val {
				maxerr = val
			}
		}
	}
	fmt.Println(maxerr)
	fmt.Println(-math.Log2(maxerr))
	maxValue := 0.0
	for _, v := range value {
		if v > maxValue {
			maxValue = v
		}
	}
	fmt.Println(-math.Log2(maxerr / maxValue))

}

// ModUp raise the modulus from q to Q, scales the message  and applies the Trace if the ciphertext is sparsely packed.
func ModUp(eval *bootstrapping.Evaluator, ctIn *rlwe.Ciphertext) (ctOut *rlwe.Ciphertext, err error) {

	params := eval.BootstrappingParameters

	ringQ := params.RingQ().AtLevel(ctIn.Level())

	// Extend the ciphertext from q to Q with zero values.
	ctIn.Resize(ctIn.Degree(), params.MaxLevel())

	levelQ := params.QCount() - 1

	ringQ = ringQ.AtLevel(levelQ)

	Q := ringQ.ModuliChain()
	q := Q[0]
	BRCQ := ringQ.BRedConstants()

	var coeff, tmp, pos, neg uint64

	N := ringQ.N()

	// ModUp q->Q for ctIn[0] centered around q
	for j := 0; j < N; j++ {

		coeff = ctIn.Value[0].Coeffs[0][j]
		pos, neg = 1, 0
		if coeff >= (q >> 1) {
			coeff = q - coeff
			pos, neg = 0, 1
		}

		for i := 1; i < levelQ+1; i++ {
			tmp = ring.BRedAdd(coeff, Q[i], BRCQ[i])
			ctIn.Value[0].Coeffs[i][j] = tmp*pos + (Q[i]-tmp)*neg
		}
	}

	for j := 0; j < N; j++ {

		coeff = ctIn.Value[1].Coeffs[0][j]
		pos, neg = 1, 0
		if coeff >= (q >> 1) {
			coeff = q - coeff
			pos, neg = 0, 1
		}

		for i := 1; i < levelQ+1; i++ {
			tmp = ring.BRedAdd(coeff, Q[i], BRCQ[i])
			ctIn.Value[1].Coeffs[i][j] = tmp*pos + (Q[i]-tmp)*neg
		}
	}
	return ctIn, nil
}

func printMemUsage() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	fmt.Println("////////////////////////////////////////////////////")
	fmt.Printf("Alloc = %v MiB\n", bToMb(m.Alloc))
	fmt.Printf("Sys = %v MiB\n", bToMb(m.Sys))
	fmt.Printf("NumGC = %v\n", m.NumGC)
	fmt.Println("////////////////////////////////////////////////////")

	// 현재 실행 중인 프로세스 정보 가져오기
	p, _ := process.NewProcess(int32(os.Getpid()))

	// 실제 물리 메모리 점유량 (RSS) 가져오기
	memInfo, _ := p.MemoryInfo()

	// RSS: 실제 물리 램 점유량
	// VMS: 가상 메모리 전체 (현재 보시는 24TB 수치는 여기에 해당)
	fmt.Printf("실제 물리 메모리 (RSS): %v MiB\n", memInfo.RSS/1024/1024)
	fmt.Printf("가상 메모리 (VMS): %v MiB\n", memInfo.VMS/1024/1024)
}

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}
