package test

import (
	"fmt"
	"math/cmplx"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/lifejade/mm/src/matmult"
	"github.com/lifejade/mm/src/transpose"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/schemes/ckks"
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

func debugCTS(cts []*rlwe.Ciphertext, params hefloat.Parameters, encoder *hefloat.Encoder, decryptor *rlwe.Decryptor) {
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////")
	fmt.Println("debug cts")
	var n int
	if cts[0].IsBatched {
		n = params.MaxSlots()
	} else {
		n = params.N()
	}
	value := make([]float64, n)
	for i := range cts {
		dept := decryptor.DecryptNew(cts[i])
		encoder.Decode(dept, value)
		for j := range value {
			fmt.Print(value[j], " ")
		}
		fmt.Println()
	}
	fmt.Println("debug cts end")
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////")
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
	debugCTS(ctT, params, encoder, decryptor)
	res00 := matmult.PPMM_Flint_CRT(ctT, mat_00, params, 2*n)
	for i := range res00 {
		evaluator.Mul(res00[i], 1/matscale, res00[i])
		evaluator.Rescale(res00[i], res00[i])
	}

	debugCTS(res00, params, encoder, decryptor)
	result0 := transpose.Transpose(res00, params, evaluator, encoder, 2*n)
	debugCTS(result0, params, encoder, decryptor)

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
	// debugCTS(res00, params, encoder, decryptor)
	// debugCTS(res01, params, encoder, decryptor)
	// res0 := make([]*rlwe.Ciphertext, 2*n)
	// for i := range res0 {
	// 	res0[i], _ = evaluator.AddNew(res00[i], res01[i])
	// 	evaluator.Mul(res0[i], 1.0/(scale), res0[i])
	// 	evaluator.Rescale(res0[i], res0[i])

	// }
	// debugCTS(res0, params, encoder, decryptor)

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
	debugCTS(result0, params, encoder, decryptor)
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

	debugCTS(result, params, encoder, decryptor)
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
	}

	// Parameters of the homomorphic modular reduction x mod 1
	Mod1ParametersLiteral := hefloat.Mod1ParametersLiteral{
		LevelStart:      params.MaxLevel() - 2,
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
		LevelStart:   params.MaxLevel() - 10,
		Levels:       []int{1, 1, 1}, // qiSlotsToCoeffs
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
	scale := float64(1 << 40)
	mat0, mat1, mat2, mat3 := matmult.GenC2SMat(SFI, scale, params)
	mat0_, mat1_, mat2_, mat3_ := matmult.GenS2CMat(SF, scale, params)

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

	fmt.Println("gen cts & mats")
	printMemUsage()

	starttime_ := time.Now()
	cts1_, cts2_ := matmult.C2S_OnceMul(cts, params, evaluator, encoder, mat0, mat1, mat2, mat3, scale)
	elapse_ := time.Since(starttime_)
	fmt.Println("cts time : ", elapse_)
	fmt.Println("ckeck cts")
	printMemUsage()

	starttime_ = time.Now()
	for i := range cts1_ {
		cts1_[i], _ = btp.EvalMod(cts1_[i])
	}
	for i := range cts2_ {
		cts2_[i], _ = btp.EvalMod(cts2_[i])
	}
	elapse_ = time.Since(starttime_)
	fmt.Println("eval time : ", elapse_)

	starttime_ = time.Now()
	res := matmult.S2C_OnceMul(cts1_, cts2_, params, evaluator, encoder, mat0_, mat1_, mat2_, mat3_, scale)
	elapse_ = time.Since(starttime_)
	fmt.Println("stc time : ", elapse_)
	fmt.Println("ckeck stc")
	printMemUsage()
	_ = res

	res = nil
	cts1_ = nil
	cts2_ = nil
	mat0, mat1, mat2, mat3, mat0_, mat1_, mat2_, mat3_ = nil, nil, nil, nil, nil, nil, nil, nil
	printMemUsage()

	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")

	starttime_ = time.Now()
	cts2 := make([]*rlwe.Ciphertext, 2*n)
	for i := range cts2 {
		cts2[i], _, _ = btp.DFTEvaluator.CoeffsToSlotsNew(cts[i], btp.C2SDFTMatrix)
	}
	elapse_ = time.Since(starttime_)
	fmt.Println("cts time(origin) : ", elapse_)

	starttime_ = time.Now()
	for i := range cts2 {
		cts2[i], _ = btp.EvalMod(cts2[i])
	}
	elapse_ = time.Since(starttime_)
	fmt.Println("eval time(origin) : ", elapse_)

	starttime_ = time.Now()
	for i := range cts2 {
		cts2[i], _ = btp.DFTEvaluator.SlotsToCoeffsNew(cts[i], nil, btp.S2CDFTMatrix)
	}
	elapse_ = time.Since(starttime_)
	fmt.Println("stc time(origin) : ", elapse_)
}

func printMemUsage() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	fmt.Println("////////////////////////////////////////////////////")
	fmt.Printf("Alloc = %v MiB\n", bToMb(m.Alloc))
	fmt.Printf("Sys = %v MiB\n", bToMb(m.Sys))
	fmt.Printf("NumGC = %v\n", m.NumGC)
	fmt.Println("////////////////////////////////////////////////////")
}

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}
