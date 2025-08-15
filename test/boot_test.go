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
				fmt.Println(cmplx.Abs(a[i][j]-b[i][j]), " ", i, " ", j)
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

func Test_Boot(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
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
		Levels:       []int{1, 1}, //qiCoeffsToSlots
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

	fmt.Println("ckks parameter init end")
	fmt.Println(btpParams.BootstrappingParameters.NthRoot())
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
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)

	btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	if err != nil {
		panic(err)
	}
	fmt.Println("generate Evaluator end")

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

	mat_00 := make([][]uint64, 2*n)
	mat_01 := make([][]uint64, 2*n)
	scale := params.DefaultScale().Float64()
	for i := range 2 * n {
		mat_00[i] = make([]uint64, 2*n)
		mat_01[i] = make([]uint64, 2*n)
		for j := range 2 * n {
			if i < n && j < n {
				mat_00[i][j] = uint64(real(SFI[i][j]) * scale)
			}
			if i >= n && j >= n {
				mat_01[i][j] = uint64(imag(SFI[i%n][j%n]) * scale)
			}

		}
	}
	mat_10 := make([][]uint64, 2*n)
	mat_11 := make([][]uint64, 2*n)
	for i := range 2 * n {
		mat_10[i] = make([]uint64, 2*n)
		mat_11[i] = make([]uint64, 2*n)
		for j := range 2 * n {
			if i < n && j < n {
				//inv please
				mat_10[i][j] = uint64(real(SFI[i][j]) * scale)
			}
			if i >= n && j >= n {
				//inv please
				mat_11[i][j] = uint64(imag(SFI[i%n][j%n]) * scale)
			}

		}
	}

	_, _, _, _ = encoder, encryptor, decryptor, evaluator
	_ = btp

	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, 3)
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	cts := make([]*rlwe.Ciphertext, 2*n)
	for i := range cts {
		cts[i] = ct.CopyNew()
	}

	starttime = time.Now()
	starttime_ := time.Now()
	ctT := transpose.Transpose(cts, params, evaluator, encoder, 2*n)
	elapse_ := time.Since(starttime_)
	fmt.Println("transpose time per once : ", elapse_)

	ctT2 := make([]*rlwe.Ciphertext, 2*n)
	for i := range ctT2 {
		if i < n {
			ctT2[i], _ = evaluator.MulNew(ctT[i+n], -1)
		} else {
			ctT2[i] = ctT[i-n].CopyNew()
		}
	}

	res00 := matmult.PPMM_Flint(ctT, mat_00, params, 2*n)
	res01 := matmult.PPMM_Flint(ctT2, mat_01, params, 2*n)
	res0 := make([]*rlwe.Ciphertext, 2*n)
	for i := range res0 {
		res0[i], _ = evaluator.AddNew(res00[i], res01[i])
		evaluator.Mul(res0[i], 1.0/params.DefaultScale().Float64(), res0[i])
		evaluator.Rescale(res0[i], res0[i])
	}

	starttime_ = time.Now()
	res10 := matmult.PPMM_Flint(ctT, mat_10, params, 2*n)
	elapse_ = time.Since(starttime_)
	fmt.Println("ppmm time per twice : ", elapse_)

	res11 := matmult.PPMM_Flint(ctT2, mat_11, params, 2*n)
	res1 := make([]*rlwe.Ciphertext, 2*n)
	for i := range res1 {
		res1[i], _ = evaluator.AddNew(res10[i], res11[i])
		evaluator.Mul(res1[i], 1.0/params.DefaultScale().Float64(), res1[i])
		evaluator.Rescale(res1[i], res1[i])
	}

	result0 := transpose.Transpose(res0, params, evaluator, encoder, 2*n)
	result1 := transpose.Transpose(res1, params, evaluator, encoder, 2*n)
	_ = result1

	elapse = time.Since(starttime)
	fmt.Println("total time : ", elapse)

	starttime_ = time.Now()
	ringQ := params.RingQ().AtLevel(cts[0].Level())
	for i := range cts {
		ringQ.NTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.NTT(cts[i].Value[1], cts[i].Value[1])
		ringQ.INTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.INTT(cts[i].Value[1], cts[i].Value[1])
	}
	elapse_ = time.Since(starttime_)
	fmt.Println("NTT / INTT per 2 * n : ", elapse_)

	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")

	starttime = time.Now()
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

	elapse = time.Since(starttime)
	fmt.Println("origin total time : ", elapse)

	for i := range result0 {
		dept := rlwe.NewPlaintext(params, 1)
		value := make([]float64, n)

		result0[i].IsBatched = true
		decryptor.Decrypt(result0[i], dept)
		encoder.Decode(dept, value)

		fmt.Println(value[0:10])
	}

}

func Test_Boot3(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
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
		Levels:       []int{1, 1}, //qiCoeffsToSlots
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

	fmt.Println("ckks parameter init end")
	fmt.Println(btpParams.BootstrappingParameters.NthRoot())
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
	galEls := make([]uint64, 16)
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
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)

	btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	if err != nil {
		panic(err)
	}
	fmt.Println("generate Evaluator end")

	_, _, _, _ = encoder, encryptor, decryptor, evaluator
	_ = btp

	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, 3)
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	cts := make([]*rlwe.Ciphertext, 2*n)
	for i := range cts {
		cts[i] = ct.CopyNew()
	}

	starttime = time.Now()
	starttime_ := time.Now()
	ctT := transpose.Transpose(cts, params, evaluator, encoder, 2*n)
	elapse_ := time.Since(starttime_)
	fmt.Println("transpose time per once : ", elapse_)

	ctT2 := make([]*rlwe.Ciphertext, 2*n)
	for i := range ctT2 {
		if i < n {
			ctT2[i], _ = evaluator.MulNew(ctT[i+n], -1)
		} else {
			ctT2[i] = ctT[i-n].CopyNew()
		}
	}

	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")

	starttime = time.Now()
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

	elapse = time.Since(starttime)
	fmt.Println("origin total time : ", elapse)
}

func Test_CoefToSlot(t *testing.T) {

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
		Levels:       []int{1, 1}, //qiCoeffsToSlots
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

	fmt.Println("ckks parameter init end")
	fmt.Println(btpParams.BootstrappingParameters.NthRoot())
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
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)

	btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	if err != nil {
		panic(err)
	}
	fmt.Println("generate Evaluator end")

	_, _, _, _ = encoder, encryptor, decryptor, evaluator

	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.01 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, 2)
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	fmt.Println(ct.Level())
	fmt.Println(ct.LogScale())
	_ = ct
	ct1, _, _ := btp.CoeffsToSlots(ct)

	ct1.IsBatched = true
	// var ct1 *rlwe.Ciphertext
	// if ct1, err = btp.Bootstrap(ct); err != nil {
	// 	panic(err)
	// }
	// fmt.Println(ct1.Level())
	// fmt.Println(ct1.LogScale())

	dept := decryptor.DecryptNew(ct1)
	result := make([]float64, n)
	encoder.Decode(dept, result)

	for i := range result {
		if i > 10 {
			break
		}
		fmt.Print(result[i]/0.125, " ")
	}
	fmt.Println()
	fmt.Println(result[n/2])
}
