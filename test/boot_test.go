package test

import (
	"fmt"
	"math/big"
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
	_ = btp

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

	ctT := transpose.Transpose(cts, params, evaluator, encoder, 2*n)
	_ = ctT

	roots := ckks.GetRootsBigComplex(n<<2, params.EncodingPrecision())
	roots2 := make([]uint64, 4*n)

	for i := range roots2 {
		temp := roots[i].Real()
		scale := new(big.Float)
		scale.SetFloat64(params.DefaultScale().Float64())

		temp.Mul(temp, scale)
		tempint := new(big.Int)
		temp.Int(tempint)

		r := new(big.Int)
		r.Mod(tempint, params.RingQ().ModulusAtLevel[0])

		roots2[i] = r.Uint64()
		fmt.Print(r, " ")

	}
	fmt.Println()

	pow5 := make([]int, (n<<1)+1)
	pow5[0] = 1
	for i := 1; i < (n<<1)+1; i++ {
		pow5[i] = pow5[i-1] * 5
		pow5[i] &= (n << 2) - 1
	}

	mat := make([][]uint64, 2*n)
	for i := range mat {
		mat[i] = make([]uint64, 2*n)
		for j := range mat[i] {
			idx := (pow5[i] * j) & ((n << 2) - 1)
			mat[i][j] = roots2[idx]
			fmt.Print(idx, " ")
		}
		fmt.Println()
	}

	res := matmult.PPMM_Flint(ctT, mat, params, 2*n)
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////")

	result2 := transpose.Transpose(res, params, evaluator, encoder, 2*n)

	for i := range result2 {
		dept := rlwe.NewPlaintext(params, 1)
		value := make([]float64, n)
		evaluator.Mul(result2[i], 1.0/params.DefaultScale().Float64(), result2[i])
		evaluator.Rescale(result2[i], result2[i])

		result2[i].IsBatched = true
		decryptor.Decrypt(result2[i], dept)
		encoder.Decode(dept, value)

		fmt.Println(value[0:10])
	}
	fmt.Println(result2[0].LogScale())

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
