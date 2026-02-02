package main

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
)

func main() {
	CheckBootOrigin2()
}

func CheckBootOrigin2() {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU())

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            5,
		LogQ:            []int{48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 39, 39, 39, 60, 60, 60, 60, 60, 60, 60, 60, 56, 56, 56},
		LogP:            []int{61, 61, 61},
		LogDefaultScale: 40,
	}

	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	// params, err := hefloat.NewParametersFromLiteral(examples.HEFloatComplexParamsPN16QP1761)
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

	galLen := 256
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
	fmt.Println("generate Evaluator end")

	_, _, _, _ = encoder, encryptor, decryptor, evaluator
	// CoeffsToSlots parameters (homomorphic encoding)
	CoeffsToSlotsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicEncode,
		Format:       hefloat.RepackImagAsReal, // Returns the real and imaginary part into separate ciphertexts
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel(),
		Levels:       []int{1, 1, 1}, //qiCoeffsToSlots
		LogBSGSRatio: 1,
	}

	// Parameters of the homomorphic modular reduction x mod 1
	Mod1ParametersLiteral := hefloat.Mod1ParametersLiteral{
		LevelStart:      params.MaxLevel() - 3,
		LogScale:        60,                  // Matches qiEvalMod
		Mod1Type:        hefloat.CosDiscrete, // Multi-interval Chebyshev interpolation
		Mod1Degree:      30,                  // Depth 5
		DoubleAngle:     3,                   // Depth 3
		K:               16,                  // With EphemeralSecretWeight = 32 and 2^{15} slots, ensures < 2^{-138.7} failure probability
		LogMessageRatio: 8,                   // q/|m| = 2^10
		Mod1InvDegree:   0,                   // Depth 0
	}

	// SlotsToCoeffs parameters (homomorphic decoding)
	SlotsToCoeffsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicDecode,
		Format:       hefloat.RepackImagAsReal,
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel() - 11,
		Levels:       []int{1, 1, 1}, // qiSlotsToCoeffs
		LogBSGSRatio: 1,
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
	btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	if err != nil {
		panic(err)
	}
	fmt.Println("btp gen end")

	n := 1 << params.LogMaxSlots()
	value := make([]float64, n)
	for i := range value {
		value[i] = sampling.RandFloat64(-1.0, 1.0)
	}

	pt := hefloat.NewPlaintext(params, 0)
	pt.IsBatched = true

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	valuesDecBefore := debugAndPrecision(params, ct, value, decryptor, encoder, n)
	runtime.GOMAXPROCS(1)

	res, _ := btp.Bootstrap(ct)

	debugAndPrecision(params, res, valuesDecBefore, decryptor, encoder, n)
}

func CheckBootOrigin() {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU())

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 39, 39, 39, 60, 60, 60, 60, 60, 60, 60, 60, 56, 56, 56},
		LogP:            []int{61, 61, 61},
		LogDefaultScale: 40,
	}

	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	// params, err := hefloat.NewParametersFromLiteral(examples.HEFloatComplexParamsPN16QP1761)
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

	galLen := 256
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
	fmt.Println("generate Evaluator end")

	_, _, _, _ = encoder, encryptor, decryptor, evaluator
	// CoeffsToSlots parameters (homomorphic encoding)
	CoeffsToSlotsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicEncode,
		Format:       hefloat.RepackImagAsReal, // Returns the real and imaginary part into separate ciphertexts
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel(),
		Levels:       []int{1, 1, 1}, //qiCoeffsToSlots
		LogBSGSRatio: 1,
	}

	// Parameters of the homomorphic modular reduction x mod 1
	Mod1ParametersLiteral := hefloat.Mod1ParametersLiteral{
		LevelStart:      params.MaxLevel() - 3,
		LogScale:        60,                  // Matches qiEvalMod
		Mod1Type:        hefloat.CosDiscrete, // Multi-interval Chebyshev interpolation
		Mod1Degree:      30,                  // Depth 5
		DoubleAngle:     3,                   // Depth 3
		K:               16,                  // With EphemeralSecretWeight = 32 and 2^{15} slots, ensures < 2^{-138.7} failure probability
		LogMessageRatio: 8,                   // q/|m| = 2^10
		Mod1InvDegree:   0,                   // Depth 0
	}

	// SlotsToCoeffs parameters (homomorphic decoding)
	SlotsToCoeffsParameters := hefloat.DFTMatrixLiteral{
		Type:         hefloat.HomomorphicDecode,
		Format:       hefloat.RepackImagAsReal,
		LogSlots:     params.LogMaxSlots(),
		LevelStart:   params.MaxLevel() - 11,
		Levels:       []int{1, 1, 1}, // qiSlotsToCoeffs
		LogBSGSRatio: 1,
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
	logNBootstrap := 16
	btpLit := bootstrapping.ParametersLiteral{
		LogN: &logNBootstrap,
		// CoeffsToSlotsFactorizationDepthAndLogScales: [][]int{{56}, {56}, {56}, {56}},
	}
	fmt.Printf("btpLit.GetDefaultXs(): %v\n", btpLit.GetDefaultXs())
	fmt.Printf("btpLit.GetDefaultXe(): %v\n", btpLit.GetDefaultXe())
	fmt.Printf("btpParams.BootstrappingParameters.Xs(): %v\n", btpParams.BootstrappingParameters.Xs())
	fmt.Printf("btpParams.BootstrappingParameters.Xe(): %v\n", btpParams.BootstrappingParameters.Xe())
	// 3. 부트스트래핑 파라미터 인스턴스 생성
	// btpParams, err := bootstrapping.NewParametersFromLiteral(params, btpLit)
	// if err != nil {
	// 	panic(err)
	// }
	fmt.Printf("btpParams: %v\n", btpParams)
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)
	btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	if err != nil {
		panic(err)
	}
	fmt.Println("btp gen end")

	n := 1 << params.LogMaxSlots()
	value := make([]float64, n)
	for i := range value {
		value[i] = sampling.RandFloat64(-1.0, 1.0)
	}

	pt := hefloat.NewPlaintext(params, 0)
	pt.IsBatched = true

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	valuesDecBefore := debugAndPrecision(params, ct, value, decryptor, encoder, n)
	runtime.GOMAXPROCS(1)

	res, _ := btp.Bootstrap(ct)

	debugAndPrecision(params, res, valuesDecBefore, decryptor, encoder, n)
	// fmt.Println(btpParams.SlotsToCoeffsParameters.LevelStart)
	fmt.Printf("btpParams.SlotsToCoeffsParameters: %v\n", btpParams.SlotsToCoeffsParameters)
	fmt.Printf("btpParams.Mod1ParametersLiteral: %v\n", btpParams.Mod1ParametersLiteral)
	fmt.Printf("btpParams.CoeffsToSlotsParameters: %v\n", btpParams.CoeffsToSlotsParameters)
	// fmt.Printf("btpParams.BootstrappingParameters.Parameters.Parameters.LogQi(): %v\n", btpParams.BootstrappingParameters.Parameters.Parameters.LogQi())
}

func debugAndPrecision(
	params hefloat.Parameters,
	ct *rlwe.Ciphertext,
	valuesRef []float64,
	decryptor *rlwe.Decryptor,
	encoder *hefloat.Encoder,
	n int,
) (valuesDec []float64) {

	valuesDec = make([]float64, n)
	if err := encoder.Decode(decryptor.DecryptNew(ct), valuesDec); err != nil {
		panic(err)
	}

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ct.Level(), params.LogQLvl(ct.Level()))
	fmt.Printf("Scale: 2^%.4f\n", math.Log2(ct.Scale.Float64()))
	fmt.Printf("Decoded[0..3]:   %6.10f %6.10f %6.10f %6.10f\n",
		valuesDec[0], valuesDec[1], valuesDec[2], valuesDec[3])
	fmt.Printf("Reference[0..3]: %6.10f %6.10f %6.10f %6.10f\n",
		valuesRef[0], valuesRef[1], valuesRef[2], valuesRef[3])

	// hefloat.GetPrecisionStats 를 이용해서 정밀도 통계 계산
	//  - 내부적으로 slot별 상대오차를 보고 log2 기준의 precision 비트 수 등을 계산
	//  - 출력 문자열에는 Min/Max/Mean precision, 표준편차 등이 포함됨
	precStats := hefloat.GetPrecisionStats(params, encoder, nil, valuesRef, valuesDec, 0, false)
	fmt.Println(precStats.String())
	fmt.Println()

	return
}
