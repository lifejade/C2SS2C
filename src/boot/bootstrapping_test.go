package boot

import (
	"fmt"
	"math/big"
	"math/bits"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/lifejade/mm/src/matmult"
	"github.com/lifejade/mm/src/util"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
)

func Test_Modup(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU())

	logN := 8
	sparseN := 1 << 5
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
	fmt.Println("ckks parameter init end")

	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	N := 1 << params.LogN()

	galLen := 1

	P := []uint32{3422539, 3370361, 3231143, 3545881, 3577031, 3832931, 4064197, 3617099, 3651497, 3711319, 3439693, 3502001, 3555509, 3552013, 4031179, 4115407, 3167453, 3365393, 3291143, 3204973, 4182419, 3495781, 3315883, 3403391, 3529153, 3390899, 3453773, 3705469, 3180337, 4091993, 3503221, 3598949, 3822277, 3277853, 3547249, 3278053, 3696257, 3849409, 3725257, 3239449, 3730721, 3393619, 3361363, 3732997, 3661573, 3158971, 3516031, 3737039, 3882649, 3614969, 3518491, 3169759, 3326417, 4165333, 3853097, 3845357, 3721603, 3494831, 3255467, 3442987, 3381641, 4188433, 3960053, 3825473, 3269713, 3373781, 3403843, 4177609, 3265337, 3382231, 3342137, 3330179, 3272629, 3725357, 3667453, 3960049, 3435323, 3664249, 3632423, 3515269, 3784733, 3377657, 4064143, 3702119, 3835367, 3564937, 3507397, 3345877, 4169129, 3206783, 3397769, 4145293, 3773477, 3229319, 3161617, 3517427, 3456743, 3687163, 3389423, 3553541}
	PLevel := 33
	QPratio := new(big.Float).SetFloat64(1)
	for i := range params.Q() {
		QPratio.Mul(QPratio, new(big.Float).SetUint64(params.Q()[i]))
	}
	for i := range P[:PLevel+1] {
		QPratio.Quo(QPratio, new(big.Float).SetUint64(uint64(P[i])))
	}
	fmt.Println(QPratio)

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
	// for i := range ratio {
	// 	k := N >> i
	// 	galEls = append(galEls, uint64(k+1))
	// }

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
	util.SContext = util.SecretContext{
		Sk:        sk,
		Decryptor: decryptor,
	}
	util.Debug = util.DebugContext{
		IsDebug:   true,
		AccTime:   time.Duration(0),
		StartTime: time.Time{},
	}

	ratio := N / sparseN
	// context := InitContext(params, encoder, encryptor, N, sparseN, evaluator, P, MatmultParamsLiteral{}, MatmultParamsLiteral{})
	util.SContext.Values = make([][]float64, sparseN)
	for i := range util.SContext.Values {
		util.SContext.Values[i] = make([]float64, N)
	}
	//ct generation

	values_arr := make([][]float64, sparseN)
	for j := range values_arr {
		values_arr[j] = make([]float64, N)
		for i := range values_arr[j] {
			if i%ratio == 0 {
				values_arr[j][i] = sampling.RandFloat64(-1, 1)
			}
		}
		fmt.Println(values_arr[j])
	}

	cts := make([]*rlwe.Ciphertext, sparseN)
	wg.Add(sparseN)
	for i := 0; i < sparseN; i++ {
		go func() {
			i := i
			defer wg.Done()
			encoder := hefloat.NewEncoder(params)
			encryptor := rlwe.NewEncryptor(params, pk)
			plaintext := hefloat.NewPlaintext(params, params.MaxLevel())

			plaintext.IsBatched = false
			encoder.Encode(values_arr[i], plaintext)
			cts[i], _ = encryptor.EncryptNew(plaintext)
			params.RingQ().AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
			params.RingQ().AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
		}()
	}
	wg.Wait()
	fmt.Println("ct gen end")
	// context.ModUp(cts, cts)
	ModUp(cts, params, encoder, encryptor, evaluator, N, sparseN, 32, cts)
	for i := range cts {
		vals := make([]float64, N)
		cttmp := cts[i].CopyNew()
		params.RingQ().AtLevel(cttmp.Level()).NTT(cttmp.Value[0], cttmp.Value[0])
		params.RingQ().AtLevel(cttmp.Level()).NTT(cttmp.Value[1], cttmp.Value[1])

		ptres := decryptor.DecryptNew(cttmp)
		encoder.Decode(ptres, vals)

		for j := range vals {
			util.SContext.Values[i][j] = vals[j]

		}
		fmt.Println(vals)
	}

}

func Test_CoeffToSlotBench(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU())

	logN := 10
	sparses := []int{8}
	lenCL := []int{2}
	// sparses := []int{5}
	// lenCL := []int{2}

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            logN,
		LogQ:            []int{48, 40, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52, 52, 52},
		LogDefaultScale: 40,
	}

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

	galLen := 1

	P := []uint32{3422539, 3370361, 3231143, 3545881, 3577031, 3832931, 4064197, 3617099, 3651497, 3711319, 3439693, 3502001, 3555509, 3552013, 4031179, 4115407, 3167453, 3365393, 3291143, 3204973, 4182419, 3495781, 3315883, 3403391, 3529153, 3390899, 3453773, 3705469, 3180337, 4091993, 3503221, 3598949, 3822277, 3277853, 3547249, 3278053, 3696257, 3849409, 3725257, 3239449, 3730721, 3393619, 3361363, 3732997, 3661573, 3158971, 3516031, 3737039, 3882649, 3614969, 3518491, 3169759, 3326417, 4165333, 3853097, 3845357, 3721603, 3494831, 3255467, 3442987, 3381641, 4188433, 3960053, 3825473, 3269713, 3373781, 3403843, 4177609, 3265337, 3382231, 3342137, 3330179, 3272629, 3725357, 3667453, 3960049, 3435323, 3664249, 3632423, 3515269, 3784733, 3377657, 4064143, 3702119, 3835367, 3564937, 3507397, 3345877, 4169129, 3206783, 3397769, 4145293, 3773477, 3229319, 3161617, 3517427, 3456743, 3687163, 3389423, 3553541}
	PLevel := 33
	// QPratio := new(big.Float).SetFloat64(1)
	// for i := range params.Q() {
	// 	QPratio.Mul(QPratio, new(big.Float).SetUint64(params.Q()[i]))
	// }
	// for i := range P[:PLevel+1] {
	// 	QPratio.Quo(QPratio, new(big.Float).SetUint64(P[i]))
	// }
	// fmt.Println(QPratio)

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
	// for i := range ratio {
	// 	k := N >> i
	// 	galEls = append(galEls, uint64(k+1))
	// }

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
	util.SContext = util.SecretContext{
		Sk:        sk,
		Decryptor: decryptor,
	}
	util.Debug = util.DebugContext{
		IsDebug:   true,
		AccTime:   time.Duration(0),
		StartTime: time.Time{},
	}

	for _, v := range sparses {
		sparseN := 1 << v
		ratio := N / sparseN
		// context := InitContext(params, encoder, encryptor, N, sparseN, evaluator, P, MatmultParamsLiteral{}, MatmultParamsLiteral{})
		util.SContext.Values = make([][]float64, sparseN)
		for i := range util.SContext.Values {
			util.SContext.Values[i] = make([]float64, N)
		}
		//ct generation

		values_arr := make([][]float64, sparseN)
		for j := range values_arr {
			values_arr[j] = make([]float64, N)
			for i := range values_arr[j] {
				if i%ratio == 0 {
					values_arr[j][i] = sampling.RandFloat64(-1, 1)
				}
			}
		}

		cts := make([]*rlwe.Ciphertext, sparseN)
		wg.Add(sparseN)
		for i := 0; i < sparseN; i++ {
			go func() {
				i := i
				defer wg.Done()
				encoder := hefloat.NewEncoder(params)
				encryptor := rlwe.NewEncryptor(params, pk)
				plaintext := hefloat.NewPlaintext(params, params.MaxLevel())

				plaintext.IsBatched = false
				encoder.Encode(values_arr[i], plaintext)
				cts[i], _ = encryptor.EncryptNew(plaintext)
				params.RingQ().AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
				params.RingQ().AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
			}()
		}
		wg.Wait()
		fmt.Println("ct gen end")
		// context.ModUp(cts, cts)
		ModUp(cts, params, encoder, encryptor, evaluator, N, sparseN, 32, cts)
		for i := range cts {
			vals := make([]float64, N)
			cttmp := cts[i].CopyNew()
			params.RingQ().AtLevel(cttmp.Level()).NTT(cttmp.Value[0], cttmp.Value[0])
			params.RingQ().AtLevel(cttmp.Level()).NTT(cttmp.Value[1], cttmp.Value[1])

			ptres := decryptor.DecryptNew(cttmp)
			encoder.Decode(ptres, vals)

			for j := range vals {
				util.SContext.Values[i][j] = vals[j]

			}
		}

		for idx, cl := range lenCL {
			if cl > v-1 {
				continue
			}
			l := bits.Len64(uint64(sparseN)) - 2
			CLS := CompositionsSumToN(l, cl)
			CLS = CLS[len(CLS)/2 : len(CLS)/2+1]

			for _, CLs := range CLS {
				CL_arr := CLs
				// pq, _ := QPratio.Float64()
				CTSParams := MatmultParamsLiteral{
					StartLevel: params.MaxLevel(),
					EndLevel:   params.MaxLevel(),
					PLevel:     PLevel,
					Scaling:    1,
					CL_arr:     CL_arr,
				}
				context := InitContext(params, encoder, encryptor, N, sparseN, evaluator, P, CTSParams, MatmultParamsLiteral{})
				// _, SFI := matmult.GenSFMat_CL2(params, sparseN>>1, nil, CTSParams.CL_arr)
				// fmt.Println("Mat Gen end")
				// context.ContextPreAlloc()
				// context.C2SParams = context.GenMatParams(CTSParams, SFI)
				fmt.Println("params alloc really end")
				runtime.GC()

				fmt.Println()
				fmt.Println()
				fmt.Println("***************************************************************************************************************************************************************************************************************************************************************")
				util.Debug.AccTime = 0
				util.Debug.StartTime = time.Time{}
				util.PrintMemUsage()
				CoeffToSlot_Testing(context, cts, idx == 0)
				fmt.Println("***************************************************************************************************************************************************************************************************************************************************************")
				fmt.Println()
				fmt.Println()
				// fmt.Println(cts[0].Level())
			}
		}
	}

}

func CoeffToSlot_Testing(context *Context, cts []*rlwe.Ciphertext, isOriginTest bool) {

	N := context.N
	sparseN := context.SparseN
	ratio := N / sparseN
	CL_arr := context.C2SParams.params.CL_arr
	params := context.params
	PLevel := context.C2SParams.params.PLevel
	encoder := context.Encoder
	decryptor := util.SContext.Decryptor
	sk := util.SContext.Sk

	fmt.Println("N, SparseN, Ratio, CL_Arr : ", N, ", ", sparseN, ", ", ratio, ", ", CL_arr)
	fmt.Println("MaxLevel, PLevel : ", params.MaxLevel(), PLevel)
	// fmt.Println("Q/P ratio : ", QPratio)
	fmt.Println()

	runtime.GC()
	runtime.GOMAXPROCS(1)
	result1, result2 := context.CoeffToSlot2(cts)
	fmt.Println()
	fmt.Println("#############################################################")
	fmt.Println("Our Total Elapse : ", util.Debug.AccTime)

	rescompare := make([][]float64, sparseN)
	for i := range rescompare {
		rescompare[i] = make([]float64, sparseN/2)
		for j := range rescompare[i] {
			rescompare[i][j] = util.SContext.Values[j][i*ratio]
		}
	}

	util.DebugPrec(result1[:sparseN/2], params, encoder, decryptor, rescompare, 1, true)

	for i := range rescompare {
		for j := range rescompare[i] {
			rescompare[i][j] = util.SContext.Values[sparseN/2+j][i*ratio]
		}
	}
	util.DebugPrec(result2[:sparseN/2], params, encoder, decryptor, rescompare, 1, true)
	fmt.Println("#############################################################")
	if !isOriginTest {
		return
	}

	runtime.GC()

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

	for i := range cts {
		result1[i].Resize(1, cts[i].Level())
		params.RingQ().AtLevel(cts[i].Level()).NTT(cts[i].Value[0], result1[i].Value[0])
		params.RingQ().AtLevel(cts[i].Level()).NTT(cts[i].Value[1], result1[i].Value[1])
	}

	starttime := time.Now()
	for i := range cts {
		result1[i], result2[i], err = btp.CoeffsToSlots(result1[i])
		if err != nil {
			panic(err)
		}
	}
	elapse := time.Since(starttime)
	fmt.Println()
	fmt.Println("#############################################################")
	fmt.Println("Original Total Elapse", elapse)

	bitlen := params.LogN() - 1
	rescompare = make([][]float64, sparseN)
	sc, _ := btp.CoeffsToSlotsParameters.Scaling.Float64()
	for i := range rescompare {
		rescompare[i] = make([]float64, sparseN/2)
		for j := range rescompare[i] {

			rescompare[i][j] = util.SContext.Values[i][bitReverse(j*ratio, bitlen)] * sc
		}
	}
	util.DebugPrec(result1, params, encoder, decryptor, rescompare, ratio, true)

	for i := range rescompare {
		rescompare[i] = make([]float64, sparseN/2)
		for j := range rescompare[i] {
			rescompare[i][j] = util.SContext.Values[i][N/2+bitReverse(j*ratio, bitlen)] * sc
		}
	}
	util.DebugPrec(result2, params, encoder, decryptor, rescompare, ratio, true)
	fmt.Println("#############################################################")
}

func Test_CoeffToSlot(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU())

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            5,
		LogQ:            []int{48, 40, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}

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
	sparseN := N >> 0
	ratio := N / sparseN
	CL_arr := []int{2, 2}

	galLen := 1

	P := []uint32{3422539, 3370361, 3231143, 3545881, 3577031, 3832931, 4064197, 3617099, 3651497, 3711319, 3439693, 3502001, 3555509, 3552013, 4031179, 4115407, 3167453, 3365393, 3291143, 3204973, 4182419, 3495781, 3315883, 3403391, 3529153, 3390899, 3453773, 3705469, 3180337, 4091993, 3503221, 3598949, 3822277, 3277853, 3547249, 3278053, 3696257, 3849409, 3725257, 3239449, 3730721, 3393619, 3361363, 3732997, 3661573, 3158971, 3516031, 3737039, 3882649, 3614969, 3518491, 3169759, 3326417, 4165333, 3853097, 3845357, 3721603, 3494831, 3255467, 3442987, 3381641, 4188433, 3960053, 3825473, 3269713, 3373781, 3403843, 4177609, 3265337, 3382231, 3342137, 3330179, 3272629, 3725357, 3667453, 3960049, 3435323, 3664249, 3632423, 3515269, 3784733, 3377657, 4064143, 3702119, 3835367, 3564937, 3507397, 3345877, 4169129, 3206783, 3397769, 4145293, 3773477, 3229319, 3161617, 3517427, 3456743, 3687163, 3389423, 3553541}
	PLevel := 33

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
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
	util.SContext = util.SecretContext{
		Sk:        sk,
		Decryptor: decryptor,
	}
	util.Debug = util.DebugContext{
		IsDebug:   true,
		AccTime:   time.Duration(0),
		StartTime: time.Time{},
	}
	CTSParams := MatmultParamsLiteral{
		StartLevel: params.MaxLevel(),
		EndLevel:   params.MaxLevel(),
		PLevel:     PLevel,
		Scaling:    1,
		CL_arr:     CL_arr,
	}
	context := InitContext(params, encoder, encryptor, N, sparseN, evaluator, P, CTSParams, MatmultParamsLiteral{})
	// util.SContext.Values = make([][]float64, sparseN)
	// for i := range util.SContext.Values {
	// 	util.SContext.Values[i] = make([]float64, N)
	// }
	//ct generation

	values_arr := make([][]float64, sparseN)
	for j := range values_arr {
		values_arr[j] = make([]float64, N)
		for i := range values_arr[j] {
			if i%ratio == 0 {
				values_arr[j][i] = sampling.RandFloat64(-1, 1)
				// values_arr[j][j] = 0.001*float64(i*i) - float64(j)*0.02
			}
		}
		fmt.Println(values_arr[j])
	}
	fmt.Println()
	cts := make([]*rlwe.Ciphertext, sparseN)
	wg.Add(sparseN)
	for i := 0; i < sparseN; i++ {
		go func() {
			i := i
			defer wg.Done()
			encoder := hefloat.NewEncoder(params)
			encryptor := rlwe.NewEncryptor(params, pk)
			plaintext := hefloat.NewPlaintext(params, params.MaxLevel())

			plaintext.IsBatched = false
			encoder.Encode(values_arr[i], plaintext)
			cts[i], _ = encryptor.EncryptNew(plaintext)
			params.RingQ().AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
			params.RingQ().AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
		}()
	}
	wg.Wait()

	fmt.Println("ct gen end")
	fmt.Println("N, SparseN, Ratio, CL_Arr : ", N, ", ", sparseN, ", ", ratio, ", ", CL_arr)
	fmt.Println("MaxLevel, PLevel : ", params.MaxLevel(), PLevel)
	// fmt.Println("Q/P ratio : ", QPratio)
	fmt.Println()

	runtime.GC()
	runtime.GOMAXPROCS(1)
	result1, result2 := context.CoeffToSlot2(cts)
	fmt.Println()
	fmt.Println("#############################################################")
	fmt.Println("Our Total Elapse : ", util.Debug.AccTime)
	res := make([]float64, N/2)
	for i := range result1 {
		result1[i].IsBatched = true
		ptres := decryptor.DecryptNew(result1[i])
		encoder.Decode(ptres, res)
		fmt.Println(res)
	}
	_ = result2
	fmt.Println()
	for i := range result2 {
		result2[i].IsBatched = true
		ptres := decryptor.DecryptNew(result2[i])
		encoder.Decode(ptres, res)
		fmt.Println(res)
	}
	fmt.Println()
}

func Test_SlotToCoeff(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU())

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            10,
		LogQ:            []int{48, 40, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}

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
	sparseN := N >> 2
	ratio := N / sparseN
	CL_arr := []int{7}

	galLen := 1

	P := []uint32{3422539, 3370361, 3231143, 3545881, 3577031, 3832931, 4064197, 3617099, 3651497, 3711319, 3439693, 3502001, 3555509, 3552013, 4031179, 4115407, 3167453, 3365393, 3291143, 3204973, 4182419, 3495781, 3315883, 3403391, 3529153, 3390899, 3453773, 3705469, 3180337, 4091993, 3503221, 3598949, 3822277, 3277853, 3547249, 3278053, 3696257, 3849409, 3725257, 3239449, 3730721, 3393619, 3361363, 3732997, 3661573, 3158971, 3516031, 3737039, 3882649, 3614969, 3518491, 3169759, 3326417, 4165333, 3853097, 3845357, 3721603, 3494831, 3255467, 3442987, 3381641, 4188433, 3960053, 3825473, 3269713, 3373781, 3403843, 4177609, 3265337, 3382231, 3342137, 3330179, 3272629, 3725357, 3667453, 3960049, 3435323, 3664249, 3632423, 3515269, 3784733, 3377657, 4064143, 3702119, 3835367, 3564937, 3507397, 3345877, 4169129, 3206783, 3397769, 4145293, 3773477, 3229319, 3161617, 3517427, 3456743, 3687163, 3389423, 3553541}
	PLevel := 33

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
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
	util.SContext = util.SecretContext{
		Sk:        sk,
		Decryptor: decryptor,
	}
	util.Debug = util.DebugContext{
		IsDebug:   true,
		AccTime:   time.Duration(0),
		StartTime: time.Time{},
	}
	STCParams := MatmultParamsLiteral{
		StartLevel: params.MaxLevel(),
		EndLevel:   params.MaxLevel(),
		PLevel:     PLevel,
		Scaling:    1,
		CL_arr:     CL_arr,
	}
	context := InitContext(params, encoder, encryptor, N, sparseN, evaluator, P, MatmultParamsLiteral{}, STCParams)

	values_arr := make([][]complex128, sparseN)
	for j := range values_arr {
		values_arr[j] = make([]complex128, N/2)
		for i := range values_arr[j] {
			values_arr[j][i] = complex(float64(i%(sparseN/2))*0.01, -float64(+i%(sparseN/2))*0.01)
		}
		fmt.Println(values_arr[j])
	}

	cts := make([]*rlwe.Ciphertext, sparseN)
	wg.Add(sparseN)
	for i := 0; i < sparseN; i++ {
		go func() {
			i := i
			defer wg.Done()
			encoder := hefloat.NewEncoder(params)
			encryptor := rlwe.NewEncryptor(params, pk)
			plaintext := hefloat.NewPlaintext(params, params.MaxLevel())

			plaintext.IsBatched = true
			encoder.Encode(values_arr[i], plaintext)
			cts[i], _ = encryptor.EncryptNew(plaintext)
		}()
	}
	wg.Wait()
	fmt.Println("ct gen end")

	fmt.Println("N, SparseN, Ratio, CL_Arr : ", N, ", ", sparseN, ", ", ratio, ", ", CL_arr)
	fmt.Println("MaxLevel, PLevel : ", params.MaxLevel(), PLevel)
	// fmt.Println("Q/P ratio : ", QPratio)
	fmt.Println()

	runtime.GC()
	runtime.GOMAXPROCS(1)
	result1 := context.SlotToCoeff2(cts, nil)
	fmt.Println(result1[0].LogScale())
	fmt.Println()
	fmt.Println("#############################################################")
	fmt.Println("Our Total Elapse : ", util.Debug.AccTime)

	res := make([]float64, N)
	for i := range result1 {
		result1[i].IsBatched = false
		ptres := decryptor.DecryptNew(result1[i])
		encoder.Decode(ptres, res)
		fmt.Println(res)
	}
	fmt.Println()
}

func Test_CTSSTC(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU())

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            10,
		LogQ:            []int{48, 40, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}

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
	sparseN := N >> 2
	ratio := N / sparseN
	CL_arr := []int{7}

	galLen := 1

	P := []uint32{3422539, 3370361, 3231143, 3545881, 3577031, 3832931, 4064197, 3617099, 3651497, 3711319, 3439693, 3502001, 3555509, 3552013, 4031179, 4115407, 3167453, 3365393, 3291143, 3204973, 4182419, 3495781, 3315883, 3403391, 3529153, 3390899, 3453773, 3705469, 3180337, 4091993, 3503221, 3598949, 3822277, 3277853, 3547249, 3278053, 3696257, 3849409, 3725257, 3239449, 3730721, 3393619, 3361363, 3732997, 3661573, 3158971, 3516031, 3737039, 3882649, 3614969, 3518491, 3169759, 3326417, 4165333, 3853097, 3845357, 3721603, 3494831, 3255467, 3442987, 3381641, 4188433, 3960053, 3825473, 3269713, 3373781, 3403843, 4177609, 3265337, 3382231, 3342137, 3330179, 3272629, 3725357, 3667453, 3960049, 3435323, 3664249, 3632423, 3515269, 3784733, 3377657, 4064143, 3702119, 3835367, 3564937, 3507397, 3345877, 4169129, 3206783, 3397769, 4145293, 3773477, 3229319, 3161617, 3517427, 3456743, 3687163, 3389423, 3553541}
	PLevel := 33

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
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
	util.SContext = util.SecretContext{
		Sk:        sk,
		Decryptor: decryptor,
	}
	util.Debug = util.DebugContext{
		IsDebug:   true,
		AccTime:   time.Duration(0),
		StartTime: time.Time{},
	}
	CTSParams := MatmultParamsLiteral{
		StartLevel: params.MaxLevel(),
		EndLevel:   params.MaxLevel(),
		PLevel:     PLevel,
		Scaling:    1,
		CL_arr:     CL_arr,
	}
	STCParams := MatmultParamsLiteral{
		StartLevel: params.MaxLevel() - 1,
		EndLevel:   params.MaxLevel() - 1,
		PLevel:     PLevel - 0,
		Scaling:    1,
		CL_arr:     CL_arr,
	}
	context := InitContext(params, encoder, encryptor, N, sparseN, evaluator, P, CTSParams, STCParams)
	values_arr := make([][]float64, sparseN)
	for j := range values_arr {
		values_arr[j] = make([]float64, N)
		for i := range values_arr[j] {
			if i%ratio == 0 {
				values_arr[j][i] = sampling.RandFloat64(-1, 1)
				// values_arr[j][j] = 0.001*float64(i*i) - float64(j)*0.02
			}
		}
		fmt.Println(values_arr[j])
	}
	fmt.Println()
	cts := make([]*rlwe.Ciphertext, sparseN)
	wg.Add(sparseN)
	for i := 0; i < sparseN; i++ {
		go func() {
			i := i
			defer wg.Done()
			encoder := hefloat.NewEncoder(params)
			encryptor := rlwe.NewEncryptor(params, pk)
			plaintext := hefloat.NewPlaintext(params, params.MaxLevel())

			plaintext.IsBatched = false
			encoder.Encode(values_arr[i], plaintext)
			cts[i], _ = encryptor.EncryptNew(plaintext)
			params.RingQ().AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
			params.RingQ().AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
		}()
	}
	wg.Wait()

	fmt.Println("ct gen end")
	fmt.Println("N, SparseN, Ratio, CL_Arr : ", N, ", ", sparseN, ", ", ratio, ", ", CL_arr)
	fmt.Println("MaxLevel, PLevel : ", params.MaxLevel(), PLevel)
	// fmt.Println("Q/P ratio : ", QPratio)
	fmt.Println()

	runtime.GC()
	runtime.GOMAXPROCS(1)
	result1, result2 := context.CoeffToSlot2(cts)
	for i := range context.alloced.aux {
		context.alloced.aux[i] = context.alloced.ctzero.CopyNew()
		context.alloced.work[i] = context.alloced.ctzero.CopyNew()
	}
	result := context.SlotToCoeff2(result1, result2)

	fmt.Println()
	fmt.Println("#############################################################")
	fmt.Println("Our Total Elapse : ", util.Debug.AccTime)
	res := make([]float64, N/2)
	for i := range result {
		result[i].IsBatched = false
		ptres := decryptor.DecryptNew(result[i])
		encoder.Decode(ptres, res)
		fmt.Println(res)
	}

	util.DebugPrec(result, params, encoder, decryptor, values_arr, 1, false)
}

func Test_Boot(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU())

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            10,
		LogQ:            []int{48, 60, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 48, 60, 60},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}

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
	sparseN := N >> 2
	ratio := N / sparseN
	CL_arr := []int{4, 3}

	galLen := 1

	P := []uint32{3422539, 3370361, 3231143, 3545881, 3577031, 3832931, 4064197, 3617099, 3651497, 3711319, 3439693, 3502001, 3555509, 3552013, 4031179, 4115407, 3167453, 3365393, 3291143, 3204973, 4182419, 3495781, 3315883, 3403391, 3529153, 3390899, 3453773, 3705469, 3180337, 4091993, 3503221, 3598949, 3822277, 3277853, 3547249, 3278053, 3696257, 3849409, 3725257, 3239449, 3730721, 3393619, 3361363, 3732997, 3661573, 3158971, 3516031, 3737039, 3882649, 3614969, 3518491, 3169759, 3326417, 4165333, 3853097, 3845357, 3721603, 3494831, 3255467, 3442987, 3381641, 4188433, 3960053, 3825473, 3269713, 3373781, 3403843, 4177609, 3265337, 3382231, 3342137, 3330179, 3272629, 3725357, 3667453, 3960049, 3435323, 3664249, 3632423, 3515269, 3784733, 3377657, 4064143, 3702119, 3835367, 3564937, 3507397, 3345877, 4169129, 3206783, 3397769, 4145293, 3773477, 3229319, 3161617, 3517427, 3456743, 3687163, 3389423, 3553541}
	PLevel := 40

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
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
		CircuitOrder:            bootstrapping.ModUpThenEncode,
	}
	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)
	btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	if err != nil {
		panic(err)
	}
	_ = btp
	C2SScale, _ := btp.CoeffsToSlotsParameters.Scaling.Float64()
	_ = C2SScale

	util.SContext = util.SecretContext{
		Sk:        sk,
		Decryptor: decryptor,
	}
	util.Debug = util.DebugContext{
		IsDebug:   true,
		AccTime:   time.Duration(0),
		StartTime: time.Time{},
	}
	CTSParams := MatmultParamsLiteral{
		StartLevel: params.MaxLevel(),
		EndLevel:   params.MaxLevel(),
		PLevel:     PLevel,
		Scaling:    C2SScale,
		CL_arr:     CL_arr,
	}
	S2CScale, _ := btp.SlotsToCoeffsParameters.Scaling.Float64()
	STCParams := MatmultParamsLiteral{
		StartLevel: params.MaxLevel() - 10,
		EndLevel:   params.MaxLevel() - 10,
		PLevel:     PLevel,
		Scaling:    S2CScale,
		CL_arr:     CL_arr,
	}
	context := InitContext(params, encoder, encryptor, N, sparseN, evaluator, P, CTSParams, STCParams)
	_ = context
	values_arr := make([][]float64, sparseN)
	for j := range values_arr {
		values_arr[j] = make([]float64, N)
		for i := range values_arr[j] {
			if i%ratio == 0 {
				values_arr[j][i] = sampling.RandFloat64(-1, 1)
				// values_arr[j][i] = float64(j) * 0.02
			}
		}
		// fmt.Println(values_arr[j])
	}
	fmt.Println()
	cts := make([]*rlwe.Ciphertext, sparseN)
	wg.Add(sparseN)
	for i := 0; i < sparseN; i++ {
		go func() {
			i := i
			defer wg.Done()
			encoder := hefloat.NewEncoder(params)
			encryptor := rlwe.NewEncryptor(params, pk)
			plaintext := hefloat.NewPlaintext(params, 2)

			plaintext.IsBatched = false
			encoder.Encode(values_arr[i], plaintext)
			cts[i], _ = encryptor.EncryptNew(plaintext)
			params.RingQ().AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
			params.RingQ().AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
		}()
	}
	wg.Wait()

	fmt.Println("ct gen end")
	fmt.Println("N, SparseN, Ratio, CL_Arr : ", N, ", ", sparseN, ", ", ratio, ", ", CL_arr)
	fmt.Println("MaxLevel, PLevel : ", params.MaxLevel(), PLevel)
	ringP, err := matmult.NewRing(params.N(), P)
	if err != nil {
		panic(err)
	}
	QP := new(big.Float).SetInt(params.RingQ().ModulusAtLevel[params.MaxLevel()])
	QP = QP.Quo(QP, new(big.Float).SetInt(ringP.ModulusAtLevel[PLevel]))
	fmt.Println("Q/P ratio 1 : ", QP)
	QP = new(big.Float).SetInt(params.RingQ().ModulusAtLevel[CTSParams.StartLevel])
	QP = QP.Quo(QP, new(big.Float).SetInt(ringP.ModulusAtLevel[STCParams.PLevel]))
	fmt.Println("Q/P ratio 2 : ", QP)
	fmt.Println()
	res := make([]float64, N)

	runtime.GC()
	runtime.GOMAXPROCS(1)
	result := make([]*rlwe.Ciphertext, sparseN)
	for i := range result {
		result[i] = cts[i].CopyNew()
	}
	ModUp(result, params, encoder, encryptor, evaluator, N, sparseN, 32, result)
	result1, result2 := context.CoeffToSlot2(result)
	for i := range result1 {
		result1[i], err = btp.EvalMod(result1[i])
		result2[i], _ = btp.EvalMod(result2[i])
		if err != nil {
			panic(err)
		}
	}
	result1 = context.SlotToCoeff2(result1, result2)

	fmt.Println()
	fmt.Println("#############################################################")
	fmt.Println("Our Total Elapse : ", util.Debug.AccTime)

	for i := range result1 {
		result1[i].IsBatched = false
		ptres := decryptor.DecryptNew(result1[i])
		encoder.Decode(ptres, res)
		// fmt.Println(res)
	}
	util.DebugPrec(result1, params, encoder, decryptor, values_arr, 1, false)
	// util.DebugPrec(result2, params, encoder, decryptor, values_arr, 1, false)

	for i := range cts {
		params.RingQ().AtLevel(cts[i].Level()).NTT(cts[i].Value[0], cts[i].Value[0])
		params.RingQ().AtLevel(cts[i].Level()).NTT(cts[i].Value[1], cts[i].Value[1])
	}

	starttime := time.Now()
	for i := range cts {
		cts[i], _ = btp.Bootstrap(cts[i])
	}
	elapse := time.Since(starttime)
	fmt.Println("#############################################################")
	fmt.Println("Origin Total Elapse : ", elapse)
	for i := range cts {
		cts[i].IsBatched = false
		ptres := decryptor.DecryptNew(cts[i])
		encoder.Decode(ptres, res)
		// fmt.Println(res)
	}
	util.DebugPrec(cts, params, encoder, decryptor, values_arr, 1, false)
}

func Test_BootBench(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU())

	logN := 15
	sparses := []int{10, 11, 12, 13}
	lenCL := []int{2}
	k := 1
	// sparses := []int{5}
	// lenCL := []int{2}

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            logN,
		LogQ:            []int{48, 48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}

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

	galLen := 1

	P := []uint32{3422539, 3370361, 3231143, 3545881, 3577031, 3832931, 4064197, 3617099, 3651497, 3711319, 3439693, 3502001, 3555509, 3552013, 4031179, 4115407, 3167453, 3365393, 3291143, 3204973, 4182419, 3495781, 3315883, 3403391, 3529153, 3390899, 3453773, 3705469, 3180337, 4091993, 3503221, 3598949, 3822277, 3277853, 3547249, 3278053, 3696257, 3849409, 3725257, 3239449, 3730721, 3393619, 3361363, 3732997, 3661573, 3158971, 3516031, 3737039, 3882649, 3614969, 3518491, 3169759, 3326417, 4165333, 3853097, 3845357, 3721603, 3494831, 3255467, 3442987, 3381641, 4188433, 3960053, 3825473, 3269713, 3373781, 3403843, 4177609, 3265337, 3382231, 3342137, 3330179, 3272629, 3725357, 3667453, 3960049, 3435323, 3664249, 3632423, 3515269, 3784733, 3377657, 4064143, 3702119, 3835367, 3564937, 3507397, 3345877, 4169129, 3206783, 3397769, 4145293, 3773477, 3229319, 3161617, 3517427, 3456743, 3687163, 3389423, 3553541}
	PLevel := 40
	P = P[:PLevel+1]
	// QPratio := new(big.Float).SetFloat64(1)
	// for i := range params.Q() {
	// 	QPratio.Mul(QPratio, new(big.Float).SetUint64(params.Q()[i]))
	// }
	// for i := range P[:PLevel+1] {
	// 	QPratio.Quo(QPratio, new(big.Float).SetUint64(P[i]))
	// }
	// fmt.Println(QPratio)

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
	// for i := range ratio {
	// 	k := N >> i
	// 	galEls = append(galEls, uint64(k+1))
	// }

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
	btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	C2SScale, _ := btp.CoeffsToSlotsParameters.Scaling.Float64()
	if err != nil {
		panic(err)
	}
	util.SContext = util.SecretContext{
		Sk:        sk,
		Decryptor: decryptor,
	}
	util.Debug = util.DebugContext{
		IsDebug:   true,
		AccTime:   time.Duration(0),
		StartTime: time.Time{},
	}

	for _, v := range sparses {
		sparseN := 1 << v
		ratio := N / sparseN
		values_arr := make([][]float64, sparseN)
		for j := range values_arr {
			values_arr[j] = make([]float64, N)
			for i := range values_arr[j] {
				if i%ratio == 0 {
					values_arr[j][i] = sampling.RandFloat64(-1, 1)
				}
			}
		}
		util.SContext.Values = values_arr

		cts := make([]*rlwe.Ciphertext, sparseN)
		wg.Add(sparseN)
		for i := 0; i < sparseN; i++ {
			go func() {
				i := i
				defer wg.Done()
				encoder := hefloat.NewEncoder(params)
				encryptor := rlwe.NewEncryptor(params, pk)
				plaintext := hefloat.NewPlaintext(params, params.MaxLevel())

				plaintext.IsBatched = false
				encoder.Encode(values_arr[i], plaintext)
				cts[i], _ = encryptor.EncryptNew(plaintext)
				params.RingQ().AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
				params.RingQ().AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
			}()
		}
		wg.Wait()
		fmt.Println("ct gen end")

		for idx, cl := range lenCL {
			if cl > v-1 {
				continue
			}
			l := bits.Len64(uint64(sparseN)) - 2
			CLS := CompositionsSumToN2(l, cl, k)

			for _, CLs := range CLS {
				CL_arr := CLs
				// pq, _ := QPratio.Float64()
				CTSParams := MatmultParamsLiteral{
					StartLevel: params.MaxLevel(),
					EndLevel:   params.MaxLevel(),
					PLevel:     PLevel,
					Scaling:    C2SScale,
					CL_arr:     CL_arr,
				}
				STCParams := MatmultParamsLiteral{
					StartLevel: params.MaxLevel() - 10,
					EndLevel:   params.MaxLevel() - 10,
					PLevel:     PLevel,
					Scaling:    1,
					CL_arr:     CL_arr,
				}
				context := InitContext(params, encoder, encryptor, N, sparseN, evaluator, P, CTSParams, STCParams)
				fmt.Println("params alloc really end")
				runtime.GC()

				fmt.Println()
				fmt.Println()
				fmt.Println("***************************************************************************************************************************************************************************************************************************************************************")
				util.Debug.AccTime = 0
				util.Debug.StartTime = time.Time{}
				util.PrintMemUsage()
				Boot_Testing(context, cts, btp, idx == 99)
				fmt.Println("***************************************************************************************************************************************************************************************************************************************************************")
				fmt.Println()
				fmt.Println()
				// fmt.Println(cts[0].Level())
			}
		}
	}

}

func Boot_Testing(context *Context, cts []*rlwe.Ciphertext, btp *bootstrapping.Evaluator, isOriginTest bool) {
	N := context.N
	sparseN := context.SparseN
	ratio := N / sparseN
	CL_arr := context.C2SParams.params.CL_arr
	params := context.params
	PLevel := context.C2SParams.params.PLevel
	encoder := context.Encoder
	evaluator := context.Evaluator
	encryptor := context.Encryptor
	decryptor := util.SContext.Decryptor
	var err error

	fmt.Println("N, SparseN, Ratio, CL_Arr : ", N, ", ", sparseN, ", ", ratio, ", ", CL_arr)
	fmt.Println("MaxLevel, PLevel : ", params.MaxLevel(), PLevel)
	fmt.Println()

	// for i := range cts {
	// 	evaluator.DropLevel(cts[i], cts[i].Level()-len(CL_arr))
	// }

	runtime.GC()
	runtime.GOMAXPROCS(1)
	result := make([]*rlwe.Ciphertext, sparseN)
	for i := range result {
		result[i] = cts[i].CopyNew()
	}
	ModUp(result, params, encoder, encryptor, evaluator, N, sparseN, 32, result)
	result1, result2 := context.CoeffToSlot2(result)
	starttime := time.Now()
	for i := range result1 {
		result1[i], err = btp.EvalMod(result1[i])
		result2[i], _ = btp.EvalMod(result2[i])
		if err != nil {
			panic(err)
		}
	}
	elapse := time.Since(starttime)
	fmt.Println("ModEval : ", elapse)
	util.Debug.AccTime += elapse
	result1 = context.SlotToCoeff2(result1, result2)

	fmt.Println()
	fmt.Println("#############################################################")
	fmt.Println("Our Total Elapse : ", util.Debug.AccTime)

	util.DebugPrec(result1, params, encoder, decryptor, util.SContext.Values, 1, false)
	fmt.Println("#############################################################")
	if !isOriginTest {
		return
	}

	runtime.GC()

	for i := range cts {
		result1[i].Resize(1, cts[i].Level())
		params.RingQ().AtLevel(cts[i].Level()).NTT(cts[i].Value[0], result1[i].Value[0])
		params.RingQ().AtLevel(cts[i].Level()).NTT(cts[i].Value[1], result1[i].Value[1])
	}

	starttime = time.Now()
	for i := range cts {
		result1[i], err = btp.Bootstrap(result1[i])
		if err != nil {
			panic(err)
		}
	}
	elapse = time.Since(starttime)
	fmt.Println()
	fmt.Println("#############################################################")
	fmt.Println("Original Total Elapse", elapse)
	util.DebugPrec(result1, params, encoder, decryptor, util.SContext.Values, 1, false)
	fmt.Println("#############################################################")
}

func Test_Boottime(t *testing.T) {

	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            15,
		LogQ:            []int{48, 40, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52, 52},
		LogDefaultScale: 40,
	}
	params, _ := hefloat.NewParametersFromLiteral(SchemeParams)

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

	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	n := 1 << params.LogMaxSlots()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, 1)
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
	_, _ = evaluator, decryptor

	btpevk, _, _ := btpParams.GenEvaluationKeys(sk)
	_ = decryptor
	_ = evaluator
	btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	if err != nil {
		panic(err)
	}
	_ = btp

	fmt.Println("generate Evaluator end")

	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.00001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, 0)
	pt.IsBatched = false
	encoder.Encode(value, pt)

	ct, _ := encryptor.EncryptNew(pt)
	runtime.GOMAXPROCS(1)
	starttime = time.Now()
	ct, _ = btp.Bootstrap(ct)
	elapse = time.Since(starttime)
	fmt.Println(elapse)

	dept := decryptor.DecryptNew(ct)
	value2 := make([]float64, 2*n)
	encoder.Decode(dept, value2)
	fmt.Println(value2[:20])

	fmt.Println(value[:20])
	util.DebugPrec([]*rlwe.Ciphertext{ct}, params, encoder, decryptor, [][]float64{value}, 1, false)
}

func bitReverse(i, m int) int {
	rev := 0
	for j := 0; j < m; j++ {
		rev = (rev << 1) | (i & 1)
		i >>= 1
	}
	return rev
}
func CompositionsSumToN2(N, X, k int) [][]int {
	arr := CompositionsSumToN(N, X)

	l := len(arr)
	if l <= k {
		return arr
	}
	return arr[(l-k)/2 : (l+k)/2]
}

func CompositionsSumToN(N, X int) [][]int {
	if X <= 0 {
		if N == 0 {
			return [][]int{{}}
		}
		return nil
	}
	if N < X { // 최소합은 X (모두 1)
		return nil
	}

	var res [][]int
	cur := make([]int, X)

	var dfs func(pos, remaining int)
	dfs = func(pos, remaining int) {
		if pos == X-1 {
			if remaining >= 1 {
				cur[pos] = remaining
				tmp := make([]int, X)
				copy(tmp, cur)
				res = append(res, tmp)
			}
			return
		}

		// 남은 칸 수를 고려해서 선택 가능한 최대값 계산
		slotsLeft := (X - 1) - pos
		maxVal := remaining - slotsLeft
		for v := 1; v <= maxVal; v++ {
			cur[pos] = v
			dfs(pos+1, remaining-v)
		}
	}

	dfs(0, N)
	return res
}
func Test_CompositionsSumToN(t *testing.T) {
	fmt.Println(CompositionsSumToN(4, 3))
}

func Test_MemCheck(t *testing.T) {

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{48, 40, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}

	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}
	fmt.Println("ckks parameter init end")
	sparseN := 1 << 13
	CL_arr := []int{6, 6}
	complex, complex2 := matmult.GenSFMat_CL3(params, sparseN>>1, CL_arr, CL_arr)
	util.PrintMemUsage()
	fmt.Println(complex[0][0], complex2[0][1])
	complex, complex2 = nil, nil
	runtime.GC()
	util.PrintMemUsage()
	fmt.Println("end")
}
