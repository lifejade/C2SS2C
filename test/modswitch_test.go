package test

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/lifejade/mm/src/matmult"
	"github.com/lifejade/mm/src/util"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

func Test_ModSwitch(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            16,
		LogQ:            []int{48, 48, 48},
		LogP:            []int{50},
		Xs:              ring.Ternary{H: 256},
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

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	n := params.N()
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
		i := i
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

	fmt.Println("N is this  : ", n)
	value := make([]float64, n)
	for i, _ := range value {
		value[i] = 0.1
	}

	u := make([][][]uint64, 7)
	for i := range u {
		u[i] = make([][]uint64, 32)
		for j := range u[i] {
			u[i][j] = make([]uint64, 32)
			for k := range u[i][j] {
				u[i][j][k] = 1
			}
		}
	}

	pt := hefloat.NewPlaintext(params, 2)
	pt.IsBatched = false
	//P := []uint64{23068673, 27918337, 20316161, 28704769, 28311553, 22806529}
	P := []uint64{12451841, 14155777, 13631489, 16384001, 13238273, 8650753, 16121857}

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	ringQ := params.RingQ().AtLevel(ct.Level())
	ringP, _ := ring.NewRing(params.N(), P)

	be := ring.NewBasisExtender(ringQ, ringP)

	// cts := make([]*rlwe.Ciphertext, 2)
	// ctIn := ct.CopyNew()
	// ringQ.INTT(ctIn.Value[0], ctIn.Value[0])
	// ringQ.INTT(ctIn.Value[1], ctIn.Value[1])

	// for idx := range 2 {
	// 	rings := ringP.NewPoly()
	// 	be.ModUpQtoP(2, 5, ctIn.Value[idx], rings)
	// 	fmt.Println(ctIn.Value[idx].Coeffs)
	// 	fmt.Println(rings.Coeffs)
	// 	for range 1 {
	// 		ringP.Add(rings, rings, rings)
	// 	}
	// 	fmt.Println(rings.Coeffs)
	// 	be.ModUpPtoQ(5, 2, rings, ctIn.Value[idx])
	// 	fmt.Println(ctIn.Value[idx].Coeffs)
	// }

	// ringQ.NTT(ctIn.Value[0], ctIn.Value[0])
	// ringQ.NTT(ctIn.Value[1], ctIn.Value[1])

	// values := make([]float64, n)

	// dept := decryptor.DecryptNew(ctIn)
	// encoder.Decode(dept, values)

	// fmt.Println("non-")
	// fmt.Println(values)
	// fmt.Println()

	_ = u
	rings := make([]ring.Poly, 32)
	for i := range rings {
		rings[i] = ringP.NewPoly()
	}

	res := make([]ring.Poly, 32)
	for i := range rings {
		res[i] = ringP.NewPoly()
	}
	ctIn := ct.CopyNew()
	ringQ.INTT(ctIn.Value[0], ctIn.Value[0])
	ringQ.INTT(ctIn.Value[1], ctIn.Value[1])

	starttime = time.Now()
	for idx := range 2 {
		fmt.Println("check1")
		fmt.Println(ctIn.Value[idx].Coeffs)

		for i := range rings {
			be.ModUpQtoP(2, 6, ctIn.Value[idx], rings[i])
			//ringP.Reduce(rings[i], rings[i])

		}

		matmult.PPMM_Blas_CRT(rings, u, params, 32, 32, params.N(), 7, ringP, rings)
		// fmt.Println("ccheck")
		// fmt.Println(rings[0].Coeffs)
		// fmt.Println(res[0].Coeffs)

		be.ModUpPtoQ(6, 2, rings[0], ctIn.Value[idx])
		//ringQ.Reduce(ctIn.Value[idx], ctIn.Value[idx])
		fmt.Println("check2")

		fmt.Println(ctIn.Value[idx].Coeffs)
	}
	elapse = time.Since(starttime)
	fmt.Println(elapse)

	ringQ.NTT(ctIn.Value[0], ctIn.Value[0])
	ringQ.NTT(ctIn.Value[1], ctIn.Value[1])

	values := make([]float64, n)

	dept := decryptor.DecryptNew(ctIn)
	encoder.Decode(dept, values)

	fmt.Println(values)

	fmt.Println(ringQ.ModulusAtLevel[2])
	fmt.Println(ringP.ModulusAtLevel[5])
}

func Test_ModSwitchTime(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            16,
		LogQ:            []int{48, 48, 48},
		LogP:            []int{50},
		Xs:              ring.Ternary{H: 256},
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

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	n := params.N()
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
		i := i
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

	fmt.Println("N is this  : ", n)
	value := make([]float64, n)
	for i, _ := range value {
		value[i] = 0.1
	}

	size := 1 << 5
	scale := float64(1 << 5)
	u := make([][][]uint64, 7)
	for i := range u {
		u[i] = make([][]uint64, size)
		for j := range u[i] {
			u[i][j] = make([]uint64, size)
			for k := range u[i][j] {
				u[i][j][k] = uint64(0.2 * scale)
			}
		}
	}

	pt := hefloat.NewPlaintext(params, 2)
	pt.IsBatched = false
	//P := []uint64{23068673, 27918337, 20316161, 28704769, 28311553, 22806529}
	P := []uint64{12451841, 14155777, 13631489, 16384001, 13238273, 8650753, 16121857}

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	ringQ := params.RingQ().AtLevel(ct.Level())
	ringP, _ := ring.NewRing(params.N(), P)

	be := ring.NewBasisExtender(ringQ, ringP)

	rings := make([]ring.Poly, size)
	for i := range rings {
		rings[i] = ringP.NewPoly()
	}

	res := make([]ring.Poly, size)
	for i := range rings {
		res[i] = ringP.NewPoly()
	}
	ctIn := ct.CopyNew()
	ringQ.INTT(ctIn.Value[0], ctIn.Value[0])
	ringQ.INTT(ctIn.Value[1], ctIn.Value[1])

	starttime = time.Now()
	for idx := range 2 {

		for i := range rings {
			be.ModUpQtoP(2, 6, ctIn.Value[idx], rings[i])
		}
		starttime_ := time.Now()
		matmult.PPMM_Blas_CRT(rings, u, params, size, size, params.N(), 7, ringP, rings)
		elapse_ := time.Since(starttime_)
		fmt.Println("PPMM Time : ", elapse_)

		be.ModUpPtoQ(6, 2, rings[0], ctIn.Value[idx])
	}
	elapse = time.Since(starttime)
	fmt.Println(elapse)
	util.Mul_(evaluator, ctIn, 1/scale, ctIn)
	util.Rescale_NonNTT(evaluator, ctIn, ctIn)

	ringQ.AtLevel(ctIn.Level()).NTT(ctIn.Value[0], ctIn.Value[0])
	ringQ.AtLevel(ctIn.Level()).NTT(ctIn.Value[1], ctIn.Value[1])

	values := make([]float64, n)

	dept := decryptor.DecryptNew(ctIn)
	encoder.Decode(dept, values)

	fmt.Println(values)
}

func Test_ModSwitchTime2(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            16,
		LogQ:            []int{48, 48, 48},
		LogP:            []int{50},
		Xs:              ring.Ternary{H: 256},
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

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	n := params.N()
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
		i := i
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

	fmt.Println("N is this  : ", n)
	value := make([]float64, n)
	for i, _ := range value {
		value[i] = 0.1
	}

	size := 1 << 5
	scale := float64(1 << 5)
	u := make([]float64, 7*size*size)
	for i := range 7 {
		for j := range size {
			for k := range size {
				u[i*size*size+j*size+k] = float64(uint64(0.2 * scale))
			}
		}
	}

	pt := hefloat.NewPlaintext(params, 2)
	pt.IsBatched = false
	P := []uint64{12451841, 14155777, 13631489, 16384001, 13238273, 8650753, 16121857}
	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	ringQ := params.RingQ().AtLevel(ct.Level())
	ringP, _ := ring.NewRing(params.N(), P)

	be := matmult.NewBasisExtender(ringQ, ringP, []matmult.Key{{From: 2, To: 6}}, []matmult.Key{{From: 6, To: 2}})

	rings := make([][]ring.Poly, 2)
	for i := range rings {
		rings[i] = make([]ring.Poly, size)
		for j := range rings[i] {
			rings[i][j] = ringP.NewPoly()
		}
	}

	ctIn := ct.CopyNew()
	ringQ.INTT(ctIn.Value[0], ctIn.Value[0])
	ringQ.INTT(ctIn.Value[1], ctIn.Value[1])

	for idx := range 2 {
		for i := range rings[idx] {
			be.ModSwitchQtoP_Old(2, 6, ctIn.Value[idx], rings[idx][i])
		}
	}
	buffer1 := make([]float64, 7*params.N()*size)
	buffer2 := make([]float64, 7*params.N()*size)

	starttime = time.Now()
	matmult.PPMM_Blas_CRT_Inplace(rings, u, size, size, params.N(), 7, 2, ringP, buffer1, buffer2)
	elapse = time.Since(starttime)
	fmt.Println("PPMM Time : ", elapse)
	fmt.Println(ringP.ModuliChain()[0])

	for idx := range 2 {
		be.ModSwitchPtoQ_Old(6, 2, rings[idx][0], ctIn.Value[idx])
	}
	sc := rlwe.NewScale(ringQ.ModuliChain()[2])

	util.Mul_ScaleExact(evaluator, ctIn, 1/scale, ctIn, sc)
	util.Rescale_NonNTT(evaluator, ctIn, ctIn)

	ringQ.AtLevel(ctIn.Level()).NTT(ctIn.Value[0], ctIn.Value[0])
	ringQ.AtLevel(ctIn.Level()).NTT(ctIn.Value[1], ctIn.Value[1])

	values := make([]float64, n)

	dept := decryptor.DecryptNew(ctIn)
	encoder.Decode(dept, values)

	fmt.Println(values)
}
func Test_ModSwitch2(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(1) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            16,
		LogQ:            []int{50, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48},
		LogP:            []int{50},
		Xs:              ring.Ternary{H: 256},
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

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	n := params.N()
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
		i := i
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

	fmt.Println("N is this  : ", n)
	value := make([]float64, n)
	for i, _ := range value {
		value[i] = 0.001
	}

	P := []uint64{14624959, 15092711, 16654291, 16108999, 13227197, 16099607, 14232433, 16704799, 15543343, 12965263, 13134193, 15297563, 13536821, 13918787, 12676193, 15142703, 14437559, 12631777, 13704083, 15632377, 14880601, 15491477, 16625353, 16231427, 13351381, 15831551, 15576611, 15039943, 16321373, 16651757, 16722103, 12801337, 13858841, 13097699, 13844113, 14952997, 14788847, 15081413, 15146069, 16552919, 15883789, 13399327, 13466251, 16003619, 14104499, 15536119, 16667741, 12891587, 13922939, 13375783, 14587351, 15993793, 13077359, 13881059, 14541113, 16076227, 15457859, 13534247, 16327063, 16321843, 15841601, 12901619, 12990127, 14647547, 13025123, 13413511, 15537433, 13879141, 15439847, 13149043}
	P = P[0:40]

	startLevels := []int{14, 13, 2, 1}
	scale := float64(1 << 10)
	for l := range startLevels {
		fmt.Println("//////////////////////////////////////////////////////////////")
		startLevel := startLevels[l]
		fmt.Println("level : ", startLevel)
		sc := rlwe.NewScale(params.Q()[startLevel])
		sc = sc.Mul(params.DefaultScale())
		sc = sc.Div(rlwe.NewScale(scale))

		pt := hefloat.NewPlaintext(params, startLevel)
		pt.IsBatched = false
		encoder.Encode(value, pt)
		ct, _ := encryptor.EncryptNew(pt)
		ringQ := params.RingQ().AtLevel(startLevel)
		ringP, _ := ring.NewRing(params.N(), P)
		be := ring.NewBasisExtender(ringQ, ringP)

		PLevel := 0
		for range P {
			if ringQ.ModulusAtLevel[startLevel].Cmp(ringP.ModulusAtLevel[PLevel]) < 0 {
				break
			}
			PLevel++
		}
		fmt.Println(PLevel)

		size := 1 << 8
		if startLevel%2 == 1 {
			size = 1 << 7
		}

		u := make([][][]uint64, PLevel+1)
		for i := range u {
			u[i] = make([][]uint64, size)
			for j := range u[i] {
				u[i][j] = make([]uint64, size)
				for k := range u[i][j] {
					u[i][j][k] = uint64(0.125*scale) % P[i]
				}
			}
		}
		cts := make([]*rlwe.Ciphertext, size)
		for i := range size {
			cts[i] = ct.CopyNew()
			ringQ.INTT(cts[i].Value[0], cts[i].Value[0])
			ringQ.INTT(cts[i].Value[1], cts[i].Value[1])
		}
		rings := make([]ring.Poly, size)
		for i := range size {
			rings[i] = ringP.NewPoly()
		}

		starttime = time.Now()
		for idx := range 2 {
			time_ := time.Now()
			for i := range size {
				be.ModUpQtoP(startLevel, PLevel, cts[i].Value[idx], rings[i])
			}
			elapse_ := time.Since(time_)
			fmt.Println("modswitch", elapse_)
			time_ = time.Now()
			matmult.PPMM_Blas_CRT(rings, u, params, size, size, params.N(), PLevel+1, ringP, rings)
			elapse_ = time.Since(time_)
			fmt.Println("ppmm", elapse_)

			time_ = time.Now()
			for i := range size {
				be.ModUpPtoQ(PLevel, startLevel, rings[i], cts[i].Value[idx])
			}
			elapse_ = time.Since(time_)
			fmt.Println("modswitch2", elapse_)
		}
		for i := range size {
			util.Mul_(evaluator, cts[i], 1/scale, cts[i])
			util.Rescale_NonNTT(evaluator, cts[i], cts[i])
		}
		elapse = time.Since(starttime)
		fmt.Println(elapse)

		for i := range size {
			ringQ.AtLevel(startLevel-1).NTT(cts[i].Value[0], cts[i].Value[0])
			ringQ.AtLevel(startLevel-1).NTT(cts[i].Value[1], cts[i].Value[1])
		}

		values := make([]float64, n)

		dept := decryptor.DecryptNew(cts[2])
		encoder.Decode(dept, values)
		fmt.Println(cts[2].Level())
		fmt.Println(cts[2].LogScale())
		fmt.Println(values)
		fmt.Println("//////////////////////////////////////////////////////////////")
	}
}

func Test_ModSwitch_Opt(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            16,
		LogQ:            []int{50, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48},
		LogP:            []int{50},
		Xs:              ring.Ternary{H: 256},
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

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	n := params.N()
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
		i := i
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

	fmt.Println("N is this  : ", n)
	value := make([]float64, n)
	for i, _ := range value {
		value[i] = 0.001
	}

	P := []uint64{14624959, 15092711, 16654291, 16108999, 13227197, 16099607, 14232433, 16704799, 15543343, 12965263, 13134193, 15297563, 13536821, 13918787, 12676193, 15142703, 14437559, 12631777, 13704083, 15632377, 14880601, 15491477, 16625353, 16231427, 13351381, 15831551, 15576611, 15039943, 16321373, 16651757, 16722103, 12801337, 13858841, 13097699, 13844113, 14952997, 14788847, 15081413, 15146069, 16552919, 15883789, 13399327, 13466251, 16003619, 14104499, 15536119, 16667741, 12891587, 13922939, 13375783, 14587351, 15993793, 13077359, 13881059, 14541113, 16076227, 15457859, 13534247, 16327063, 16321843, 15841601, 12901619, 12990127, 14647547, 13025123, 13413511, 15537433, 13879141, 15439847, 13149043}
	fmt.Println("len P", len(P))
	startLevels := []int{2}
	levelstep := 2
	sizes := []int{1 << 7, 1 << 8}
	scale := float64(1 << 30)

	for l := range startLevels {
		fmt.Println("//////////////////////////////////////////////////////////////")
		pt := hefloat.NewPlaintext(params, startLevels[l])
		pt.IsBatched = false
		encoder.Encode(value, pt)
		ct, _ := encryptor.EncryptNew(pt)
		ringQ := params.RingQ().AtLevel(startLevels[l])
		ringP, _ := ring.NewRing(params.N(), P)
		be := ring.NewBasisExtender(ringQ, ringP)
		ringQ.INTT(ct.Value[0], ct.Value[0])
		ringQ.INTT(ct.Value[1], ct.Value[1])
		PLevel := 0
		for range P {
			if ringQ.ModulusAtLevel[startLevels[l]].Cmp(ringP.ModulusAtLevel[PLevel]) < 0 {
				break
			}
			PLevel++
		}
		fmt.Println(PLevel)
		PLevel += 3

		rings := make([][]ring.Poly, 2)
		for idx := range 2 {
			rings[idx] = make([]ring.Poly, 1<<10)
			for i := range 1 << 10 {
				rings[idx][i] = ringP.NewPoly()
			}
		}
		for step := range levelstep {
			startLevel := startLevels[l]
			fmt.Println("level : ", startLevel+step)

			size := sizes[step]
			u := make([][][]uint64, PLevel+1)
			for i := range u {
				u[i] = make([][]uint64, size)
				for j := range u[i] {
					u[i][j] = make([]uint64, size)
					for k := range u[i][j] {
						u[i][j][k] = uint64(0.5 * scale)
					}
				}
			}
			cts := make([]*rlwe.Ciphertext, size)
			for i := range size {
				cts[i] = ct.CopyNew()
			}

			for idx := range 2 {
				if step == 0 {
					time_ := time.Now()
					for i := range size {
						be.ModUpQtoP(startLevel, PLevel, cts[i].Value[idx], rings[idx][i])
					}
					elapse_ := time.Since(time_)
					fmt.Println("modswitch", elapse_)
				}

				time_ := time.Now()
				matmult.PPMM_Blas_CRT(rings[idx], u, params, size, size, params.N(), PLevel+1, ringP, rings[idx])
				elapse_ := time.Since(time_)
				fmt.Println("ppmm", elapse_)
				if step == levelstep-1 {
					time_ = time.Now()
					for i := range size {
						be.ModUpPtoQ(PLevel, startLevel, rings[idx][i], cts[i].Value[idx])
					}
					elapse_ = time.Since(time_)
					fmt.Println("modswitch2", elapse_)

				}
			}
			ct = cts[0].CopyNew()
		}
		ringQ.NTT(ct.Value[0], ct.Value[0])
		ringQ.NTT(ct.Value[1], ct.Value[1])
		evaluator.Mul(ct, 1/scale, ct)
		evaluator.Rescale(ct, ct)
		evaluator.Mul(ct, 1/scale, ct)
		evaluator.Rescale(ct, ct)
		elapse = time.Since(starttime)
		fmt.Println(elapse)

		values := make([]float64, n)

		dept := decryptor.DecryptNew(ct)
		encoder.Decode(dept, values)
		fmt.Println(ct.Level())
		fmt.Println(ct.LogScale())
		fmt.Println(values)
		fmt.Println("//////////////////////////////////////////////////////////////")
	}
}

func Test_ModSwitch_Opt2(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(1) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            16,
		LogQ:            []int{50, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48},
		LogP:            []int{50},
		Xs:              ring.Ternary{H: 256},
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

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	n := params.N()
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
		i := i
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

	fmt.Println("N is this  : ", n)
	value := make([]float64, n)
	for i, _ := range value {
		value[i] = 0.001
	}

	//24bit
	P := []uint64{14624959, 15092711, 16654291, 16108999, 13227197, 16099607, 14232433, 16704799, 15543343, 12965263, 13134193, 15297563, 13536821, 13918787, 12676193, 15142703, 14437559, 12631777, 13704083, 15632377, 14880601, 15491477, 16625353, 16231427, 13351381, 15831551, 15576611, 15039943, 16321373, 16651757, 16722103, 12801337, 13858841, 13097699, 13844113, 14952997, 14788847, 15081413, 15146069, 16552919, 15883789, 13399327, 13466251, 16003619, 14104499, 15536119, 16667741, 12891587, 13922939, 13375783, 14587351, 15993793, 13077359, 13881059, 14541113, 16076227, 15457859, 13534247, 16327063, 16321843, 15841601, 12901619, 12990127, 14647547, 13025123, 13413511, 15537433, 13879141, 15439847, 13149043}
	P = P[:40]
	startLevels := []int{14}
	levelstep := 2
	sizes := []int{1 << 8, 1 << 7}
	scale := float64(1 << 20)
	for l := range startLevels {
		fmt.Println("//////////////////////////////////////////////////////////////")
		pt := hefloat.NewPlaintext(params, startLevels[l])
		pt.IsBatched = false
		encoder.Encode(value, pt)
		ct, _ := encryptor.EncryptNew(pt)
		ringQ := params.RingQ().AtLevel(startLevels[l])
		ringP, _ := ring.NewRing(params.N(), P)
		be := ring.NewBasisExtender(ringQ, ringP)
		ringQ.INTT(ct.Value[0], ct.Value[0])
		ringQ.INTT(ct.Value[1], ct.Value[1])
		PLevel := 0
		for range P {
			if ringQ.ModulusAtLevel[startLevels[l]].Cmp(ringP.ModulusAtLevel[PLevel]) < 0 {
				break
			}
			PLevel++
		}
		fmt.Println(PLevel)
		PLevel += 2

		sc := rlwe.NewScale(1)
		for i := range levelstep {
			q := rlwe.NewScale(params.Q()[startLevels[l]-i])
			sc = sc.Mul(q)
		}

		ringiters := make([]ring.Poly, 2)
		for idx := range ringiters {
			ringiters[idx] = ringP.NewPoly()
		}
		var totaltime time.Duration
		for step := range levelstep {
			var steptime time.Duration
			startLevel := startLevels[l]
			fmt.Println("level : ", startLevel-step)

			size := sizes[step]
			u := make([][][]uint64, PLevel+1)
			for i := range u {
				u[i] = make([][]uint64, size)
				for j := range u[i] {
					u[i][j] = make([]uint64, size)
					for k := range u[i][j] {
						u[i][j][k] = uint64(0.5*scale) % ringP.ModuliChain()[i]
					}
				}
			}

			rings := make([][]ring.Poly, 2)
			for idx := range 2 {
				rings[idx] = make([]ring.Poly, sizes[step])
				for i := range sizes[step] {
					rings[idx][i] = ringP.NewPoly()
				}
			}

			fmt.Println("/////////////////////////////////")
			fmt.Println("step : ", step)
			if step == 0 {
				time_ := time.Now()
				for idx := range 2 {
					for i := range size {
						be.ModUpQtoP(startLevel, PLevel, ct.Value[idx], rings[idx][i])
					}
				}
				elapse_ := time.Since(time_)
				fmt.Println("modswitch", elapse_)
				steptime += elapse_
			} else {
				for idx := range 2 {
					for i := range size {
						rings[idx][i] = ringiters[idx]
					}
				}
			}

			time_ := time.Now()
			for idx := range 2 {
				matmult.PPMM_Blas_CRT(rings[idx], u, params, size, size, params.N(), PLevel+1, ringP, rings[idx])
			}
			elapse_ := time.Since(time_)
			fmt.Println("ppmm", elapse_)
			steptime += elapse_

			if step == levelstep-1 {
				time_ = time.Now()
				for idx := range 2 {
					for i := range size {
						_ = i
						be.ModUpPtoQ(PLevel, startLevel, rings[idx][i], ct.Value[idx])
					}
				}
				elapse_ = time.Since(time_)
				fmt.Println("modswitch2", elapse_)
				steptime += elapse_
			}
			for idx := range 2 {
				ringiters[idx] = rings[idx][0]
			}
			if step != 0 && step != levelstep-1 {
				fmt.Println("steptime : ", steptime)
				steptime = steptime * 12 * (1 << 10)
				fmt.Println("steptotaltime : ", steptime)
			} else {
				if step == 0 {

					fmt.Println("steptime : ", steptime)
					steptime = steptime * 8 * (1 << 7)
					fmt.Println("steptotaltime : ", steptime)
				} else {
					fmt.Println("steptime : ", steptime)
					steptime = steptime * 8 * (1 << 8)
					fmt.Println("steptotaltime : ", steptime)
				}
			}
			totaltime += steptime
		}

		sscale := 1.0
		for range levelstep {
			sscale *= scale
		}
		time_ := time.Now()
		//fmt.Println(ct.Level())
		// ringQ.NTT(ct.Value[0], ct.Value[0])
		// ringQ.NTT(ct.Value[1], ct.Value[1])

		util.Mul_ScaleExact(evaluator, ct, 1/(sscale), ct, sc)
		//evaluator.Mul2(ct, 1/(scale*scale*scale), ct, sc)
		//ct.Scale = sc
		//fmt.Println(ct.LogScale())
		util.Rescale_NonNTT(evaluator, ct, ct)
		elapse_ := time.Since(time_)
		fmt.Println("rescale (have to mult degree times)", elapse_)
		fmt.Println("rescale", elapse_*(1<<16))
		totaltime += elapse_ * (1 << 16)
		fmt.Println("totaltime (cal.) : ", totaltime.Seconds())
		ringQ.AtLevel(ct.Level()).NTT(ct.Value[0], ct.Value[0])
		ringQ.AtLevel(ct.Level()).NTT(ct.Value[1], ct.Value[1])
		values := make([]float64, n)

		dept := decryptor.DecryptNew(ct)
		encoder.Decode(dept, values)
		fmt.Println(ct.Level())
		fmt.Println(ct.LogScale())
		fmt.Println(values)
		fmt.Println("//////////////////////////////////////////////////////////////")
	}
}

func Test_ModSwitch_Opt3(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(1) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            16,
		LogQ:            []int{50, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48},
		LogP:            []int{50},
		Xs:              ring.Ternary{H: 256},
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

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	n := params.N()
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
		i := i
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

	fmt.Println("N is this  : ", n)
	value := make([]float64, n)
	for i, _ := range value {
		value[i] = 0.001
	}

	//24bit
	P := []uint64{14624959, 15092711, 16654291, 16108999, 13227197, 16099607, 14232433, 16704799, 15543343, 12965263, 13134193, 15297563, 13536821, 13918787, 12676193, 15142703, 14437559, 12631777, 13704083, 15632377, 14880601, 15491477, 16625353, 16231427, 13351381, 15831551, 15576611, 15039943, 16321373, 16651757, 16722103, 12801337, 13858841, 13097699, 13844113, 14952997, 14788847, 15081413, 15146069, 16552919, 15883789, 13399327, 13466251, 16003619, 14104499, 15536119, 16667741, 12891587, 13922939, 13375783, 14587351, 15993793, 13077359, 13881059, 14541113, 16076227, 15457859, 13534247, 16327063, 16321843, 15841601, 12901619, 12990127, 14647547, 13025123, 13413511, 15537433, 13879141, 15439847, 13149043}
	//23bit
	P = []uint64{4194319, 4194329, 4194353, 4194371, 4194389, 4194397, 4194403, 4194409, 4194419, 4194433, 4194439, 4194451, 4194493, 4194503, 4194511, 4194523, 4194527, 4194559, 4194581, 4194583, 4194599, 4194601, 4194637, 4194643, 4194661, 4194677, 4194679, 4194713, 4194719, 4194739, 4194761, 4194769, 4194781}

	// P = P[:40]
	startLevels := []int{14}
	levelstep := 2
	sizes := []int{1 << 8, 1 << 7}
	scale := float64(1 << 40)
	for l := range startLevels {
		fmt.Println("//////////////////////////////////////////////////////////////")
		pt := hefloat.NewPlaintext(params, startLevels[l])
		pt.IsBatched = false
		encoder.Encode(value, pt)
		ct, _ := encryptor.EncryptNew(pt)
		ringQ := params.RingQ().AtLevel(startLevels[l])
		ringP, _ := ring.NewRing(params.N(), P)
		// PLevel := 2*startLevels[l] - 3
		PLevel := 32
		fmt.Println("ring Q : ", ringQ.ModulusAtLevel[startLevels[l]])
		fmt.Println("ring P : ", ringP.ModulusAtLevel[PLevel])

		be := matmult.NewBasisExtender(ringQ, ringP, []matmult.Key{{From: startLevels[l], To: PLevel}}, []matmult.Key{{From: PLevel, To: startLevels[l]}})
		ringQ.INTT(ct.Value[0], ct.Value[0])
		ringQ.INTT(ct.Value[1], ct.Value[1])

		sc := rlwe.NewScale(1)
		for i := range levelstep {
			q := rlwe.NewScale(params.Q()[startLevels[l]-i])
			sc = sc.Mul(q)
		}

		ringiters := make([]ring.Poly, 2)
		for idx := range ringiters {
			ringiters[idx] = ringP.NewPoly()
		}
		var totaltime time.Duration
		for step := range levelstep {
			var steptime time.Duration
			startLevel := startLevels[l]
			fmt.Println("level : ", startLevel-step)

			size := sizes[step]
			u := make([][][]uint64, PLevel+1)
			for i := range u {
				u[i] = make([][]uint64, size)
				for j := range u[i] {
					u[i][j] = make([]uint64, size)
					for k := range u[i][j] {
						u[i][j][k] = uint64(0.5*scale) % ringP.ModuliChain()[i]
					}
				}
			}

			rings := make([][]ring.Poly, 2)
			for idx := range 2 {
				rings[idx] = make([]ring.Poly, sizes[step])
				for i := range sizes[step] {
					rings[idx][i] = ringP.NewPoly()
				}
			}

			fmt.Println("/////////////////////////////////")
			fmt.Println("step : ", step)
			if step == 0 {
				time_ := time.Now()
				for idx := range 2 {
					for i := range size {
						be.ModSwitchQtoP_Old(startLevel, PLevel, ct.Value[idx], rings[idx][i])
					}
				}
				elapse_ := time.Since(time_)
				fmt.Println("modswitch", elapse_)
				steptime += elapse_
			} else {
				for idx := range 2 {
					for i := range size {
						rings[idx][i] = ringiters[idx]
					}
				}
			}

			time_ := time.Now()
			for idx := range 2 {
				matmult.PPMM_Blas_CRT(rings[idx], u, params, size, size, params.N(), PLevel+1, ringP, rings[idx])
			}
			elapse_ := time.Since(time_)
			fmt.Println("ppmm", elapse_)
			steptime += elapse_

			if step == levelstep-1 {
				time_ = time.Now()
				for idx := range 2 {
					for i := range size {
						_ = i
						be.ModSwitchPtoQ_Old(PLevel, startLevel, rings[idx][i], ct.Value[idx])
					}
				}
				elapse_ = time.Since(time_)
				fmt.Println("modswitch2", elapse_)
				steptime += elapse_
			}
			for idx := range 2 {
				ringiters[idx] = rings[idx][0]
			}
			if step != 0 && step != levelstep-1 {
				fmt.Println("steptime : ", steptime)
				steptime = steptime * 12 * (1 << 10)
				fmt.Println("steptotaltime : ", steptime)
			} else {
				if step == 0 {

					fmt.Println("steptime : ", steptime)
					steptime = steptime * 8 * (1 << 7)
					fmt.Println("steptotaltime : ", steptime)
				} else {
					fmt.Println("steptime : ", steptime)
					steptime = steptime * 8 * (1 << 8)
					fmt.Println("steptotaltime : ", steptime)
				}
			}
			totaltime += steptime
		}

		sscale := 1.0
		for range levelstep {
			sscale *= scale
		}
		time_ := time.Now()
		//fmt.Println(ct.Level())
		// ringQ.NTT(ct.Value[0], ct.Value[0])
		// ringQ.NTT(ct.Value[1], ct.Value[1])

		util.Mul_ScaleExact(evaluator, ct, 1/(sscale), ct, sc)
		//evaluator.Mul2(ct, 1/(scale*scale*scale), ct, sc)
		//ct.Scale = sc
		//fmt.Println(ct.LogScale())
		util.Rescale_NonNTT(evaluator, ct, ct)
		elapse_ := time.Since(time_)
		fmt.Println("rescale (have to mult degree times)", elapse_)
		fmt.Println("rescale", elapse_*(1<<16))
		totaltime += elapse_ * (1 << 16)
		fmt.Println("totaltime (cal.) : ", totaltime.Seconds())
		ringQ.AtLevel(ct.Level()).NTT(ct.Value[0], ct.Value[0])
		ringQ.AtLevel(ct.Level()).NTT(ct.Value[1], ct.Value[1])
		values := make([]float64, n)

		dept := decryptor.DecryptNew(ct)
		encoder.Decode(dept, values)
		fmt.Println(ct.Level())
		fmt.Println(ct.LogScale())
		fmt.Println(values)
		fmt.Println("//////////////////////////////////////////////////////////////")
	}
}

func Test_ModSwitch_Opt4(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(1) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            16,
		LogQ:            []int{50, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48},
		LogP:            []int{50},
		Xs:              ring.Ternary{H: 256},
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

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	n := params.N()
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
		i := i
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

	fmt.Println("N is this  : ", n)
	value := make([]float64, n)
	for i, _ := range value {
		value[i] = 0.001
	}

	//24bit
	P := []uint64{14624959, 15092711, 16654291, 16108999, 13227197, 16099607, 14232433, 16704799, 15543343, 12965263, 13134193, 15297563, 13536821, 13918787, 12676193, 15142703, 14437559, 12631777, 13704083, 15632377, 14880601, 15491477, 16625353, 16231427, 13351381, 15831551, 15576611, 15039943, 16321373, 16651757, 16722103, 12801337, 13858841, 13097699, 13844113, 14952997, 14788847, 15081413, 15146069, 16552919, 15883789, 13399327, 13466251, 16003619, 14104499, 15536119, 16667741, 12891587, 13922939, 13375783, 14587351, 15993793, 13077359, 13881059, 14541113, 16076227, 15457859, 13534247, 16327063, 16321843, 15841601, 12901619, 12990127, 14647547, 13025123, 13413511, 15537433, 13879141, 15439847, 13149043}
	//23bit
	P = []uint64{4194319, 4194329, 4194353, 4194371, 4194389, 4194397, 4194403, 4194409, 4194419, 4194433, 4194439, 4194451, 4194493, 4194503, 4194511, 4194523, 4194527, 4194559, 4194581, 4194583, 4194599, 4194601, 4194637, 4194643, 4194661, 4194677, 4194679, 4194713, 4194719, 4194739, 4194761, 4194769, 4194781}

	// P = P[:40]
	startLevels := []int{2}
	levelstep := 2
	sizes := []int{1 << 8, 1 << 7}
	scale := float64(1 << 30)
	for l := range startLevels {
		fmt.Println("//////////////////////////////////////////////////////////////")
		pt := hefloat.NewPlaintext(params, startLevels[l])
		pt.IsBatched = false
		encoder.Encode(value, pt)
		ct, _ := encryptor.EncryptNew(pt)
		ringQ := params.RingQ().AtLevel(startLevels[l])
		ringP, _ := ring.NewRing(params.N(), P)
		// PLevel := 2*startLevels[l] - 3
		PLevel := 6
		fmt.Println("ring Q : ", ringQ.ModulusAtLevel[startLevels[l]])
		fmt.Println("ring P : ", ringP.ModulusAtLevel[PLevel])

		be := matmult.NewBasisExtender(ringQ, ringP, []matmult.Key{{From: startLevels[l], To: PLevel}}, []matmult.Key{{From: PLevel, To: startLevels[l]}})
		bredP := make([]uint64, len(P))
		for i := range bredP {
			bredP[i] = ring.BRedConstant(P[i])[0]
		}

		ringQ.INTT(ct.Value[0], ct.Value[0])
		ringQ.INTT(ct.Value[1], ct.Value[1])

		sc := rlwe.NewScale(1)
		for i := range levelstep {
			q := rlwe.NewScale(params.Q()[startLevels[l]-i])
			sc = sc.Mul(q)
		}

		ringiters := make([]ring.Poly, 2)
		for idx := range ringiters {
			ringiters[idx] = ringP.NewPoly()
		}
		var totaltime time.Duration
		for step := range levelstep {
			var steptime time.Duration
			startLevel := startLevels[l]
			fmt.Println("level : ", startLevel-step)

			size := sizes[step]
			u := make([][][]uint64, PLevel+1)
			for i := range u {
				u[i] = make([][]uint64, size)
				for j := range u[i] {
					u[i][j] = make([]uint64, size)
					for k := range u[i][j] {
						u[i][j][k] = uint64(0.5*scale) % ringP.ModuliChain()[i]
					}
				}
			}

			rings := make([][]ring.Poly, 2)
			for idx := range 2 {
				rings[idx] = make([]ring.Poly, sizes[step])
				for i := range sizes[step] {
					rings[idx][i] = ringP.NewPoly()
				}
			}

			fmt.Println("/////////////////////////////////")
			fmt.Println("step : ", step)
			if step == 0 {
				time_ := time.Now()
				for idx := range 2 {
					for i := range size {
						be.ModSwitchQtoP_Old(startLevel, PLevel, ct.Value[idx], rings[idx][i])
					}
				}
				elapse_ := time.Since(time_)
				fmt.Println("modswitch", elapse_)
				steptime += elapse_
			} else {
				for idx := range 2 {
					for i := range size {
						rings[idx][i] = ringiters[idx]
					}
				}
			}

			time_ := time.Now()
			for idx := range 2 {
				// fmt.Println(rings[idx][0])
				matmult.PPMM_Blas_CRTBarret(rings[idx], u, params, size, size, params.N(), PLevel+1, bredP, ringP, rings[idx])
				// matmult.PPMM_Blas_CRT(rings[idx], u, params, size, size, params.N(), PLevel+1, ringP, rings[idx])
				// fmt.Println(rings[idx][0])
			}
			elapse_ := time.Since(time_)
			fmt.Println("ppmm", elapse_)
			steptime += elapse_

			if step == levelstep-1 {
				time_ = time.Now()
				for idx := range 2 {
					for i := range size {
						_ = i
						be.ModSwitchPtoQ_Old(PLevel, startLevel, rings[idx][i], ct.Value[idx])
					}
				}
				elapse_ = time.Since(time_)
				fmt.Println("modswitch2", elapse_)
				steptime += elapse_
			}
			for idx := range 2 {
				ringiters[idx] = rings[idx][0]
			}
			if step != 0 && step != levelstep-1 {
				fmt.Println("steptime : ", steptime)
				steptime = steptime * 12 * (1 << 10)
				fmt.Println("steptotaltime : ", steptime)
			} else {
				if step == 0 {

					fmt.Println("steptime : ", steptime)
					steptime = steptime * 8 * (1 << 7)
					fmt.Println("steptotaltime : ", steptime)
				} else {
					fmt.Println("steptime : ", steptime)
					steptime = steptime * 8 * (1 << 8)
					fmt.Println("steptotaltime : ", steptime)
				}
			}
			totaltime += steptime
		}

		sscale := 1.0
		for range levelstep {
			sscale *= scale
		}
		time_ := time.Now()
		//fmt.Println(ct.Level())
		// ringQ.NTT(ct.Value[0], ct.Value[0])
		// ringQ.NTT(ct.Value[1], ct.Value[1])

		util.Mul_ScaleExact(evaluator, ct, 1/(sscale), ct, sc)
		//evaluator.Mul2(ct, 1/(scale*scale*scale), ct, sc)
		//ct.Scale = sc
		//fmt.Println(ct.LogScale())
		util.Rescale_NonNTT(evaluator, ct, ct)
		elapse_ := time.Since(time_)
		fmt.Println("rescale (have to mult degree times)", elapse_)
		fmt.Println("rescale", elapse_*(1<<16))
		totaltime += elapse_ * (1 << 16)
		fmt.Println("totaltime (cal.) : ", totaltime.Seconds())
		ringQ.AtLevel(ct.Level()).NTT(ct.Value[0], ct.Value[0])
		ringQ.AtLevel(ct.Level()).NTT(ct.Value[1], ct.Value[1])
		values := make([]float64, n)

		dept := decryptor.DecryptNew(ct)
		encoder.Decode(dept, values)
		fmt.Println(ct.Level())
		fmt.Println(ct.LogScale())
		fmt.Println(values)
		fmt.Println("//////////////////////////////////////////////////////////////")
	}
}
