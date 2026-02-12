package transpose

import (
	"fmt"
	"math"
	"math/bits"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/lifejade/mm/src/util"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

func Test_Transpose(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// CIFAR-10
		// index [0]
		// logN = 16, full slots
		// logq = 51, logp = 46
		// scale = 1<<46
		// # special modulus = 3
		// # available levels = 16
		LogN:            10,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51},
		LogDefaultScale: 46,
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

	fmt.Println("key list:")
	fmt.Println(evaluator.EvaluationKeySet.GetGaloisKeysList())
	fmt.Println(len(evaluator.EvaluationKeySet.GetGaloisKeysList()))

	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	ringQ := params.RingQ().AtLevel(ct.Level())

	ringQ.INTT(ct.Value[0], ct.Value[0])
	ringQ.INTT(ct.Value[1], ct.Value[1])

	ninv := ringQ.NewRNSScalarFromUInt64(uint64(n * 2))
	ringQ.MFormRNSScalar(ninv, ninv)
	ringQ.Inverse(ninv)

	cts := make([]*rlwe.Ciphertext, 2*n)
	for i := range cts {
		cts[i] = ct.CopyNew()
		ringQ.MultByMonomial(cts[i].Value[0], i, cts[i].Value[0])
		ringQ.MultByMonomial(cts[i].Value[1], i, cts[i].Value[1])

		ringQ.NTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.NTT(cts[i].Value[1], cts[i].Value[1])

	}
	aux := Tweak(cts, params, evaluator, encoder, 2*n)
	res := make([]*rlwe.Ciphertext, 2*n)
	for i := range res {
		idx, ch := ModInv(uint64(2*i+1), uint64(4*n))
		if !ch {
			fmt.Println("err ", i, " ", 4*n-1, " ", idx)
		}
		res[i] = aux[(idx-1)/2].CopyNew()

		ringQ.INTT(res[i].Value[0], res[i].Value[0])
		ringQ.INTT(res[i].Value[1], res[i].Value[1])
		ringQ.MForm(res[i].Value[0], res[i].Value[0])
		ringQ.MForm(res[i].Value[1], res[i].Value[1])

		ringQ.MulRNSScalarMontgomery(res[i].Value[0], ninv, res[i].Value[0])
		ringQ.MulRNSScalarMontgomery(res[i].Value[1], ninv, res[i].Value[1])

		ringQ.IMForm(res[i].Value[0], res[i].Value[0])
		ringQ.IMForm(res[i].Value[1], res[i].Value[1])
		ringQ.NTT(res[i].Value[0], res[i].Value[0])
		ringQ.NTT(res[i].Value[1], res[i].Value[1])

		if err := evaluator.Automorphism(res[i], uint64(2*i+1), res[i]); err != nil {
			fmt.Println(err)
		}

	}

	res2 := Tweak(res, params, evaluator, encoder, 2*n)
	result := make([]*rlwe.Ciphertext, 2*n)
	for i := range 2 * n {
		ringQ.INTT(res2[i].Value[0], res2[i].Value[0])
		ringQ.INTT(res2[i].Value[1], res2[i].Value[1])

		ringQ.MultByMonomial(res2[i].Value[0], i, res2[i].Value[0])
		ringQ.MultByMonomial(res2[i].Value[1], i, res2[i].Value[1])
		if i != 0 {
			ringQ.Neg(res2[i].Value[0], res2[i].Value[0])
			ringQ.Neg(res2[i].Value[1], res2[i].Value[1])
		}
		ringQ.NTT(res2[i].Value[0], res2[i].Value[0])
		ringQ.NTT(res2[i].Value[1], res2[i].Value[1])

		result[(2*n-i)%(2*n)] = res2[i]
	}

	reval := make([][]float64, 2*n)
	for i := range reval {
		reval[i] = make([]float64, 2*n)
		dept := decryptor.DecryptNew(result[i])
		err := encoder.Decode(dept, reval[i])
		if err != nil {
			fmt.Println(err)
		}
	}

	for i := range 2 * n {
		if i > 3 {
			break
		}
		fmt.Println(reval[i][0:10])
	}

	fmt.Println("check all")
	for i := range 2 * n {
		for j := range 2 * n {
			if int(math.Round(reval[i][j]*1000)) != i {
				fmt.Println("err : ", i, j, reval[i][j])

			}
		}
	}
}

func Test_Transpose2(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(1) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	test := func(level int) {
		logq := []int{51}
		for range level {
			logq = append(logq, 51)
		}
		//ckks parameter init
		SchemeParams := hefloat.ParametersLiteral{
			// CIFAR-10
			// index [0]
			// logN = 16, full slots
			// logq = 51, logp = 46
			// scale = 1<<46
			// # special modulus = 3
			// # available levels = 16
			LogN:            16,
			LogQ:            logq,
			LogP:            []int{51},
			LogDefaultScale: 46,
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

		n := 1 << params.LogMaxSlots()

		var pk *rlwe.PublicKey
		var rlk *rlwe.RelinearizationKey
		var rtk []*rlwe.GaloisKey

		fmt.Println("generated bootstrapper end")
		pk = kgen.GenPublicKeyNew(sk)
		rlk = kgen.GenRelinearizationKeyNew(sk)

		// generate keys - Rotating key
		galEls := make([]uint64, 20)
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

		fmt.Println("key list:")
		fmt.Println(evaluator.EvaluationKeySet.GetGaloisKeysList())
		fmt.Println(len(evaluator.EvaluationKeySet.GetGaloisKeysList()))

		value := make([]float64, 2*n)
		for i := range value {
			value[i] = 0.001 * float64(i)
		}

		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		pt.IsBatched = false

		encoder.Encode(value, pt)
		cts := make([]*rlwe.Ciphertext, 2*n)
		ct, _ := encryptor.EncryptNew(pt)
		params.RingQ().AtLevel(ct.Level()).INTT(ct.Value[0], ct.Value[0])
		params.RingQ().AtLevel(ct.Level()).INTT(ct.Value[1], ct.Value[1])
		for i := range cts {
			cts[i] = ct
		}
		fmt.Println("ctgen end")

		result := Transpose2(cts, params, evaluator, 2*n)

		reval := make([][]float64, 2*n)
		for i := range reval {
			reval[i] = make([]float64, 2*n)
			params.RingQ().AtLevel(result[i].Level()).NTT(result[i].Value[0], result[i].Value[0])
			params.RingQ().AtLevel(result[i].Level()).NTT(result[i].Value[1], result[i].Value[1])
			dept := decryptor.DecryptNew(result[i])
			err := encoder.Decode(dept, reval[i])
			if err != nil {
				fmt.Println(err)
			}
		}

		for i := range 2 * n {
			if i > 3 {
				break
			}
			fmt.Println(reval[i][0:10])
		}
	}

	levels := []int{12}
	for _, v := range levels {
		fmt.Println("//////////////////////////////////////////////////////////////////////")
		fmt.Println("level :", v)
		test(v)
		runtime.GC()
		fmt.Println("//////////////////////////////////////////////////////////////////////")
	}

	// fmt.Println("check all")
	// for i := range 2 * n {
	// 	for j := range 2 * n {
	// 		if int(math.Round(reval[i][j]*1000)) != i {
	// 			fmt.Println("err : ", i, j, reval[i][j])

	// 		}
	// 	}
	// }
}

func Test_ScalarCoeffMul(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            10,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51},
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

	n := 1 << params.LogMaxSlots()
	fmt.Println("degree : ", n*2)

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
	fmt.Println(params.GaloisElementForComplexConjugation())
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

	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.01 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	fmt.Println(ct.IsMontgomery)
	fmt.Println(ct.IsNTT)

	ringQ := params.RingQ().AtLevel(ct.Level())

	ninv := ringQ.NewRNSScalarFromUInt64(uint64(n * 2))
	ringQ.MFormRNSScalar(ninv, ninv)
	ringQ.Inverse(ninv)

	temp := ct.CopyNew()

	ringQ.INTT(temp.Value[0], temp.Value[0])
	ringQ.INTT(temp.Value[1], temp.Value[1])
	fmt.Println(ninv[0])
	fmt.Println(temp.Value[0].Coeffs[0][0:10])

	ringQ.MForm(temp.Value[0], temp.Value[0])
	ringQ.MForm(temp.Value[1], temp.Value[1])

	ringQ.MulRNSScalarMontgomery(temp.Value[0], ninv, temp.Value[0])
	ringQ.MulRNSScalarMontgomery(temp.Value[1], ninv, temp.Value[1])

	ringQ.IMForm(temp.Value[0], temp.Value[0])
	ringQ.IMForm(temp.Value[1], temp.Value[1])

	fmt.Println(temp.Value[0].Coeffs[0][0:10])
	fmt.Println(params.Q()[0])
	ringQ.NTT(temp.Value[0], temp.Value[0])
	ringQ.NTT(temp.Value[1], temp.Value[1])

	res, _ := evaluator.MulNew(temp, 2*n)

	reval := make([]float64, 2*n)
	dept := decryptor.DecryptNew(res)
	err = encoder.Decode(dept, reval)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(reval[0:20])
}

func Test_Auto(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            10,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51},
		LogDefaultScale: 46,
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

	n := 1 << params.LogMaxSlots()
	fmt.Println("degree : ", n*2)

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
	fmt.Println(params.GaloisElementForComplexConjugation())
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

	fmt.Println("key list:")
	fmt.Println(evaluator.EvaluationKeySet.GetGaloisKeysList())
	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	ringQ := params.RingQ().AtLevel(ct.Level())
	_ = ringQ
	cts := make([]*rlwe.Ciphertext, 2*n)
	for i := range cts {
		cts[i] = ct.CopyNew()
	}

	res := make([]*rlwe.Ciphertext, 2*n)
	for i := range res {
		res[i] = cts[i].CopyNew()
		if _, err := evaluator.EvaluationKeySet.GetGaloisKey(uint64(2*i + 1)); err != nil {
			evaluator.Automorphism(res[i], uint64(2*i+1), res[i])
		} else {
			evaluator.Automorphism(res[i], uint64(2*i+1), res[i])
		}
	}

	reval := make([][]float64, 2*n)
	for i := range reval {
		reval[i] = make([]float64, 2*n)
		dept := decryptor.DecryptNew(res[i])
		err := encoder.Decode(dept, reval[i])
		if err != nil {
			fmt.Println(err)
		}
	}
	fmt.Println(reval[1][0:10])

	fmt.Println(reval[10][20:30])

	fmt.Println(reval[200][70:80])
}

func Test_Tweak(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// CIFAR-10
		// index [0]
		// logN = 16, full slots
		// logq = 51, logp = 46
		// scale = 1<<46
		// # special modulus = 3
		// # available levels = 16
		LogN:            16,
		LogQ:            []int{51},
		LogP:            []int{51},
		LogDefaultScale: 46,
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
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	convRot := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	galEls := make([]uint64, len(convRot))
	for i, x := range convRot {
		galEls[i] = params.GaloisElement(x)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	for i := range 16 {
		galEls = append(galEls, params.GaloisElement((1<<i)/2))
	}

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

	n := 1 << params.LogMaxSlots()
	tweakN := 2 * n
	value := make([]float64, 2*n)
	// value2 := make([]float64, 2*n)
	// value3 := make([]float64, 2*n)
	// value4 := make([]float64, 2*n)
	// for i := range value {
	// 	switch i % 4 {
	// 	case 0:
	// 		value[i] = 0.1
	// 		value2[i] = 0.5
	// 		value3[i] = 0.01
	// 		value4[i] = 0.6
	// 	case 1:
	// 		value[i] = 0.7
	// 		value2[i] = 0.1
	// 		value3[i] = 0.07
	// 		value4[i] = 0.3
	// 	case 2:
	// 		value[i] = 0.11
	// 		value2[i] = 0.1
	// 		value3[i] = 0.09
	// 		value4[i] = 0.3
	// 	case 3:
	// 		value[i] = 0.4
	// 		value2[i] = 0.9
	// 		value3[i] = 0.1
	// 		value4[i] = 0.9
	// 	}
	// }

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	ringQ := params.RingQ().AtLevel(pt.Level())

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	// encoder.Encode(value2, pt)
	// ct2, _ := encryptor.EncryptNew(pt)
	// encoder.Encode(value3, pt)
	// ct3, _ := encryptor.EncryptNew(pt)
	// encoder.Encode(value4, pt)
	// ct4, _ := encryptor.EncryptNew(pt)

	ringQ.INTT(ct.Value[0], ct.Value[0])
	ringQ.INTT(ct.Value[1], ct.Value[1])
	// ringQ.INTT(ct2.Value[0], ct2.Value[0])
	// ringQ.INTT(ct2.Value[1], ct2.Value[1])
	// ringQ.INTT(ct3.Value[0], ct3.Value[0])
	// ringQ.INTT(ct3.Value[1], ct3.Value[1])
	// ringQ.INTT(ct4.Value[0], ct4.Value[0])
	// ringQ.INTT(ct4.Value[1], ct4.Value[1])

	cts := make([]*rlwe.Ciphertext, tweakN)
	for i := range cts {
		// switch i % 4 {
		// case 0:
		// 	cts[i] = ct.CopyNew()
		// case 1:
		// 	cts[i] = ct2.CopyNew()
		// case 2:
		// 	cts[i] = ct3.CopyNew()
		// case 3:
		// 	cts[i] = ct4.CopyNew()
		// }
		cts[i] = ct.CopyNew()
	}
	printMemUsage()
	fmt.Println("Tweak start")
	starttime = time.Now()
	// results := Tweak3(cts, params, evaluator, encoder, tweakN)
	TweakInplace(cts, params, evaluator, tweakN)
	results := cts
	elapse = time.Since(starttime)
	fmt.Println("Tweak End", len(results))
	fmt.Println("elapse ", elapse)

	for i := range results {
		ringQ.NTT(results[i].Value[0], results[i].Value[0])
		ringQ.NTT(results[i].Value[1], results[i].Value[1])
	}

	reval := make([][]float64, tweakN)
	for i := range reval {
		reval[i] = make([]float64, 2*n)
		dept := decryptor.DecryptNew(results[i])
		err := encoder.Decode(dept, reval[i])
		if err != nil {
			fmt.Println(err)
		}
	}

	fmt.Println(reval[0][0:10])

	// for i := range reval {
	// 	for j := range reval {
	// 		if math.Round(reval[i][j]) != 0 {
	// 			fmt.Println(i, ", ", j, " : ", reval[i][j])
	// 		}
	// 	}
	// }
}

func Test_INTTAdd(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            10,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51},
		LogDefaultScale: 46,
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

	n := 1 << params.LogMaxSlots()
	fmt.Println("degree : ", n*2)

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, 1)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())
	fmt.Println(params.GaloisElementForComplexConjugation())
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

	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	ct2, _ := encryptor.EncryptNew(pt)
	ringQ := params.RingQ().AtLevel(ct.Level())
	ringQ.INTT(ct.Value[0], ct.Value[0])
	ringQ.INTT(ct.Value[1], ct.Value[1])
	ringQ.INTT(ct2.Value[0], ct2.Value[0])
	ringQ.INTT(ct2.Value[1], ct2.Value[1])

	res, _ := evaluator.SubNew(ct, ct2)
	ringQ.NTT(res.Value[0], res.Value[0])
	ringQ.NTT(res.Value[1], res.Value[1])

	dept := decryptor.DecryptNew(res)
	encoder.Decode(dept, value)

	fmt.Println(value[:10])

}

func Test_Shift(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// CIFAR-10
		// index [0]
		// logN = 16, full slots
		// logq = 51, logp = 46
		// scale = 1<<46
		// # special modulus = 3
		// # available levels = 16
		LogN:            10,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51},
		LogDefaultScale: 46,
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
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	convRot := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	galEls := make([]uint64, len(convRot))
	for i, x := range convRot {
		galEls[i] = params.GaloisElement(x)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	for i := range 16 {
		galEls = append(galEls, params.GaloisElement((1<<i)/2))
	}

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

	n := 1 << params.LogMaxSlots()
	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)

	fmt.Println(len(ct.Value))
	fmt.Println(len(ct.Value[0].Coeffs))
	fmt.Println(len(ct.Value[0].Coeffs[0]))

	ringQ := params.RingQ().AtLevel(ct.Level())
	// p := ringQ.NewPoly()
	// for i := range p.Coeffs{
	// 	p.Coeffs[i][1] = 1
	// }
	// ringQ.NTT(p,p)

	// tarr := make([]uint64, 2*n)
	// tarr[1] = 1

	ringQ.INTT(ct.Value[0], ct.Value[0])
	ringQ.INTT(ct.Value[1], ct.Value[1])

	fmt.Println(ct.Value[0].Coeffs[0][0:10])
	fmt.Println(ct.Value[1].Coeffs[0][0:10])

	fmt.Println(ct.Value[0].Coeffs[0][1023])
	fmt.Println(ct.Value[1].Coeffs[0][1023])

	// var c0, c1 ring.Poly
	// c0 = ringQ.NewPoly()
	// c1 = ringQ.NewPoly()
	// ringQ.MForm(ct.Value[0], c0)
	// ringQ.MForm(ct.Value[1], c1)

	ringQ.MultByMonomial(ct.Value[0], 512, ct.Value[0])
	ringQ.MultByMonomial(ct.Value[1], 512, ct.Value[1])

	// ringQ.IMForm(ct.Value[0], ct.Value[0])
	// ringQ.IMForm(ct.Value[1], ct.Value[1])

	fmt.Println(ct.Value[0].Coeffs[0][0:10])
	fmt.Println(ct.Value[1].Coeffs[0][0:10])

	ringQ.NTT(ct.Value[0], ct.Value[0])
	ringQ.NTT(ct.Value[1], ct.Value[1])

	reval := make([]float64, 2*n)
	dept := decryptor.DecryptNew(ct)
	err = encoder.Decode(dept, reval)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(reval[0:10])

	fmt.Println(reval[20:30])

	fmt.Println(reval[510:520])

}

func Test_TweakSparse(t *testing.T) {
	runtime.GOMAXPROCS(1)

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            10,
		LogQ:            []int{51, 51, 51, 51, 51},
		LogP:            []int{51},
		LogDefaultScale: 46,
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
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	convRot := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	galEls := make([]uint64, len(convRot))
	for i, x := range convRot {
		galEls[i] = params.GaloisElement(x)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	for i := range 16 {
		galEls = append(galEls, params.GaloisElement((1<<i)/2))
	}

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

	n := 1 << params.LogN()
	tweakN := n / 4

	ringQ := params.RingQ().AtLevel(params.MaxLevel())

	cts := make([]*rlwe.Ciphertext, tweakN)
	for i := range cts {
		value := make([]float64, n)
		for j := range value {
			if j%(n/tweakN) != 0 {
				continue
			}
			value[j] = 0.001 * float64(i*n+j)
		}
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		pt.IsBatched = false
		encoder.Encode(value, pt)
		cts[i], _ = encryptor.EncryptNew(pt)
		ringQ.INTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.INTT(cts[i].Value[1], cts[i].Value[1])
	}
	fmt.Println("Tweak start")
	starttime = time.Now()
	results := Tweak(cts, params, evaluator, encoder, tweakN)
	elapse = time.Since(starttime)
	fmt.Println("Tweak End", len(results))
	fmt.Println("elapse ", elapse)

	for i := range results {
		ringQ.NTT(results[i].Value[0], results[i].Value[0])
		ringQ.NTT(results[i].Value[1], results[i].Value[1])
	}

	reval := make([][]float64, tweakN)
	for i := range reval {
		reval[i] = make([]float64, 2*n)
		dept := decryptor.DecryptNew(results[i])
		err := encoder.Decode(dept, reval[i])
		if err != nil {
			fmt.Println(err)
		}
	}

	fmt.Println(reval[0][0:30])
	fmt.Println(reval[1][0:30])
	fmt.Println(reval[2][0:30])
	fmt.Println(reval[3][0:30])
}

func Test_TransSparse(t *testing.T) {
	runtime.GOMAXPROCS(1)

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            5,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51},
		LogDefaultScale: 46,
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

	n := 1 << params.LogN()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, n)
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

	value := make([]float64, n)
	ratio := 4
	sparseN := n / ratio
	for i := range value {
		if i%ratio != 0 {
			value[i] = 0
			continue
		}
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	ringQ := params.RingQ().AtLevel(ct.Level())

	ringQ.INTT(ct.Value[0], ct.Value[0])
	ringQ.INTT(ct.Value[1], ct.Value[1])

	cts := make([]*rlwe.Ciphertext, sparseN)
	for i := range cts {
		cts[i] = ct.CopyNew()
	}
	elapse = time.Since(starttime)
	fmt.Println("mult mono : ", elapse)
	total := elapse

	starttime = time.Now()
	work := make([]*rlwe.Ciphertext, sparseN)
	aux := make([]*rlwe.Ciphertext, sparseN)
	ctzero := util.CtZero(params, encoder, encryptor)
	for i := range work {
		work[i] = ctzero.CopyNew()
		aux[i] = ctzero.CopyNew()
	}
	starttime = time.Now()
	// cts = Transpose2(cts, params, evaluator, n)
	Transpose3(cts, params, evaluator, ringQ, n, sparseN, work, aux, cts)
	// for i := range work {
	// 	work[i] = ctzero.CopyNew()
	// 	aux[i] = ctzero.CopyNew()
	// }
	// Transpose_Sparse(cts, params, evaluator, ringQ, n, sparseN, work, aux, cts)
	total += elapse
	fmt.Println("total time : ", total)

	reval := make([][]float64, sparseN)
	for i := range reval {
		reval[i] = make([]float64, n)
		ringQ.NTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.NTT(cts[i].Value[1], cts[i].Value[1])
		// evaluator.Mul(cts[i], 1.0/float64(sparseN), cts[i])
		// evaluator.Mul(cts[i], 1.0/float64(sparseN), cts[i])
		// evaluator.Rescale(cts[i], cts[i])

		dept := decryptor.DecryptNew(cts[i])
		err := encoder.Decode(dept, reval[i])
		if err != nil {
			fmt.Println(err)
		}
		fmt.Println(reval[i])
	}

}

func Test_TransSparse2(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU())

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51},
		LogDefaultScale: 46,
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

	n := 1 << params.LogN()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	galLen := 256
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

	fmt.Println("generate Evaluator end")
	runtime.GOMAXPROCS(1)

	_, _, _, _ = encoder, encryptor, decryptor, evaluator

	value := make([]float64, n)
	testing := func(ratio int) {

		sparseN := n / ratio
		fmt.Println(sparseN)
		for i := range value {
			if i%ratio == 0 {
				value[i] = 0.001 * float64(i)
			} else {
				value[i] = 0
			}
		}

		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		pt.IsBatched = false

		encoder.Encode(value, pt)
		ct, _ := encryptor.EncryptNew(pt)
		ringQ := params.RingQ().AtLevel(ct.Level())

		ringQ.INTT(ct.Value[0], ct.Value[0])
		ringQ.INTT(ct.Value[1], ct.Value[1])

		ninv := ringQ.NewRNSScalarFromUInt64(uint64(sparseN))
		ringQ.MFormRNSScalar(ninv, ninv)
		ringQ.Inverse(ninv)

		idxarr := make([]uint64, sparseN)

		for i := range idxarr {
			idx, ch := ModInv(uint64(2*i+1), uint64(2*sparseN))
			_ = ch
			idxarr[i] = idx
		}

		for i := range value {
			value[i] = 0
		}
		encoder.Encode(value, pt)
		// ctzero, _ := encryptor.EncryptNew(pt)

		cts := make([]*rlwe.Ciphertext, sparseN)
		for i := range cts {
			if ratio < 64 {
				cts[i] = ct
			} else {
				cts[i] = ct.CopyNew()
			}

		}

		starttime := time.Now()
		for i := range cts {
			ringQ.MultByMonomial(cts[i].Value[0], i*ratio, cts[i].Value[0])
			ringQ.MultByMonomial(cts[i].Value[1], i*ratio, cts[i].Value[1])
		}
		elapse := time.Since(starttime)
		fmt.Println("mult mono : ", elapse)
		total := elapse

		starttime = time.Now()
		aux := Tweak3(cts, params, evaluator, ringQ, sparseN)
		elapse = time.Since(starttime)
		fmt.Println("tweak1 : ", elapse)
		total += elapse

		// var except time.Duration
		starttime = time.Now()
		res := make([]*rlwe.Ciphertext, sparseN)
		for i := range res {
			st := time.Now()
			res[i] = aux[(idxarr[i]-1)/2].CopyNew()

			ringQ.MForm(res[i].Value[0], res[i].Value[0])
			ringQ.MForm(res[i].Value[1], res[i].Value[1])

			ringQ.MulRNSScalarMontgomery(res[i].Value[0], ninv, res[i].Value[0])
			ringQ.MulRNSScalarMontgomery(res[i].Value[1], ninv, res[i].Value[1])

			ringQ.IMForm(res[i].Value[0], res[i].Value[0])
			ringQ.IMForm(res[i].Value[1], res[i].Value[1])

			res[i].IsNTT = false

			// exctime := time.Now()
			// galEl := uint64((2*i + 1))
			// kgen_ := rlwe.NewKeyGenerator(params)
			// gk := kgen_.GenGaloisKeyNew(galEl, sk)
			// _ = gk
			// except += time.Since(exctime)
			// Automorphism(evaluator, ringQ, res[i], galEl, gk, res[i])

			if i == 255 || i == 3 {
				el := time.Since(st)
				fmt.Println(el)
				st = time.Now()
			}
			galEl := uint64((2*i + 1))
			var gk *rlwe.GaloisKey
			if gk, err = evaluator.CheckAndGetGaloisKey(galEl); err != nil {
				fmt.Println("cannot apply Automorphism:", err)
			}
			Automorphism(evaluator, ringQ, res[i], galEl, gk, res[i])

			res[i].IsNTT = true
			if i == 255 || i == 3 {
				el := time.Since(st)
				fmt.Println(el)
			}

		}
		elapse = time.Since(starttime)
		// elapse -= except
		fmt.Println("auto : ", elapse)
		// fmt.Println("except : ", except)
		total += elapse

		starttime = time.Now()
		res2 := Tweak3(res, params, evaluator, ringQ, sparseN)
		elapse = time.Since(starttime)
		fmt.Println("tweak2 : ", elapse)
		total += elapse

		result := make([]*rlwe.Ciphertext, sparseN)
		starttime = time.Now()
		for idx := range sparseN {
			// idx := i / ratio
			i := idx * ratio

			ringQ.MultByMonomial(res2[idx].Value[0], i, res2[idx].Value[0])
			ringQ.MultByMonomial(res2[idx].Value[1], i, res2[idx].Value[1])
			if i != 0 {
				ringQ.Neg(res2[idx].Value[0], res2[idx].Value[0])
				ringQ.Neg(res2[idx].Value[1], res2[idx].Value[1])
			}
			result[(sparseN-idx)%(sparseN)] = res2[idx]
		}
		elapse = time.Since(starttime)
		fmt.Println("mult mono : ", elapse)
		total += elapse
		fmt.Println("total time : ", total)

		reval := make([][]float64, sparseN)
		for i := range reval {
			reval[i] = make([]float64, n)
			ringQ.NTT(result[i].Value[0], result[i].Value[0])
			ringQ.NTT(result[i].Value[1], result[i].Value[1])

			dept := decryptor.DecryptNew(result[i])
			err := encoder.Decode(dept, reval[i])
			if err != nil {
				fmt.Println(err)
			}
		}

		// for i := range n {
		// 	fmt.Println(reval[i])
		// }

		// fmt.Println("check all")
		// for i := range sparseN {
		// 	for j := range n {
		// 		if (j % ratio) != 0 {
		// 			if int(math.Round(reval[i][j]*1000)) != 0 {
		// 				fmt.Println("err! : ", i, j, reval[i][j])
		// 			}
		// 		} else if int(math.Round(reval[i][j]*1000)) != i*ratio {
		// 			fmt.Println("err : ", i, j, reval[i][j])
		// 		}
		// 	}
		// }
	}
	testing(256)

	// ratio := 1024
	// for range 4 {
	// 	ratio >>= 2
	// 	testing(ratio)
	// 	runtime.GC()
	// 	fmt.Println("///////////////////////////////////////////////////////////////////////")
	// }

}

func Test_Transpose3(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU())

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51},
		LogDefaultScale: 46,
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

	n := 1 << params.LogN()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	ratio := 1 << 0
	sparseN := n / ratio
	galLen := 1
	fmt.Println("galLen : ", galLen)

	// generate keys - Rotating key
	galEls := make([]uint64, galLen)
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
	runtime.GOMAXPROCS(1)

	_, _, _, _ = encoder, encryptor, decryptor, evaluator

	value := make([]float64, n)

	for i := range value {
		if i%ratio == 0 {
			value[i] = 0.001 * float64(i)
		} else {
			value[i] = 0
		}
	}

	pt := hefloat.NewPlaintext(params, 1)
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	ringQ := params.RingQ().AtLevel(ct.Level())

	ringQ.INTT(ct.Value[0], ct.Value[0])
	ringQ.INTT(ct.Value[1], ct.Value[1])

	ninv := ringQ.NewRNSScalarFromUInt64(uint64(sparseN))
	ringQ.MFormRNSScalar(ninv, ninv)
	ringQ.Inverse(ninv)

	for i := range value {
		value[i] = 0
	}
	encoder.Encode(value, pt)
	// ctzero, _ := encryptor.EncryptNew(pt)

	cts := make([]*rlwe.Ciphertext, sparseN)
	for i := range cts {
		cts[i] = ct
	}

	starttime = time.Now()
	starttime_ := time.Now()
	for i := range cts {
		ringQ.MultByMonomial(cts[i].Value[0], i*ratio, cts[i].Value[0])
		ringQ.MultByMonomial(cts[i].Value[1], i*ratio, cts[i].Value[1])
	}
	elapse_ := time.Since(starttime_)
	fmt.Println("Mono mult ", elapse_)
	fmt.Println("tweak1 start")

	starttime_ = time.Now()
	aux := Tweak3(cts, params, evaluator, ringQ, sparseN)
	elapse_ = time.Since(starttime_)
	fmt.Println("Tweak1 ", elapse_)

	fmt.Println("auto start")
	starttime_ = time.Now()
	res := make([]*rlwe.Ciphertext, sparseN)
	for i := range res {
		idx, ch := ModInv(uint64(2*i+1), uint64(2*sparseN))
		if !ch {
			fmt.Println("err ", i, " ", 2*n-1, " ", idx)
		}
		res[i] = aux[(idx-1)/2].CopyNew()

		ringQ.MForm(res[i].Value[0], res[i].Value[0])
		ringQ.MForm(res[i].Value[1], res[i].Value[1])

		ringQ.MulRNSScalarMontgomery(res[i].Value[0], ninv, res[i].Value[0])
		ringQ.MulRNSScalarMontgomery(res[i].Value[1], ninv, res[i].Value[1])

		ringQ.IMForm(res[i].Value[0], res[i].Value[0])
		ringQ.IMForm(res[i].Value[1], res[i].Value[1])

		res[i].IsNTT = false

		if err := evaluator.Automorphism(res[i], uint64((1)), res[i]); err != nil {
			fmt.Println(err)
		}

		res[i].IsNTT = true
	}
	elapse_ = time.Since(starttime_)
	fmt.Println("Auto", elapse_)

	// reval = make([][]float64, len(res))
	// for i := range reval {
	// 	reval[i] = make([]float64, n)
	// 	ringQ.NTT(res[i].Value[0], res[i].Value[0])
	// 	ringQ.NTT(res[i].Value[1], res[i].Value[1])
	// 	dept := decryptor.DecryptNew(res[i])
	// 	err := encoder.Decode(dept, reval[i])
	// 	if err != nil {
	// 		fmt.Println(err)
	// 	}
	// 	ringQ.INTT(res[i].Value[0], res[i].Value[0])
	// 	ringQ.INTT(res[i].Value[1], res[i].Value[1])
	// }

	// for i := range len(res) {
	// 	fmt.Println(reval[i])
	// }
	// fmt.Println()
	fmt.Println("tweak2 start")
	starttime_ = time.Now()
	res2 := Tweak3(res, params, evaluator, ringQ, sparseN)
	elapse_ = time.Since(starttime_)
	fmt.Println("tweak2", elapse_)

	starttime_ = time.Now()
	result := make([]*rlwe.Ciphertext, sparseN)
	for idx := range sparseN {
		// idx := i / ratio
		i := idx * ratio

		ringQ.MultByMonomial(res2[idx].Value[0], i, res2[idx].Value[0])
		ringQ.MultByMonomial(res2[idx].Value[1], i, res2[idx].Value[1])
		if i != 0 {
			ringQ.Neg(res2[idx].Value[0], res2[idx].Value[0])
			ringQ.Neg(res2[idx].Value[1], res2[idx].Value[1])
		}
		ringQ.NTT(res2[idx].Value[0], res2[idx].Value[0])
		ringQ.NTT(res2[idx].Value[1], res2[idx].Value[1])
		result[(sparseN-idx)%(sparseN)] = res2[idx]
	}
	elapse_ = time.Since(starttime_)
	fmt.Println("Mono Mul", elapse_)
	elapse = time.Since(starttime)
	fmt.Println("total time: ", elapse)

	reval := make([][]float64, sparseN)
	for i := range reval {
		reval[i] = make([]float64, n)
		// evaluator.Mul(result[i], 1.0/float64(sparseN), result[i])
		// evaluator.Rescale(result[i], result[i])

		dept := decryptor.DecryptNew(result[i])
		err := encoder.Decode(dept, reval[i])
		if err != nil {
			fmt.Println(err)
		}
	}

	// for i := range n {
	// 	fmt.Println(reval[i])
	// }

	fmt.Println("check all")
	for i := range sparseN {
		for j := range n {
			if (j % ratio) != 0 {
				if int(math.Round(reval[i][j]*1000)) != 0 {
					fmt.Println("err! : ", i, j, reval[i][j])
				}
			} else if int(math.Round(reval[i][j]*1000)) != i*ratio {
				fmt.Println("err : ", i, j, reval[i][j])
			}
		}
	}

}

func Test_AutoTime(t *testing.T) {
	test := func(level int) {
		logq := []int{51}
		for range level {
			logq = append(logq, 51)
		}
		//ckks parameter init
		SchemeParams := hefloat.ParametersLiteral{
			// CIFAR-10
			// index [0]
			// logN = 16, full slots
			// logq = 51, logp = 46
			// scale = 1<<46
			// # special modulus = 3
			// # available levels = 16
			LogN:            16,
			LogQ:            logq,
			LogP:            []int{51},
			LogDefaultScale: 46,
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

		n := 1 << params.LogMaxSlots()

		var pk *rlwe.PublicKey
		var rlk *rlwe.RelinearizationKey
		var rtk []*rlwe.GaloisKey

		fmt.Println("generated bootstrapper end")
		pk = kgen.GenPublicKeyNew(sk)
		rlk = kgen.GenRelinearizationKeyNew(sk)

		// generate keys - Rotating key
		galEls := make([]uint64, 20)
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

		fmt.Println("key list:")
		fmt.Println(evaluator.EvaluationKeySet.GetGaloisKeysList())
		fmt.Println(len(evaluator.EvaluationKeySet.GetGaloisKeysList()))

		value := make([]float64, 2*n)
		for i := range value {
			value[i] = 0.001 * float64(i)
		}

		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		pt.IsBatched = false

		encoder.Encode(value, pt)
		ct, _ := encryptor.EncryptNew(pt)
		// ct.IsNTT = false

		st := time.Now()
		evaluator.Automorphism(ct, 11, ct)
		el := time.Since(st)
		fmt.Println(level, "of res ", el)

	}

	for _, v := range []int{0, 4, 9, 12} {
		test(v)
		fmt.Println()
	}
}

func Test_TweakTime(t *testing.T) {
	test := func(level int) {
		logq := []int{51}
		for range level {
			logq = append(logq, 51)
		}
		//ckks parameter init
		SchemeParams := hefloat.ParametersLiteral{
			// CIFAR-10
			// index [0]
			// logN = 16, full slots
			// logq = 51, logp = 46
			// scale = 1<<46
			// # special modulus = 3
			// # available levels = 16
			LogN:            14,
			LogQ:            logq,
			LogP:            []int{51},
			LogDefaultScale: 46,
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

		n := 1 << params.LogMaxSlots()

		var pk *rlwe.PublicKey
		var rlk *rlwe.RelinearizationKey
		var rtk []*rlwe.GaloisKey

		fmt.Println("generated bootstrapper end")
		pk = kgen.GenPublicKeyNew(sk)
		rlk = kgen.GenRelinearizationKeyNew(sk)

		// generate keys - Rotating key
		galEls := make([]uint64, 20)
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

		fmt.Println("key list:")
		fmt.Println(evaluator.EvaluationKeySet.GetGaloisKeysList())
		fmt.Println(len(evaluator.EvaluationKeySet.GetGaloisKeysList()))

		value := make([]float64, 2*n)
		for i := range value {
			value[i] = 0.001 * float64(i)
		}

		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		pt.IsBatched = false

		encoder.Encode(value, pt)
		ct, _ := encryptor.EncryptNew(pt)
		cts := make([]*rlwe.Ciphertext, 2*n)
		for i := range cts {
			cts[i] = ct
		}
		ringQ := params.RingQ()
		// ct.IsNTT = false

		st := time.Now()
		Tweak3(cts, params, evaluator, ringQ, 2*n)
		el := time.Since(st)
		fmt.Println(level, "of res ", el)

	}

	for _, v := range []int{0, 4, 9, 12} {
		test(v)
		runtime.GC()
		fmt.Println()
	}
}

func Test_TweakSoundness(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU())

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            8,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51},
		LogDefaultScale: 46,
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

	n := 1 << params.LogN()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, 1)
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
	runtime.GOMAXPROCS(1)

	_, _, _, _ = encoder, encryptor, decryptor, evaluator

	value := make([]float64, n)
	ratio := 16
	sparseN := n / ratio
	for i := range value {
		if i%ratio == 0 {
			value[i] = 0.001 * float64(i)
		} else {
			value[i] = 0
		}
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	ringQ := params.RingQ().AtLevel(ct.Level())

	ringQ.INTT(ct.Value[0], ct.Value[0])
	ringQ.INTT(ct.Value[1], ct.Value[1])

	cts := make([]*rlwe.Ciphertext, sparseN)
	for i := range cts {
		cts[i] = ct.CopyNew()
	}

	starttime := time.Now()
	twk1 := Tweak3(cts, params, evaluator, ringQ, sparseN)
	elapse := time.Since(starttime)
	fmt.Println("tweak_origin : ", elapse)
	value1 := make([][]float64, sparseN)
	for i := range value1 {
		value1[i] = make([]float64, n)
		pt := decryptor.DecryptNew(twk1[i])
		encoder.Decode(pt, value1[i])
	}

	for i := range cts {
		cts[i] = ct.CopyNew()
	}
	starttime = time.Now()
	twk2 := Tweak3_check1(cts, params, evaluator, ringQ, sparseN)
	elapse = time.Since(starttime)
	fmt.Println("tweak1 : ", elapse)
	value2 := make([][]float64, sparseN)
	for i := range value2 {
		value2[i] = make([]float64, n)
		pt := decryptor.DecryptNew(twk2[i])
		encoder.Decode(pt, value2[i])
	}
	for i := range sparseN {
		for j := range n {
			if value1[i][j] != value2[i][j] {
				fmt.Println("err at ", i, j, value1[i][j], value2[i][j])
			}
		}
	}

	for i := range cts {
		cts[i] = ct.CopyNew()
	}
	work := make([]*rlwe.Ciphertext, sparseN)
	for i := range work {
		work[i] = cts[0].CopyNew()
	}
	starttime = time.Now()
	twk3 := Tweak3_check2(cts, params, evaluator, ringQ, sparseN, work, 0)
	elapse = time.Since(starttime)
	fmt.Println("tweak2 : ", elapse)
	value3 := make([][]float64, sparseN)
	for i := range value3 {
		value3[i] = make([]float64, n)
		pt := decryptor.DecryptNew(twk3[i])
		encoder.Decode(pt, value3[i])
	}
	for i := range sparseN {
		for j := range n {
			if value1[i][j] != value3[i][j] {
				fmt.Println("err at ", i, j, value1[i][j], value3[i][j])
				break
			}
		}
	}

	for i := range cts {
		cts[i] = ct.CopyNew()
	}
	twk4 := make([]*rlwe.Ciphertext, sparseN)
	for i := range twk4 {
		twk4[i] = cts[0].CopyNew()
	}
	starttime = time.Now()
	Tweak3_check3(cts, params, evaluator, ringQ, sparseN, work, 0, twk4, 0)
	elapse = time.Since(starttime)
	fmt.Println("tweak3 : ", elapse)
	value4 := make([][]float64, sparseN)
	for i := range value4 {
		value4[i] = make([]float64, n)
		pt := decryptor.DecryptNew(twk4[i])
		encoder.Decode(pt, value4[i])
	}
	for i := range sparseN {
		for j := range n {
			if value1[i][j] != value4[i][j] {
				fmt.Println("err at ", i, j, value1[i][j], value4[i][j])
				break
			}
		}
	}

	for i := range sparseN {
		fmt.Println(value1[i])
	}
	fmt.Println()

	for i := range sparseN {
		fmt.Println(value4[i])
	}
}

func Test_TweakSoundnessTime(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU())

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51},
		LogDefaultScale: 46,
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

	n := 1 << params.LogN()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, 1)
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

	runtime.GOMAXPROCS(1)
	testing := func(ratio int) {
		sparseN := n / ratio
		fmt.Println(sparseN)
		value := make([]float64, sparseN)
		for i := range value {
			if i%ratio == 0 {
				value[i] = 0.001 * float64(i)
			} else {
				value[i] = 0
			}
		}

		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		pt.IsBatched = false

		encoder.Encode(value, pt)
		ct, _ := encryptor.EncryptNew(pt)
		ringQ := params.RingQ().AtLevel(ct.Level())

		ringQ.INTT(ct.Value[0], ct.Value[0])
		ringQ.INTT(ct.Value[1], ct.Value[1])

		cts := make([]*rlwe.Ciphertext, sparseN)
		for i := range cts {
			cts[i] = ct.CopyNew()
		}

		starttime := time.Now()
		twk1 := Tweak3(cts, params, evaluator, ringQ, sparseN)
		elapse := time.Since(starttime)
		fmt.Println("tweak_origin : ", elapse)
		value1 := make([][]float64, sparseN)
		for i := range value1 {
			value1[i] = make([]float64, n)
			pt := decryptor.DecryptNew(twk1[i])
			encoder.Decode(pt, value1[i])
		}

		for i := range cts {
			cts[i] = ct.CopyNew()
		}
		starttime = time.Now()
		twk2 := Tweak3_check1(cts, params, evaluator, ringQ, sparseN)
		elapse = time.Since(starttime)
		fmt.Println("tweak1 : ", elapse)
		value2 := make([][]float64, sparseN)
		for i := range value2 {
			value2[i] = make([]float64, n)
			pt := decryptor.DecryptNew(twk2[i])
			encoder.Decode(pt, value2[i])
		}
		for i := range sparseN {
			for j := range n {
				if value1[i][j] != value2[i][j] {
					fmt.Println("err at ", i, j, value1[i][j], value2[i][j])
				}
			}
		}

		for i := range cts {
			cts[i] = ct.CopyNew()
		}
		work := make([]*rlwe.Ciphertext, sparseN)
		for i := range work {
			work[i] = cts[0].CopyNew()
		}
		starttime = time.Now()
		twk3 := Tweak3_check2(cts, params, evaluator, ringQ, sparseN, work, 0)
		elapse = time.Since(starttime)
		fmt.Println("tweak2 : ", elapse)
		value3 := make([][]float64, sparseN)
		for i := range value3 {
			value3[i] = make([]float64, n)
			pt := decryptor.DecryptNew(twk3[i])
			encoder.Decode(pt, value3[i])
		}
		for i := range sparseN {
			for j := range n {
				if value1[i][j] != value3[i][j] {
					fmt.Println("err at ", i, j, value1[i][j], value3[i][j])
					break
				}
			}
		}

		for i := range cts {
			cts[i] = ct.CopyNew()
		}
		twk4 := make([]*rlwe.Ciphertext, sparseN)
		for i := range twk4 {
			twk4[i] = cts[0].CopyNew()
		}
		starttime = time.Now()
		Tweak3_check3(cts, params, evaluator, ringQ, sparseN, work, 0, twk4, 0)
		elapse = time.Since(starttime)
		fmt.Println("tweak3 : ", elapse)
		value4 := make([][]float64, sparseN)
		for i := range value4 {
			value4[i] = make([]float64, n)
			pt := decryptor.DecryptNew(twk4[i])
			encoder.Decode(pt, value4[i])
		}
		for i := range sparseN {
			for j := range n {
				if value1[i][j] != value4[i][j] {
					fmt.Println("err at ", i, j, value1[i][j], value4[i][j])
					break
				}
			}
		}

		// for i := range sparseN {
		// 	fmt.Println(value1[i])
		// }
		// fmt.Println()

		// for i := range sparseN {
		// 	fmt.Println(value4[i])
		// }
	}

	ratio := 32
	for range 5 {
		ratio >>= 1
		testing(ratio)
		runtime.GC()
		fmt.Println("///////////////////////////////////////////////////////////////////////")
	}
}

// for test
func Tweak3_check1(cts []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, ringQ *ring.Ring, n int) []*rlwe.Ciphertext {
	if n == 1 {
		return []*rlwe.Ciphertext{cts[0]}
	}

	out := make([]*rlwe.Ciphertext, n)
	out[0] = cts[0]

	logn := bits.Len(uint(n)) - 1

	for i := 1; i < n; i++ {
		out[i] = out[0].CopyNew()
	}
	maxPowl := n / 2
	tempBuf := make([]*rlwe.Ciphertext, maxPowl)

	for l := 0; l < logn; l++ {
		powl := 1 << l
		temp := tempBuf[:powl]
		den := powl * 2
		for j := 0; j < powl; j++ {
			idx := ((2*j + 1) * n) / den
			temp[j] = cts[idx]
		}

		aux := Tweak3_check1(temp, params, eval, ringQ, powl)

		step := (params.MaxSlots() * 2) / powl

		for j := 0; j < powl; j++ {
			tmp := aux[j]

			shift := step * j
			ringQ.MultByMonomial(tmp.Value[0], shift, tmp.Value[0])
			ringQ.MultByMonomial(tmp.Value[1], shift, tmp.Value[1])

			if err := eval.Sub(out[j], tmp, out[j+powl]); err != nil {
				panic(fmt.Errorf("Sub failed: %w", err))
			}

			if err := eval.Add(out[j], tmp, out[j]); err != nil {
				panic(fmt.Errorf("Add failed: %w", err))
			}
		}
	}

	return out
}

func Tweak3_check2(cts []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, ringQ *ring.Ring, n int, work []*rlwe.Ciphertext, workOff int) []*rlwe.Ciphertext {
	if n == 1 {
		return []*rlwe.Ciphertext{cts[0]}
	}
	out := make([]*rlwe.Ciphertext, n)
	out[0] = cts[0]
	for i := 1; i < n; i++ {
		out[i] = out[0].CopyNew()
	}

	logn := bits.Len(uint(n)) - 1
	maxPowl := n / 2
	tempBuf := work[workOff : workOff+maxPowl]

	for l := 0; l < logn; l++ {
		powl := 1 << l
		temp := tempBuf[:powl]
		den := powl * 2
		for j := 0; j < powl; j++ {
			idx := ((2*j + 1) * n) / den
			temp[j] = cts[idx]
		}

		aux := Tweak3_check2(temp, params, eval, ringQ, powl, work, workOff+powl)

		step := (params.MaxSlots() * 2) / powl

		for j := 0; j < powl; j++ {
			tmp := aux[j]

			shift := step * j
			ringQ.MultByMonomial(tmp.Value[0], shift, tmp.Value[0])
			ringQ.MultByMonomial(tmp.Value[1], shift, tmp.Value[1])

			if err := eval.Sub(out[j], tmp, out[j+powl]); err != nil {
				panic(fmt.Errorf("Sub failed: %w", err))
			}

			if err := eval.Add(out[j], tmp, out[j]); err != nil {
				panic(fmt.Errorf("Add failed: %w", err))
			}
		}
	}
	return out
}

func Tweak3_check3(cts []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, ringQ *ring.Ring, n int, work []*rlwe.Ciphertext, workOff int, out []*rlwe.Ciphertext, outOff int) {
	out[outOff] = cts[0]
	if n == 1 {
		return
	}

	logn := bits.Len(uint(n)) - 1
	maxPowl := n / 2
	tempBuf := work[workOff : workOff+maxPowl]

	for l := 0; l < logn; l++ {
		powl := 1 << l
		temp := tempBuf[:powl]
		den := powl * 2
		for j := 0; j < powl; j++ {
			idx := ((2*j + 1) * n) / den
			temp[j] = cts[idx]
		}
		Tweak3_check3(temp, params, eval, ringQ, powl, work, workOff+powl, out, outOff+powl)
		res := out[outOff : outOff+n]

		step := (params.MaxSlots() * 2) / powl

		for j := 0; j < powl; j++ {
			work[len(work)-1].Copy(res[powl+j])
			tmp := work[len(work)-1]

			shift := step * j
			ringQ.MultByMonomial(tmp.Value[0], shift, tmp.Value[0])
			ringQ.MultByMonomial(tmp.Value[1], shift, tmp.Value[1])

			if err := eval.Sub(res[j], tmp, res[j+powl]); err != nil {
				panic(fmt.Errorf("Sub failed: %w", err))
			}

			if err := eval.Add(res[j], tmp, res[j]); err != nil {
				panic(fmt.Errorf("Add failed: %w", err))
			}
		}
	}
}

func Test_TransSparse3(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU())

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51},
		LogDefaultScale: 46,
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

	n := 1 << params.LogN()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	galLen := 256
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

	fmt.Println("generate Evaluator end")
	runtime.GOMAXPROCS(1)

	_, _, _, _ = encoder, encryptor, decryptor, evaluator

	value := make([]float64, n)
	testing := func(ratio int) {

		sparseN := n / ratio
		fmt.Println(sparseN)
		for i := range value {
			if i%ratio == 0 {
				value[i] = 0.001 * float64(i)
			} else {
				value[i] = 0
			}
		}

		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		pt.IsBatched = false

		encoder.Encode(value, pt)
		ct, _ := encryptor.EncryptNew(pt)
		ringQ := params.RingQ().AtLevel(ct.Level())

		ringQ.INTT(ct.Value[0], ct.Value[0])
		ringQ.INTT(ct.Value[1], ct.Value[1])

		ninv := ringQ.NewRNSScalarFromUInt64(uint64(sparseN))
		ringQ.MFormRNSScalar(ninv, ninv)
		ringQ.Inverse(ninv)

		idxarr := make([]uint64, sparseN)

		for i := range idxarr {
			idx, ch := ModInv(uint64(2*i+1), uint64(2*sparseN))
			_ = ch
			idxarr[i] = idx
		}

		for i := range value {
			value[i] = 0
		}
		encoder.Encode(value, pt)
		// ctzero, _ := encryptor.EncryptNew(pt)

		cts := make([]*rlwe.Ciphertext, sparseN)
		for i := range cts {
			if ratio < 64 {
				cts[i] = ct.CopyNew()
			} else {
				cts[i] = ct.CopyNew()
			}

		}

		work := make([]*rlwe.Ciphertext, sparseN)
		for i := range work {
			if ratio < 64 {
				work[i] = ct.CopyNew()
			} else {
				work[i] = ct.CopyNew()
			}
		}
		aux := make([]*rlwe.Ciphertext, sparseN)
		for i := range aux {
			if ratio < 64 {
				aux[i] = ct.CopyNew()
			} else {
				aux[i] = ct.CopyNew()
			}
		}

		starttime := time.Now()
		// for i := range cts {
		// 	ringQ.MultByMonomial(cts[i].Value[0], i*ratio, cts[i].Value[0])
		// 	ringQ.MultByMonomial(cts[i].Value[1], i*ratio, cts[i].Value[1])
		// }
		elapse := time.Since(starttime)
		fmt.Println("mult mono : ", elapse)
		total := elapse

		starttime = time.Now()
		Tweak4(cts, params, evaluator, ringQ, sparseN, work, 0, aux, 0)
		elapse = time.Since(starttime)
		fmt.Println("tweak1 : ", elapse)
		total += elapse

		var except time.Duration
		starttime = time.Now()
		res := make([]*rlwe.Ciphertext, sparseN)
		for i := range res {
			st := time.Now()
			res[i] = aux[(idxarr[i]-1)/2].CopyNew()

			ringQ.MForm(res[i].Value[0], res[i].Value[0])
			ringQ.MForm(res[i].Value[1], res[i].Value[1])

			ringQ.MulRNSScalarMontgomery(res[i].Value[0], ninv, res[i].Value[0])
			ringQ.MulRNSScalarMontgomery(res[i].Value[1], ninv, res[i].Value[1])

			ringQ.IMForm(res[i].Value[0], res[i].Value[0])
			ringQ.IMForm(res[i].Value[1], res[i].Value[1])

			res[i].IsNTT = false

			// exctime := time.Now()
			// galEl := uint64((2*i + 1))
			// kgen_ := rlwe.NewKeyGenerator(params)
			// gk := kgen_.GenGaloisKeyNew(galEl, sk)
			// _ = gk
			// except += time.Since(exctime)
			// Automorphism(evaluator, ringQ, res[i], galEl, gk, res[i])

			// if i == 255 || i == 3 {
			// 	el := time.Since(st)
			// 	fmt.Println(el)
			// 	st = time.Now()
			// }
			// galEl := uint64((2*i + 1))
			// var gk *rlwe.GaloisKey
			// if gk, err = evaluator.CheckAndGetGaloisKey(galEl); err != nil {
			// 	fmt.Println("cannot apply Automorphism:", err)
			// }
			// Automorphism(evaluator, ringQ, res[i], galEl, gk, res[i])

			res[i].IsNTT = true
			if i == 255 || i == 3 {
				el := time.Since(st)
				fmt.Println(el)
			}

		}
		elapse = time.Since(starttime)
		elapse -= except
		fmt.Println("auto : ", elapse)
		// fmt.Println("except : ", except)
		total += elapse

		starttime = time.Now()
		Tweak4(res, params, evaluator, ringQ, sparseN, work, 0, aux, 0)
		res2 := aux
		elapse = time.Since(starttime)
		fmt.Println("tweak2 : ", elapse)
		total += elapse

		result := make([]*rlwe.Ciphertext, sparseN)
		starttime = time.Now()
		for idx := range sparseN {
			// idx := i / ratio
			i := idx * ratio

			ringQ.MultByMonomial(res2[idx].Value[0], i, res2[idx].Value[0])
			ringQ.MultByMonomial(res2[idx].Value[1], i, res2[idx].Value[1])
			if i != 0 {
				ringQ.Neg(res2[idx].Value[0], res2[idx].Value[0])
				ringQ.Neg(res2[idx].Value[1], res2[idx].Value[1])
			}
			result[(sparseN-idx)%(sparseN)] = res2[idx]
		}
		elapse = time.Since(starttime)
		fmt.Println("mult mono : ", elapse)
		total += elapse
		fmt.Println("total time : ", total)

		reval := make([][]float64, sparseN)
		for i := range reval {
			reval[i] = make([]float64, n)
			ringQ.NTT(result[i].Value[0], result[i].Value[0])
			ringQ.NTT(result[i].Value[1], result[i].Value[1])

			dept := decryptor.DecryptNew(result[i])
			err := encoder.Decode(dept, reval[i])
			if err != nil {
				fmt.Println(err)
			}
		}

		// for i := range n {
		// 	fmt.Println(reval[i])
		// }

		// fmt.Println("check all")
		// for i := range sparseN {
		// 	for j := range n {
		// 		if (j % ratio) != 0 {
		// 			if int(math.Round(reval[i][j]*1000)) != 0 {
		// 				fmt.Println("err! : ", i, j, reval[i][j])
		// 			}
		// 		} else if int(math.Round(reval[i][j]*1000)) != i*ratio {
		// 			fmt.Println("err : ", i, j, reval[i][j])
		// 		}
		// 	}
		// }
	}
	// testing(256)

	ratio := 16
	for range 1 {
		ratio >>= 1
		testing(ratio)
		runtime.GC()
		fmt.Println("///////////////////////////////////////////////////////////////////////")
	}

}

func BenchmarkTweak4(t *testing.B) {
	runtime.GOMAXPROCS(runtime.NumCPU())

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51},
		LogDefaultScale: 46,
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

	n := 1 << params.LogN()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

	galLen := 1
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

	fmt.Println("generate Evaluator end")
	runtime.GOMAXPROCS(1)

	_, _, _, _ = encoder, encryptor, decryptor, evaluator

	value := make([]float64, n)
	testing := func(ratio int) {

		sparseN := n / ratio
		fmt.Println(sparseN)
		for i := range value {
			if i%ratio == 0 {
				value[i] = 0.001 * float64(i)
			} else {
				value[i] = 0
			}
		}

		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		pt.IsBatched = false

		encoder.Encode(value, pt)
		ct, _ := encryptor.EncryptNew(pt)
		ringQ := params.RingQ().AtLevel(ct.Level())

		ringQ.INTT(ct.Value[0], ct.Value[0])
		ringQ.INTT(ct.Value[1], ct.Value[1])

		ninv := ringQ.NewRNSScalarFromUInt64(uint64(sparseN))
		ringQ.MFormRNSScalar(ninv, ninv)
		ringQ.Inverse(ninv)

		idxarr := make([]uint64, sparseN)

		for i := range idxarr {
			idx, ch := ModInv(uint64(2*i+1), uint64(2*sparseN))
			_ = ch
			idxarr[i] = idx
		}

		for i := range value {
			value[i] = 0
		}
		encoder.Encode(value, pt)
		// ctzero, _ := encryptor.EncryptNew(pt)

		cts := make([]*rlwe.Ciphertext, sparseN)
		for i := range cts {
			if ratio < 64 {
				cts[i] = ct.CopyNew()
			} else {
				cts[i] = ct.CopyNew()
			}

		}

		work := make([]*rlwe.Ciphertext, sparseN)
		for i := range work {
			if ratio < 64 {
				work[i] = ct.CopyNew()
			} else {
				work[i] = ct.CopyNew()
			}
		}
		aux := make([]*rlwe.Ciphertext, sparseN)
		for i := range aux {
			if ratio < 64 {
				aux[i] = ct.CopyNew()
			} else {
				aux[i] = ct.CopyNew()
			}
		}

		starttime := time.Now()
		// for i := range cts {
		// 	ringQ.MultByMonomial(cts[i].Value[0], i*ratio, cts[i].Value[0])
		// 	ringQ.MultByMonomial(cts[i].Value[1], i*ratio, cts[i].Value[1])
		// }
		elapse := time.Since(starttime)
		fmt.Println("mult mono : ", elapse)
		total := elapse

		starttime = time.Now()
		Tweak4_nonrecur(cts, params, evaluator, ringQ, sparseN, work, 0, aux, 0)
		elapse = time.Since(starttime)
		fmt.Println("tweak1 : ", elapse)
		total += elapse

		// var except time.Duration
		starttime = time.Now()
		res := make([]*rlwe.Ciphertext, sparseN)
		for i := range res {
			st := time.Now()
			res[i] = aux[(idxarr[i]-1)/2].CopyNew()

			ringQ.MForm(res[i].Value[0], res[i].Value[0])
			ringQ.MForm(res[i].Value[1], res[i].Value[1])

			ringQ.MulRNSScalarMontgomery(res[i].Value[0], ninv, res[i].Value[0])
			ringQ.MulRNSScalarMontgomery(res[i].Value[1], ninv, res[i].Value[1])

			ringQ.IMForm(res[i].Value[0], res[i].Value[0])
			ringQ.IMForm(res[i].Value[1], res[i].Value[1])

			res[i].IsNTT = false

			// exctime := time.Now()
			// galEl := uint64((2*i + 1))
			// kgen_ := rlwe.NewKeyGenerator(params)
			// gk := kgen_.GenGaloisKeyNew(galEl, sk)
			// _ = gk
			// except += time.Since(exctime)
			// Automorphism(evaluator, ringQ, res[i], galEl, gk, res[i])

			// if i == 255 || i == 3 {
			// 	el := time.Since(st)
			// 	fmt.Println(el)
			// 	st = time.Now()
			// }
			// galEl := uint64((2*i + 1))
			// var gk *rlwe.GaloisKey
			// if gk, err = evaluator.CheckAndGetGaloisKey(galEl); err != nil {
			// 	fmt.Println("cannot apply Automorphism:", err)
			// }
			// Automorphism(evaluator, ringQ, res[i], galEl, gk, res[i])

			res[i].IsNTT = true
			if i == 255 || i == 3 {
				el := time.Since(st)
				fmt.Println(el)
			}

		}
		// elapse = time.Since(starttime)
		// elapse -= except
		// fmt.Println("auto : ", elapse)
		// // fmt.Println("except : ", except)
		// total += elapse

		// starttime = time.Now()
		// Tweak4(res, params, evaluator, ringQ, sparseN, work, 0, aux, 0)
		// res2 := aux
		// elapse = time.Since(starttime)
		// fmt.Println("tweak2 : ", elapse)
		// total += elapse

		// result := make([]*rlwe.Ciphertext, sparseN)
		// starttime = time.Now()
		// for idx := range sparseN {
		// 	// idx := i / ratio
		// 	i := idx * ratio

		// 	ringQ.MultByMonomial(res2[idx].Value[0], i, res2[idx].Value[0])
		// 	ringQ.MultByMonomial(res2[idx].Value[1], i, res2[idx].Value[1])
		// 	if i != 0 {
		// 		ringQ.Neg(res2[idx].Value[0], res2[idx].Value[0])
		// 		ringQ.Neg(res2[idx].Value[1], res2[idx].Value[1])
		// 	}
		// 	result[(sparseN-idx)%(sparseN)] = res2[idx]
		// }
		// elapse = time.Since(starttime)
		// fmt.Println("mult mono : ", elapse)
		// total += elapse
		// fmt.Println("total time : ", total)

		// reval := make([][]float64, sparseN)
		// for i := range reval {
		// 	reval[i] = make([]float64, n)
		// 	ringQ.NTT(result[i].Value[0], result[i].Value[0])
		// 	ringQ.NTT(result[i].Value[1], result[i].Value[1])

		// 	dept := decryptor.DecryptNew(result[i])
		// 	err := encoder.Decode(dept, reval[i])
		// 	if err != nil {
		// 		fmt.Println(err)
		// 	}
		// }

		// for i := range n {
		// 	fmt.Println(reval[i])
		// }

		// fmt.Println("check all")
		// for i := range sparseN {
		// 	for j := range n {
		// 		if (j % ratio) != 0 {
		// 			if int(math.Round(reval[i][j]*1000)) != 0 {
		// 				fmt.Println("err! : ", i, j, reval[i][j])
		// 			}
		// 		} else if int(math.Round(reval[i][j]*1000)) != i*ratio {
		// 			fmt.Println("err : ", i, j, reval[i][j])
		// 		}
		// 	}
		// }
	}
	// testing(256)

	ratio := 8
	for range 1 {
		ratio >>= 1
		testing(ratio)
		runtime.GC()
		fmt.Println("///////////////////////////////////////////////////////////////////////")
	}

}
