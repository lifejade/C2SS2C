package transpose

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
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
	aux := Tweak2(cts, params, evaluator, encoder, 2*n)
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

	res2 := Tweak2(res, params, evaluator, encoder, 2*n)
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
	cts := make([]*rlwe.Ciphertext, 2*n)
	for i := range cts {
		cts[i] = ct.CopyNew()
	}

	result := Transpose(cts, params, evaluator, encoder, 2*n)

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
	tweakN := 4
	value := make([]float64, 2*n)
	value2 := make([]float64, 2*n)
	value3 := make([]float64, 2*n)
	value4 := make([]float64, 2*n)
	for i := range value {
		switch i % 4 {
		case 0:
			value[i] = 0.1
			value2[i] = 0.5
			value3[i] = 0.01
			value4[i] = 0.6
		case 1:
			value[i] = 0.7
			value2[i] = 0.1
			value3[i] = 0.07
			value4[i] = 0.3
		case 2:
			value[i] = 0.11
			value2[i] = 0.1
			value3[i] = 0.09
			value4[i] = 0.3
		case 3:
			value[i] = 0.4
			value2[i] = 0.9
			value3[i] = 0.1
			value4[i] = 0.9
		}
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	encoder.Encode(value2, pt)
	ct2, _ := encryptor.EncryptNew(pt)
	encoder.Encode(value3, pt)
	ct3, _ := encryptor.EncryptNew(pt)
	encoder.Encode(value4, pt)
	ct4, _ := encryptor.EncryptNew(pt)

	cts := make([]*rlwe.Ciphertext, tweakN)
	for i := range cts {
		switch i % 4 {
		case 0:
			cts[i] = ct.CopyNew()
		case 1:
			cts[i] = ct2.CopyNew()
		case 2:
			cts[i] = ct3.CopyNew()
		case 3:
			cts[i] = ct4.CopyNew()
		}
	}
	fmt.Println("Tweak start")
	results := Tweak2(cts, params, evaluator, encoder, tweakN)

	fmt.Println("Tweak End", len(results))

	reval := make([][]float64, tweakN)
	for i := range reval {
		reval[i] = make([]float64, 2*n)
		dept := decryptor.DecryptNew(results[i])
		err := encoder.Decode(dept, reval[i])
		if err != nil {
			fmt.Println(err)
		}
	}

	for i := range tweakN {
		fmt.Println(reval[i][0:10])
	}

	// for i := range reval {
	// 	for j := range reval {
	// 		if math.Round(reval[i][j]) != 0 {
	// 			fmt.Println(i, ", ", j, " : ", reval[i][j])
	// 		}
	// 	}
	// }
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
