package test

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"

	"github.com/lifejade/mm/src/matmult"
	"github.com/lifejade/mm/src/transpose"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/schemes/ckks"

	"time"
)

func Test_Basic(t *testing.T) {
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
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
		LogP:            []int{51, 51, 51},
		Xs:              ring.Ternary{H: 192},
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
	for i, _ := range value {
		value[i] = 1
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	ct2, _ := evaluator.MulNew(ct, pt)

	pt2 := decryptor.DecryptNew(ct2)

	value2 := make([]float64, 2*n)
	encoder.Decode(pt2, value2)
	for i := range 10 {
		fmt.Println(value2[i])
	}
	fmt.Println(pt2.IsBatched)
}

func Test_Basic2(t *testing.T) {
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
		LogQ:            []int{51, 46, 46},
		LogP:            []int{51, 51, 51},
		Xs:              ring.Ternary{H: 192},
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
	for i, _ := range value {
		value[i] = 1
		if i%2 == 0 {
			value[i] = 2
		}
	}

	pt := hefloat.NewPlaintext(params, 0)
	pt.IsBatched = false

	encoder.Encode(value, pt)

	params.RingQ().AtLevel(pt.Level()).INTT(pt.Value, pt.Value)

	fmt.Println(pt.IsMontgomery)
	fmt.Println(pt.Scale.BigInt())
	fmt.Println(pt.Value.Coeffs[0][1])
	fmt.Println(pt.Value.Coeffs[0][2])

	value2 := make([]float64, 2*n)
	encoder.Decode(pt, value2)
	// for i := range 10 {
	// 	fmt.Println(value2[i])
	// }
	fmt.Println(pt.IsBatched)
}

func Test_CipherMemory(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// CIFAR-10
		// index [0]
		// logN = 13, full slots
		// logq = 51, logp = 46
		// scale = 1<<46
		// # special modulus = 3
		// # available levels =
		LogN:            13,
		LogQ:            []int{51, 46, 46, 46},
		LogP:            []int{60},
		Xs:              ring.Ternary{H: 192},
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

	for i := range 12 {
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
	fmt.Println("slot = ", n)
	value := make([]float64, 2*n)
	for i, _ := range value {
		value[i] = 1
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	starttime = time.Now()
	encoder.Encode(value, pt)
	cts := make([]*rlwe.Ciphertext, n)
	for i, _ := range cts {
		cts[i], _ = encryptor.EncryptNew(pt)
	}
	elapse = time.Since(starttime)
	fmt.Println("cipher make time : ", elapse)
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	fmt.Printf("Alloc = %v KB\n", memStats.Alloc/1024)
	fmt.Printf("TotalAlloc = %v KB\n", memStats.TotalAlloc/1024)
	fmt.Printf("Sys = %v KB\n", memStats.Sys/1024)
	fmt.Printf("NumGC = %v\n", memStats.NumGC)
}

func Test_GaloisRemove(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// CIFAR-10
		// index [0]
		// logN = 13, full slots
		// logq = 51, logp = 46
		// scale = 1<<46
		// # special modulus = 3
		// # available levels =
		LogN:            13,
		LogQ:            []int{51, 46, 46, 46},
		LogP:            []int{60},
		Xs:              ring.Ternary{H: 192},
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

	for i := range params.LogMaxSlots() {
		fmt.Println(params.GaloisElement((1 << i)))
		galEls = append(galEls, params.GaloisElement((1 << i)))
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
	fmt.Println("slot = ", n)
	value := make([]float64, 2*n)
	for i, _ := range value {
		value[i] = 0.5
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false
	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	for i := range params.LogMaxSlots() {
		temp, _ := evaluator.RotateNew(ct, 1<<(params.LogMaxSlots()-1-i))
		evaluator.Add(ct, temp, ct)
	}
	fmt.Println()

	dept := decryptor.DecryptNew(ct)
	devalue := make([]float64, 2*n)
	encoder.Decode(dept, devalue)
	for i := range n {
		if devalue[i] > 1 {
			fmt.Println("over : ", devalue[i], " ", i, "idx")
		}
	}

	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	fmt.Printf("Alloc = %v KB\n", memStats.Alloc/1024)
	fmt.Printf("TotalAlloc = %v KB\n", memStats.TotalAlloc/1024)
	fmt.Printf("Sys = %v KB\n", memStats.Sys/1024)
	fmt.Printf("NumGC = %v\n", memStats.NumGC)
}

func Test_PPMM(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// CIFAR-10
		// index [0]
		// logN = 13, full slots
		// logq = 51, logp = 46
		// scale = 1<<46
		// # special modulus = 3
		// # available levels =
		LogN:            13,
		LogQ:            []int{51, 46, 46, 46},
		LogP:            []int{60},
		Xs:              ring.Ternary{H: 192},
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

	for i := range params.LogMaxSlots() {
		galEls = append(galEls, params.GaloisElement((1 << i)))
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

	n := 1 << params.LogN()
	value := make([]float64, n)
	for i := range n {
		value[i] = 1
	}

	pt := hefloat.NewPlaintext(params, 1)
	pt.IsBatched = false
	encoder.Encode(value, pt)

	ct, _ := encryptor.EncryptNew(pt)
	params.RingQ().AtLevel(ct.Level()).INTT(ct.Value[0], ct.Value[0])
	params.RingQ().AtLevel(ct.Level()).INTT(ct.Value[1], ct.Value[1])
	cts := make([]*rlwe.Ciphertext, n)

	a := make([]*[]uint64, n)
	b := make([]*[]uint64, n)
	for i := range cts {
		cts[i] = ct.CopyNew()
		a[i] = &(cts[i].Value[0].Coeffs[0])
		b[i] = &(cts[i].Value[0].Coeffs[0])
	}

	multiply(n, a, b)
}

func Transpose_check(inputs []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, encoder *hefloat.Encoder, n int) []*rlwe.Ciphertext {
	cts := make([]*rlwe.Ciphertext, n)
	ringQ := params.RingQ().AtLevel(inputs[0].Level())
	ninv := ringQ.NewRNSScalarFromUInt64(uint64(n))
	ringQ.MFormRNSScalar(ninv, ninv)
	ringQ.Inverse(ninv)

	for i := range cts {
		cts[i] = inputs[i].CopyNew()
		ringQ.INTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.INTT(cts[i].Value[1], cts[i].Value[1])

		ringQ.MultByMonomial(cts[i].Value[0], i, cts[i].Value[0])
		ringQ.MultByMonomial(cts[i].Value[1], i, cts[i].Value[1])

		ringQ.NTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.NTT(cts[i].Value[1], cts[i].Value[1])

	}
	aux := Tweak2_check(cts, params, eval, encoder, n)
	res := make([]*rlwe.Ciphertext, n)
	for i := range res {
		idx, ch := transpose.ModInv(uint64(2*i+1), uint64(2*n))
		if !ch {
			fmt.Println("err ", i, " ", idx)
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

		if err := eval.Automorphism(res[i], uint64(2*i+1), res[i]); err != nil {
			fmt.Println(err)
		}

	}

	res2 := Tweak2_check(res, params, eval, encoder, n)
	result := make([]*rlwe.Ciphertext, n)
	for i := range n {
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

		result[(n-i)%(n)] = res2[i]
	}

	return result
}

func Tweak2_check(cts []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, encoder *hefloat.Encoder, n int) []*rlwe.Ciphertext {
	if n == 1 {
		return cts
	}

	cts_ := make([]*rlwe.Ciphertext, n)
	cts_[0] = cts[0].CopyNew()
	logn := int(math.Round(math.Log2(float64(n))))

	ringQ := params.RingQ().AtLevel(cts[0].Level())
	for l := range logn {
		powl := 1 << l
		temp := make([]*rlwe.Ciphertext, powl)
		for j := range powl {
			temp[j] = cts[((2*j+1)*n)/(powl*2)].CopyNew()
		}
		aux := Tweak2_check(temp, params, eval, encoder, powl)
		for j := range powl {

			//tmp, _ := eval.MulNew(aux[j], pt)
			tmp := aux[j].CopyNew()

			ringQ.INTT(tmp.Value[0], tmp.Value[0])
			ringQ.INTT(tmp.Value[1], tmp.Value[1])
			ringQ.MultByMonomial(tmp.Value[0], params.MaxSlots()*2/powl*j, tmp.Value[0])
			ringQ.MultByMonomial(tmp.Value[1], params.MaxSlots()*2/powl*j, tmp.Value[1])

			ringQ.NTT(tmp.Value[0], tmp.Value[0])
			ringQ.NTT(tmp.Value[1], tmp.Value[1])
			//eval.Rescale(tmp, tmp)

			var err error
			cts_[j+powl], err = eval.SubNew(cts_[j], tmp)
			if err != nil {
				fmt.Println(err)
			}
			cts_[j], err = eval.AddNew(cts_[j], tmp)
			if err != nil {
				fmt.Println(err)
			}

		}
	}

	return cts_
}

func Test_imsadprof(t *testing.T) {
	PN16QP1761 := ckks.ParametersLiteral{
		LogN: 16,
		Q: []uint64{0x80000000080001, 0x2000000a0001, 0x2000000e0001, 0x1fffffc20001,
			0x200000440001, 0x200000500001, 0x200000620001, 0x1fffff980001,
			0x2000006a0001, 0x1fffff7e0001, 0x200000860001, 0x200000a60001,
			0x200000aa0001, 0x200000b20001, 0x200000c80001, 0x1fffff360001,
			0x200000e20001, 0x1fffff060001, 0x200000fe0001, 0x1ffffede0001,
			0x1ffffeca0001, 0x1ffffeb40001, 0x200001520001, 0x1ffffe760001,
			0x2000019a0001, 0x1ffffe640001, 0x200001a00001, 0x1ffffe520001,
			0x200001e80001, 0x1ffffe0c0001, 0x1ffffdee0001, 0x200002480001,
			0x1ffffdb60001, 0x200002560001},
		P:               []uint64{0x80000000440001, 0x7fffffffba0001, 0x80000000500001, 0x7fffffffaa0001},
		LogDefaultScale: 45,
	}
	params, _ := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral(PN16QP1761))
	fmt.Printf("logN=%d, MaxLevel=%d, LogDefaultScale=%d (PREC mode auto)\n",
		params.LogN(), params.MaxLevel(), params.LogDefaultScale())
	fmt.Print(params.LogQ())

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
	_ = decryptor

	fmt.Println("generate Evaluator end")
	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	fmt.Println("size of ct : ", ct.BinarySize())
	cts := make([]*rlwe.Ciphertext, 2*n)
	for i := range 2 * n {
		fmt.Println("idx : ", i)
		cts[i] = ct.CopyNew()
	}
	fmt.Println("ct gen end")
	// fmt.Println("cts level :  ", cts[0].Level())
	// fmt.Println("transpose, maxlevel")
	// starttime = time.Now()
	// transpose.Transpose(cts, params, evaluator, encoder, 2*n)
	// elapse = time.Since(starttime)
	// fmt.Println(elapse)

	_, SFI := matmult.GenSFMat(params)
	scale := float64(1 << 40)
	mat0, _, _, _ := matmult.GenC2SMat(SFI, scale, params)

	fmt.Println("ppmm, maxlevel")
	starttime = time.Now()
	res0 := matmult.PPMM_Flint_CRT(cts, mat0, params, 2*n)
	for i := range res0 {
		evaluator.Mul(res0[i], 1.0/(scale), res0[i])
		evaluator.Rescale(res0[i], res0[i])
	}
	elapse = time.Since(starttime)
	fmt.Println(elapse)

	fmt.Println("ppmm, maxlevel - 1")
	starttime = time.Now()
	res0 = matmult.PPMM_Flint_CRT(res0, mat0, params, 2*n)
	for i := range res0 {
		evaluator.Mul(res0[i], 1.0/(scale), res0[i])
		evaluator.Rescale(res0[i], res0[i])
	}
	elapse = time.Since(starttime)
	fmt.Println(elapse)

	fmt.Println("ppmm, maxlevel - 2")
	starttime = time.Now()
	res0 = matmult.PPMM_Flint_CRT(res0, mat0, params, 2*n)
	for i := range res0 {
		evaluator.Mul(res0[i], 1.0/(scale), res0[i])
		evaluator.Rescale(res0[i], res0[i])
	}
	elapse = time.Since(starttime)
	fmt.Println(elapse)

	for i := range 2 * n {
		evaluator.DropLevel(cts[i], cts[i].Level()-3)
	}

	// fmt.Println("cts level :  ", res0[0].Level())
	// fmt.Println("transpose, level = 3")
	// starttime = time.Now()
	// transpose.Transpose(res0, params, evaluator, encoder, 2*n)
	// elapse = time.Since(starttime)
	// fmt.Println(elapse)

	fmt.Println("ppmm, level = 3")
	starttime = time.Now()
	res0 = matmult.PPMM_Flint_CRT(res0, mat0, params, 2*n)
	for i := range res0 {
		evaluator.Mul(res0[i], 1.0/(scale), res0[i])
		evaluator.Rescale(res0[i], res0[i])
	}
	elapse = time.Since(starttime)
	fmt.Println(elapse)

	fmt.Println("ppmm, level = 2")
	starttime = time.Now()
	res0 = matmult.PPMM_Flint_CRT(res0, mat0, params, 2*n)
	for i := range res0 {
		evaluator.Mul(res0[i], 1.0/(scale), res0[i])
		evaluator.Rescale(res0[i], res0[i])
	}
	elapse = time.Since(starttime)
	fmt.Println(elapse)

	fmt.Println("ppmm, level = 1")
	starttime = time.Now()
	res0 = matmult.PPMM_Flint_CRT(res0, mat0, params, 2*n)
	for i := range res0 {
		evaluator.Mul(res0[i], 1.0/(scale), res0[i])
		evaluator.Rescale(res0[i], res0[i])
	}
	elapse = time.Since(starttime)
	fmt.Println(elapse)
}

func multiply(n int, a, b []*[]uint64) []*[]uint64 {
	result := make([][]uint64, n)
	resultPtrs := make([]*[]uint64, n)
	fmt.Println("result init")
	for i := 0; i < n; i++ {
		result[i] = make([]uint64, n)
		resultPtrs[i] = &result[i]
	}

	for i := 0; i < n; i++ {
		if i%4 == 0 {
			fmt.Println("if i % 4==0...", i)

		}
		for j := 0; j < n; j++ {
			var sum uint64 = 0
			for k := 0; k < n; k++ {
				sum += (*a[i])[k] * (*b[k])[j]
			}
			result[i][j] = sum
		}
	}

	return resultPtrs
}

func modPow(a, b, n int) int {
	result := 1
	a = a % n
	for b > 0 {
		if b%2 == 1 {
			result = (result * a) % n
		}
		a = (a * a) % n
		b = b / 2
	}
	return result
}
