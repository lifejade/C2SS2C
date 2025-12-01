package test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
	"testing"

	"github.com/lifejade/mm/src/matmult"
	"github.com/lifejade/mm/src/transpose"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"

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

	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{48, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}
	params, _ := hefloat.NewParametersFromLiteral(SchemeParams)
	fmt.Printf("logN=%d, MaxLevel=%d, LogDefaultScale=%d (PREC mode auto)\n",
		params.LogN(), params.MaxLevel(), params.LogDefaultScale())
	fmt.Print(params.LogQ())

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

	llen := 1 << 7
	scale := float64(1 << 40)
	mat0 := make([][]uint64, len(params.Q()))
	for l := range mat0 {
		mat0[l] = make([]uint64, llen*llen)
		for i := range llen {
			for j := range llen {
				mat0[l][i*llen+j] = uint64(rand.Float64() * scale)
			}
		}
	}
	llen = 1 << 8
	mat1 := make([][]uint64, len(params.Q()))
	for l := range mat1 {
		mat1[l] = make([]uint64, llen*llen)
		for i := range llen {
			for j := range llen {
				mat1[l][i*llen+j] = uint64(rand.Float64() * scale)
			}
		}
	}
	fmt.Println("mat init end")

	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, 14)
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	fmt.Println("size of ct : ", ct.BinarySize())
	cts := make([]*rlwe.Ciphertext, llen)
	for i := range cts {
		cts[i] = ct.CopyNew()
	}
	fmt.Println("ct gen end")

	// fmt.Println("cts level :  ", cts[0].Level())
	// fmt.Println("transpose, maxlevel")
	// starttime = time.Now()
	// transpose.Transpose(cts, params, evaluator, encoder, 2*n)
	// elapse = time.Since(starttime)
	// fmt.Println(elapse)

	fmt.Println("ppmm, maxlevel")
	starttime = time.Now()
	res := matmult.PPMM_Flint_CRT3(cts, mat0, llen, llen, 2*n, params)
	for i := range llen {
		evaluator.Mul(res[i], 1/scale, res[i])
		evaluator.Rescale(res[i], res[i])
	}
	elapse = time.Since(starttime)
	fmt.Println(elapse)
	fmt.Println(res[0].Level())

	// fmt.Println("ppmm, maxlevel - 1")
	// starttime = time.Now()
	// res = matmult.PPMM_Flint_CRT3(res, mat0, llen, llen, 2*n, params)
	// for i := range llen {
	// 	evaluator.Mul(res[i], 1/scale, res[i])
	// 	evaluator.Rescale(res[i], res[i])
	// }
	// elapse = time.Since(starttime)
	// fmt.Println(elapse)
	// fmt.Println(res[0].Level())

	fmt.Println("ppmm, maxlevel - 2")
	starttime = time.Now()
	res = matmult.PPMM_Flint_CRT3(res, mat1, llen, llen, 2*n, params)
	for i := range llen {
		evaluator.Mul(res[i], 1/scale, res[i])
		evaluator.Rescale(res[i], res[i])
	}
	elapse = time.Since(starttime)
	fmt.Println(elapse)
	fmt.Println(res[0].Level())

	// ct_temp := ct.CopyNew()
	// starttime_ := time.Now()
	// ct_coef, ct_coef2, _ := btp.DFTEvaluator.CoeffsToSlotsNew(ct_temp, btp.C2SDFTMatrix)
	// elapse_ := time.Since(starttime_)
	// fmt.Println("cts time(origin) : ", elapse_)

	// starttime_ = time.Now()
	// ct_coef, _ = btp.EvalMod(ct_coef)
	// ct_coef2, _ = btp.EvalMod(ct_coef2)
	// elapse_ = time.Since(starttime_)
	// fmt.Println("eval time(origin) : ", elapse_)

	// starttime_ = time.Now()
	// res, _ := btp.DFTEvaluator.SlotsToCoeffsNew(ct_coef, ct_coef2, btp.S2CDFTMatrix)
	// elapse_ = time.Since(starttime_)
	// fmt.Println("stc time(origin) : ", elapse_)

	// _ = res
}

func Test_imsadprof2(t *testing.T) {

	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            13,
		LogQ:            []int{48, 40, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}
	params, _ := hefloat.NewParametersFromLiteral(SchemeParams)
	fmt.Printf("logN=%d, MaxLevel=%d, LogDefaultScale=%d (PREC mode auto)\n",
		params.LogN(), params.MaxLevel(), params.LogDefaultScale())
	fmt.Println(params.LogQ())
	fmt.Println(params.Q()[0], " ", params.Q()[1])
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
	_, _ = evaluator, decryptor

	fmt.Println("generate Evaluator end")

	llen := 1 << 13
	scale := float64(1 << 40)
	mat0 := make([][]uint64, len(params.Q()))
	for l := range mat0 {
		mat0[l] = make([]uint64, llen*llen)
		for i := range llen {
			for j := range llen {
				mat0[l][i*llen+j] = uint64(rand.Float64() * scale)
			}
		}
	}
	fmt.Println("mat init end")

	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, 1)
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	fmt.Println("size of ct : ", ct.BinarySize())
	cts := make([]*rlwe.Ciphertext, llen)
	for i := range cts {
		cts[i] = ct.CopyNew()
	}
	fmt.Println("ct gen end")

	// fmt.Println("cts level :  ", cts[0].Level())
	// fmt.Println("transpose, maxlevel")
	// starttime = time.Now()
	// transpose.Transpose(cts, params, evaluator, encoder, 2*n)
	// elapse = time.Since(starttime)
	// fmt.Println(elapse)

	fmt.Println("ppmm, maxlevel")
	starttime = time.Now()
	res := matmult.PPMM_Flint_CRT3(cts, mat0, llen, llen, 2*n, params)
	for i := range llen {
		evaluator.Mul(res[i], 1/scale, res[i])
		evaluator.Rescale(res[i], res[i])
	}
	elapse = time.Since(starttime)
	fmt.Println(elapse)
	fmt.Println(res[0].Level())

	// ct_temp := ct.CopyNew()
	// starttime_ := time.Now()
	// ct_coef, ct_coef2, _ := btp.DFTEvaluator.CoeffsToSlotsNew(ct_temp, btp.C2SDFTMatrix)
	// elapse_ := time.Since(starttime_)
	// fmt.Println("cts time(origin) : ", elapse_)

	// starttime_ = time.Now()
	// ct_coef, _ = btp.EvalMod(ct_coef)
	// ct_coef2, _ = btp.EvalMod(ct_coef2)
	// elapse_ = time.Since(starttime_)
	// fmt.Println("eval time(origin) : ", elapse_)

	// starttime_ = time.Now()
	// res, _ := btp.DFTEvaluator.SlotsToCoeffsNew(ct_coef, ct_coef2, btp.S2CDFTMatrix)
	// elapse_ = time.Since(starttime_)
	// fmt.Println("stc time(origin) : ", elapse_)

	// _ = res
}

func Test_imsadprof1022(t *testing.T) {
	runtime.GOMAXPROCS(1) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	if pc, _, _, _ := runtime.Caller(0); pc != 0 {
		fmt.Println(runtime.FuncForPC(pc).Name())
	}
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{48, 40, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}
	params, _ := hefloat.NewParametersFromLiteral(SchemeParams)
	fmt.Printf("logN=%d, MaxLevel=%d, LogDefaultScale=%d (PREC mode auto)\n",
		params.LogN(), params.MaxLevel(), params.LogDefaultScale())
	fmt.Println(params.LogQ())
	fmt.Println(params.Q()[0], " ", params.Q()[1])
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
	_, _ = evaluator, decryptor

	fmt.Println("generate Evaluator end")

	fmt.Println("ckks parameter init end")
	Q := params.Q()
	P := []uint64{4107427, 3868699, 4073143, 3639397, 3835109, 3377447, 3338903, 3314141, 3816173, 3731251, 3925091, 3500261, 3507403, 3368353, 3598601, 3637573, 3387523, 3489259, 3804751, 4002811, 3417251, 3245357, 3659177, 4047647, 3367981, 3984439, 3621473, 3565147, 3789193, 3174547, 3293959, 3567803, 3856499, 3299617, 3939619, 4004683, 3803347, 3501467, 3518719, 3631919}
	PLevel := 35
	ringQ, _ := ring.NewRing(params.N(), Q)
	ringP, _ := ring.NewRing(params.N(), P[:PLevel+1])
	be := matmult.NewBasisExtender(ringQ, ringP, []matmult.Key{{0, PLevel}}, []matmult.Key{{PLevel, params.MaxLevel()}})

	value := make([]float64, 2*n)
	for i, _ := range value {
		value[i] = sampling.RandFloat64(-1, 1)
	}

	pt := hefloat.NewPlaintext(params, 0)
	pt.IsBatched = false
	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)

	ringQ.AtLevel(0).INTT(ct.Value[0], ct.Value[0])
	ringQ.AtLevel(0).INTT(ct.Value[1], ct.Value[1])

	ringiters := make([]ring.Poly, 2)
	for idx := range ringiters {
		ringiters[idx] = ringP.NewPoly()
	}
	fmt.Println("//////////////////////////////////////////////////////////////")
	var totaltime time.Duration

	starttime = time.Now()
	be.ModSwitchQtoP(0, PLevel, ct.Value[0], ringiters[0])
	be.ModSwitchQtoP(0, PLevel, ct.Value[1], ringiters[1])
	elapse = time.Since(starttime)
	fmt.Println("Q to P time: ", math.Round(((elapse*(1<<16)).Seconds())*100)/100)
	totaltime += elapse * (1 << 16)
	startLevels := 14
	levelstep := 2
	scale := float64(1 << 40)
	sizes := []int{1 << 8, 1 << 7}

	sc := rlwe.NewScale(1)
	for i := range levelstep {
		q := rlwe.NewScale(params.Q()[startLevels-i])
		sc = sc.Mul(q)
	}

	for step := range levelstep {
		var steptime time.Duration
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
		for idx := range 2 {
			for i := range size {
				rings[idx][i] = ringiters[idx]
			}
		}

		time_ := time.Now()
		for idx := range 2 {
			matmult.PPMM_Blas_CRT(rings[idx], u, params, size, size, params.N(), PLevel+1, ringP, rings[idx])
		}
		elapse_ := time.Since(time_)
		fmt.Println("ppmm", elapse_)
		steptime += elapse_
		for idx := range 2 {
			ringiters[idx] = rings[idx][0]
		}
		if step == 0 {
			fmt.Println("steptime : ", steptime)
			steptime = steptime * 8 * (1 << 7)
			fmt.Println("steptotaltime : ", math.Round(steptime.Seconds()*100)/100)
		} else {
			fmt.Println("steptime : ", steptime)
			steptime = steptime * 8 * (1 << 8)
			fmt.Println("steptotaltime : ", math.Round(steptime.Seconds()*100)/100)
		}
		totaltime += steptime
	}
	ct.Resize(1, 14)
	time_ := time.Now()
	be.ModSwitchPtoQ(PLevel, params.MaxLevel(), ringiters[0], ct.Value[0])
	be.ModSwitchPtoQ(PLevel, params.MaxLevel(), ringiters[1], ct.Value[1])
	elapse_ := time.Since(time_)
	fmt.Println("P to Q time: ", math.Round(((elapse_*(1<<16)).Seconds())*100)/100)
	totaltime += elapse_ * (1 << 16)

	sscale := 1.0
	for range levelstep {
		sscale *= scale
	}
	time_ = time.Now()
	Mul2_(evaluator, ct, 1/(sscale), ct, sc)
	Rescale_NonNTT(evaluator, ct, ct)
	Rescale_NonNTT(evaluator, ct, ct)
	elapse_ = time.Since(time_)
	fmt.Println("rescale", math.Round((elapse_*(1<<16)).Seconds()*100)/100)
	totaltime += elapse_ * (1 << 16)

	// time_ = time.Now()
	// cts := make([]*rlwe.Ciphertext, 2*n)
	// for i := range cts {
	// 	cts[i] = ct
	// }
	// cts = transpose.Transpose2(cts, params, evaluator, 2*n)
	// elapse_ = time.Since(time_)
	// fmt.Println("transpose time: ", elapse_)
	// totaltime += elapse_

	// ct = cts[0]
	fmt.Println("totaltime (cal.) : ", math.Round(totaltime.Seconds()*100)/100)
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

func Test_CheckTimePCMMSparse(t *testing.T) {
	runtime.GOMAXPROCS(1) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	if pc, _, _, _ := runtime.Caller(0); pc != 0 {
		fmt.Println(runtime.FuncForPC(pc).Name())
	}
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{48, 40, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}
	params, _ := hefloat.NewParametersFromLiteral(SchemeParams)
	fmt.Printf("logN=%d, MaxLevel=%d, LogDefaultScale=%d (PREC mode auto)\n",
		params.LogN(), params.MaxLevel(), params.LogDefaultScale())
	fmt.Println(params.LogQ())
	fmt.Println(params.Q()[0], " ", params.Q()[1])
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
	_, _ = evaluator, decryptor

	fmt.Println("generate Evaluator end")

	fmt.Println("ckks parameter init end")
	Q := params.Q()
	P := []uint64{4107427, 3868699, 4073143, 3639397, 3835109, 3377447, 3338903, 3314141, 3816173, 3731251, 3925091, 3500261, 3507403, 3368353, 3598601, 3637573, 3387523, 3489259, 3804751, 4002811, 3417251, 3245357, 3659177, 4047647, 3367981, 3984439, 3621473, 3565147, 3789193, 3174547, 3293959, 3567803, 3856499, 3299617, 3939619, 4004683, 3803347, 3501467, 3518719, 3631919}
	PLevel := 35
	ringQ, _ := ring.NewRing(params.N(), Q)
	ringP, _ := ring.NewRing(params.N(), P[:PLevel+1])
	be := matmult.NewBasisExtender(ringQ, ringP, []matmult.Key{{0, PLevel}}, []matmult.Key{{PLevel, params.MaxLevel()}})

	value := make([]float64, 2*n)
	for i, _ := range value {
		value[i] = sampling.RandFloat64(-1, 1)
	}

	pt := hefloat.NewPlaintext(params, 0)
	pt.IsBatched = false
	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)

	ringQ.AtLevel(0).INTT(ct.Value[0], ct.Value[0])
	ringQ.AtLevel(0).INTT(ct.Value[1], ct.Value[1])

	ringiters := make([]ring.Poly, 2)
	for idx := range ringiters {
		ringiters[idx] = ringP.NewPoly()
	}
	fmt.Println("//////////////////////////////////////////////////////////////")
	var totaltime time.Duration

	starttime = time.Now()
	be.ModSwitchQtoP(0, PLevel, ct.Value[0], ringiters[0])
	be.ModSwitchQtoP(0, PLevel, ct.Value[1], ringiters[1])
	elapse = time.Since(starttime)
	fmt.Println("Q to P time: ", math.Round(((elapse*(1<<16)).Seconds())*100)/100)
	totaltime += elapse * (1 << 16)
	startLevels := 14
	levelstep := 2
	scale := float64(1 << 40)
	sizes := []int{1 << 8, 1 << 7}

	sc := rlwe.NewScale(1)
	for i := range levelstep {
		q := rlwe.NewScale(params.Q()[startLevels-i])
		sc = sc.Mul(q)
	}

	for step := range levelstep {
		var steptime time.Duration
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
		for idx := range 2 {
			for i := range size {
				rings[idx][i] = ringiters[idx]
			}
		}

		time_ := time.Now()
		for idx := range 2 {
			matmult.PPMM_Blas_CRT(rings[idx], u, params, size, size, params.N(), PLevel+1, ringP, rings[idx])
		}
		elapse_ := time.Since(time_)
		fmt.Println("ppmm", elapse_)
		steptime += elapse_
		for idx := range 2 {
			ringiters[idx] = rings[idx][0]
		}
		if step == 0 {
			fmt.Println("steptime : ", steptime)
			steptime = steptime * 8 * (1 << 7)
			fmt.Println("steptotaltime : ", math.Round(steptime.Seconds()*100)/100)
		} else {
			fmt.Println("steptime : ", steptime)
			steptime = steptime * 8 * (1 << 8)
			fmt.Println("steptotaltime : ", math.Round(steptime.Seconds()*100)/100)
		}
		totaltime += steptime
	}
	ct.Resize(1, 14)
	time_ := time.Now()
	be.ModSwitchPtoQ(PLevel, params.MaxLevel(), ringiters[0], ct.Value[0])
	be.ModSwitchPtoQ(PLevel, params.MaxLevel(), ringiters[1], ct.Value[1])
	elapse_ := time.Since(time_)
	fmt.Println("P to Q time: ", math.Round(((elapse_*(1<<16)).Seconds())*100)/100)
	totaltime += elapse_ * (1 << 16)

	sscale := 1.0
	for range levelstep {
		sscale *= scale
	}
	time_ = time.Now()
	Mul2_(evaluator, ct, 1/(sscale), ct, sc)
	Rescale_NonNTT(evaluator, ct, ct)
	Rescale_NonNTT(evaluator, ct, ct)
	elapse_ = time.Since(time_)
	fmt.Println("rescale", math.Round((elapse_*(1<<16)).Seconds()*100)/100)
	totaltime += elapse_ * (1 << 16)

	// time_ = time.Now()
	// cts := make([]*rlwe.Ciphertext, 2*n)
	// for i := range cts {
	// 	cts[i] = ct
	// }
	// cts = transpose.Transpose2(cts, params, evaluator, 2*n)
	// elapse_ = time.Since(time_)
	// fmt.Println("transpose time: ", elapse_)
	// totaltime += elapse_

	// ct = cts[0]
	fmt.Println("totaltime (cal.) : ", math.Round(totaltime.Seconds()*100)/100)
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
