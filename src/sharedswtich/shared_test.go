package sharedswtich

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

func Test_Switch(t *testing.T) {
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
		LogQ:            []int{51, 46, 46, 46, 46, 46},
		LogP:            []int{51, 51},
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

	n := 1 << params.LogMaxSlots()

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

	value := make([]float64, 2*n)
	for i, _ := range value {
		value[i] = float64(i) * 0.001
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)

	// cts := make([]*rlwe.Ciphertext, 2 * n)
	// for i := range cts {
	// 	cts[i], _ = encryptor.EncryptNew(pt)
	// }

	sk2 := kgen.GenSecretKeyNew()
	evk_ := kgen.GenEvaluationKeyNew(sk, sk2)
	fmt.Println(len(params.Q()), len(params.P()))
	fmt.Println(len(evk_.GadgetCiphertext.Value))
	fmt.Println(len(evk_.GadgetCiphertext.Value[0]))
	fmt.Println(len(evk_.GadgetCiphertext.Value[0][0]))
	fmt.Println(evk_.GadgetCiphertext.BaseTwoDecomposition)

	for i := range 3 {
		fmt.Println(evk_.GadgetCiphertext.Value[i][0][0].Q.Coeffs[0][0])
	}

	st := time.Now()
	ct2, _ := evaluator.ApplyEvaluationKeyNew(ct, evk_)
	el := time.Since(st)
	fmt.Println(el)

	value2 := make([]float64, len(value))
	decryptor_ := rlwe.NewDecryptor(params, sk2)
	decryptor_.Decrypt(ct2, pt)
	encoder.Decode(pt, value2)

	check := true
	tol := 0.00001
	for i := range value {
		if math.Abs(value[i]-value2[i]) > tol {
			check = false
			break
		}
	}
	fmt.Println("Eqaul with prev", check)
	fmt.Println("Test_Switch end")
}

func Test_Switch2(t *testing.T) {
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
		LogN:            5,
		LogQ:            []int{51, 46, 46, 46, 46, 46},
		LogP:            []int{51, 51},
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

	n := params.N()
	value := make([]float64, n)
	for i, _ := range value {
		value[i] = float64(i) * 0.001
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	cts := make([]*rlwe.Ciphertext, n)
	for i := range cts {
		cts[i], _ = encryptor.EncryptNew(pt)
	}

	sh_sk := make([]*rlwe.SecretKey, n)
	for i := 0; i < n; i++ {
		sh_sk[i] = kgen.GenSecretKeyNew()
	}

}
