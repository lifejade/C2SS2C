package test

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/lifejade/mm/src/matmult"
	cwrappingflint "github.com/lifejade/mm/src/matmult/cwrapping_flint"
	"github.com/lifejade/mm/src/transpose"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

func Test_ppmm_time(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            12,
		LogQ:            []int{50, 36, 36, 36},
		LogP:            []int{50},
		Xs:              ring.Ternary{H: 256},
		LogDefaultScale: 28,
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

	n := 1 << params.LogN()
	fmt.Println("N is this  : ", n)
	value := make([]float64, n)
	for i, _ := range value {
		value[i] = 0.1
	}

	pt := hefloat.NewPlaintext(params, 1)
	pt.IsBatched = false

	encoder.Encode(value, pt)
	cts := make([]*rlwe.Ciphertext, n)
	for i := range cts {
		cts[i], _ = encryptor.EncryptNew(pt.CopyNew())
	}

	u := make([][]uint64, n)
	for i := range n {
		u[i] = make([]uint64, n)
		for j := range n {
			u[i][j] = uint64(j % 11)
		}
	}
	fmt.Println("ppmm time : ", elapse)

	fmt.Println("start flint")
	starttime = time.Now()
	result2 := matmult.PPMM_Flint(cts, u, params, n)
	elapse = time.Since(starttime)
	fmt.Println("ppmm time : ", elapse)

	values := make([][]float64, n)

	for i := range values {
		values[i] = make([]float64, n)
		dept := decryptor.DecryptNew(result2[i])
		encoder.Decode(dept, values[i])
	}

	fmt.Println(values[0][0])

	fmt.Println(values[200][700])

	fmt.Println(values[800][3])
}

func Test_MM(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	n := 1 << 5
	Q := uint64(1) << 50

	u := make([][]uint64, n)
	a := make([][]uint64, n)

	for i := range u {
		u[i] = make([]uint64, n)
		for j := range u[i] {
			u[i][j] = (uint64(1) << 40) - 5
		}
	}

	for i := range a {
		a[i] = make([]uint64, n)
		for j := range a[i] {
			a[i][j] = (uint64(1) << 40) - 5
		}
	}
	result := cwrappingflint.Mult_mod_mat(u, a, n, n, n, Q)
	for i := range result {
		for j := range result[i] {
			fmt.Print(result[i][j], " ")
		}
		fmt.Println()
	}
}

func Test_PCMM(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            10,
		LogQ:            []int{32, 32, 40, 40, 30, 30},
		LogP:            []int{50},
		Xs:              ring.Ternary{H: 256},
		LogDefaultScale: 28,
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
	n := 1 << params.LogN()
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

	fmt.Println("N is this  : ", n)
	value := make([]float64, n)
	for i, _ := range value {
		value[i] = 0.0001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, 1)
	pt.IsBatched = false

	encoder.Encode(value, pt)
	cts := make([]*rlwe.Ciphertext, n)
	for i := range cts {
		cts[i], _ = encryptor.EncryptNew(pt.CopyNew())
	}
	ctT := transpose.Transpose(cts, params, evaluator, encoder, n)
	_ = ctT

	k := float64(1 << 28)
	u := make([][][]uint64, len(params.Q()))
	for q := range u {
		u[q] = make([][]uint64, n)
		fmt.Println("q : ", params.Q()[q])
		for i := range n {
			u[q][i] = make([]uint64, n)
			for j := range n {
				u[q][i][j] = params.Q()[q] - uint64(0.5*k)
				//u[q][i][j] = uint64(1 * k)
				fmt.Print(u[q][i][j], " ")
			}
			fmt.Println()
		}

	}

	fmt.Println("ppmm time : ", elapse)

	fmt.Println("start flint")
	starttime = time.Now()
	debugCTS(cts, params, encoder, decryptor)
	result2 := matmult.PPMM_Flint_CRT(cts, u, params, n)
	debugCTS(result2, params, encoder, decryptor)
	elapse = time.Since(starttime)
	fmt.Println("ppmm time : ", elapse)

	for i := range result2 {
		evaluator.Mul(result2[i], 1/k, result2[i])
		evaluator.Rescale(result2[i], result2[i])
	}

	values := make([][]float64, n)

	for i := range values {
		values[i] = make([]float64, n)
		dept := decryptor.DecryptNew(result2[i])
		encoder.Decode(dept, values[i])

		fmt.Println(values[i])
	}
	fmt.Println(result2[0].LogScale())

	fmt.Println(MaxUint64Slice(result2[0].Value[1].Coeffs[0]))
}

func Test_NegMultConst(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            5,
		LogQ:            []int{60, 40, 40, 40, 40, 40},
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
	n := 1 << params.LogN()
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

	fmt.Println("N is this  : ", n)
	value := make([]float64, n)
	for i, _ := range value {
		value[i] = -0.0016 + 0.0001*float64(i)
	}

	pt := hefloat.NewPlaintext(params, 1)
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)

	k := float64(1 << 20)
	u := make([]uint64, len(params.Q()))
	for q := range u {
		u[q] = params.Q()[q] - uint64(0.5*k)
	}
	ringQ := params.RingQ().AtLevel(ct.Level())
	ringQ.INTT(ct.Value[0], ct.Value[0])
	ringQ.INTT(ct.Value[1], ct.Value[1])

	ringQ.MForm(ct.Value[0], ct.Value[0])
	ringQ.MForm(ct.Value[1], ct.Value[1])

	ringQ.MulRNSScalarMontgomery(ct.Value[0], u, ct.Value[0])
	ringQ.MulRNSScalarMontgomery(ct.Value[1], u, ct.Value[1])

	ringQ.NTT(ct.Value[0], ct.Value[0])
	ringQ.NTT(ct.Value[1], ct.Value[1])

	evaluator.Mul(ct, 1/k, ct)
	evaluator.Rescale(ct, ct)

	values := make([]float64, n)

	dept := decryptor.DecryptNew(ct)
	encoder.Decode(dept, values)

	fmt.Println(values)
}

func Test_NegMultVec(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            5,
		LogQ:            []int{60, 40, 40, 40, 40, 40},
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
	n := 1 << params.LogN()
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

	fmt.Println("N is this  : ", n)
	value := make([]float64, n)
	for i, _ := range value {
		value[i] = 0.00001
	}

	pt := hefloat.NewPlaintext(params, 1)
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	ringQ := params.RingQ().AtLevel(ct.Level())

	op := ringQ.NewPoly()
	k := float64(1 << 40)
	for q := range op.Coeffs {
		for n_ := range op.Coeffs[q] {
			op.Coeffs[q][n_] = params.Q()[q] - uint64(k)
			//op.Coeffs[q][n_] = uint64(k)
		}
	}
	fmt.Println(len(op.Coeffs))
	fmt.Println(len(op.Coeffs[0]))
	ringQ.NTT(op, op)

	// ringQ.INTT(ct.Value[0], ct.Value[0])
	// ringQ.INTT(ct.Value[1], ct.Value[1])

	ringQ.MForm(ct.Value[0], ct.Value[0])
	ringQ.MForm(ct.Value[1], ct.Value[1])

	ringQ.MulCoeffsMontgomery(ct.Value[0], op, ct.Value[0])
	ringQ.MulCoeffsMontgomery(ct.Value[1], op, ct.Value[1])

	// ringQ.NTT(ct.Value[0], ct.Value[0])
	// ringQ.NTT(ct.Value[1], ct.Value[1])

	evaluator.Mul(ct, 1/k, ct)
	evaluator.Rescale(ct, ct)

	values := make([]float64, n)

	dept := decryptor.DecryptNew(ct)
	encoder.Decode(dept, values)

	fmt.Println(values)
}

func MaxUint64Slice(arr []uint64) uint64 {
	if len(arr) == 0 {
		panic("slice is empty")
	}
	max := arr[0]
	for _, v := range arr[1:] {
		if v > max {
			max = v
		}
	}
	return max
}
