package test

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

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
