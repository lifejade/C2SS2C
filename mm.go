package main

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/lifejade/mm/src/matmult"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"

	"time"
)

func main() {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            12,
		LogQ:            []int{60, 36},
		LogP:            []int{60},
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
	value := make([]float64, n)
	for i, _ := range value {
		value[i] = 1
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
			u[i][j] = 1
		}
	}

	result := matmult.PPMM(cts, u, params, n)

	values := make([][]float64, n)
	for i := range values {
		values[i] = make([]float64, n)
		dept := decryptor.DecryptNew(result[i])
		encoder.Decode(dept, values[i])
	}

}
