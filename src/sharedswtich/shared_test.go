package sharedswtich

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
		LogN:            5,
		LogQ:            []int{51, 46, 46},
		LogP:            []int{51},
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

	value := make([]float64, 2*n)
	for i, _ := range value {
		value[i] = float64(i) * 0.001
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	ct2 := rlwe.NewCiphertext(params, 1, ct.Level())

	ringQ := params.RingQ().AtLevel(ct.Level())
	_ = ringQ
	//ct.IsNTT = false
	// ringQ.NTT()

	starttime = time.Now()
	evaluator.Automorphism(ct, 5, ct2)
	elapse = time.Since(starttime)
	fmt.Println(elapse)

	pt2 := decryptor.DecryptNew(ct2)

	value2 := make([]float64, 2*n)
	encoder.Decode(pt2, value2)
	fmt.Println(value2[:10])
	fmt.Println(pt2.IsBatched)
	N := params.N()
	NthRoot := params.RingQ().NthRoot()
	tt, _ := ring.AutomorphismNTTIndex(N, NthRoot, 5)
	fmt.Println(tt)
	fmt.Println(NthRoot)
	gkey, _ := evaluator.GetGaloisKey(5)

	fmt.Println("gct len", len(gkey.GadgetCiphertext.Value))
	fmt.Println(len(gkey.GadgetCiphertext.Value[0]))
	fmt.Println(len(gkey.GadgetCiphertext.Value[0][0]))
	// fmt.Println(len(gkey.GadgetCiphertext.Value[0][0][0].Q.Coeffs))
}
