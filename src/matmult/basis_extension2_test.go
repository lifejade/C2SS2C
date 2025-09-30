package matmult

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
)

func Test_Basis2(t *testing.T) {
	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            16,
		LogQ:            []int{48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48},
		LogP:            []int{50},
		Xs:              ring.Ternary{H: 256},
		LogDefaultScale: 40,
	}

	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}
	fmt.Println("ckks parameter init end")
	Q := params.Q()
	P := []uint64{14624959, 15092711, 16654291, 21552221, 16108999, 13227197, 16099607, 14232433, 16704799, 15543343, 12965263, 13134193, 15297563, 13536821, 13918787, 12676193, 15142703, 14437559, 12631777, 13704083, 15632377, 14880601, 15491477, 16625353, 16231427, 13351381, 15831551, 15576611, 15039943, 16321373, 16651757, 16722103, 12801337, 13858841, 13097699, 13844113, 14952997, 14788847, 15081413, 15146069, 16552919, 15883789, 13399327, 13466251, 16003619, 14104499, 15536119, 16667741, 12891587, 13922939, 13375783, 14587351, 15993793, 13077359, 13881059, 14541113, 16076227, 15457859, 13534247, 16327063, 16321843, 15841601, 12901619, 12990127, 14647547, 13025123, 13413511, 15537433, 13879141, 15439847, 13149043}
	Q = Q[:2]
	P = P[:4]

	// Q = []uint64{15297563, 13134193}
	// P = []uint64{15297563, 14232433}
	ringQ, _ := ring.NewRing(1<<5, Q)
	ringP, _ := ring.NewRing(1<<5, P)

	be := NewBasisExtender(ringQ, ringP, []Key{Key{1, 3}}, []Key{Key{3, 1}})
	p1 := ringQ.NewPoly()
	p2 := ringP.NewPoly()
	res := ringQ.NewPoly()

	fmt.Println(Q)
	fmt.Println(P)
	fmt.Println(be.dicConstantsQtoP[Key{1, 3}].qoverqiinvqi)
	fmt.Println(be.dicConstantsQtoP[Key{1, 3}].alphaimodp)
	fmt.Println(be.dicConstantsQtoP[Key{1, 3}].betaioverqi)
	fmt.Println(be.dicConstantsPtoQ[Key{3, 1}].qoverqiinvqi)
	fmt.Println(be.dicConstantsPtoQ[Key{3, 1}].alphaimodp)
	fmt.Println(be.dicConstantsPtoQ[Key{3, 1}].betaioverqi)

	for i := range p1.Coeffs {
		for j := range p1.Coeffs[i] {
			if i == 0 {
				p1.Coeffs[i][j] = 170733929195035 % Q[i]
			} else {
				p1.Coeffs[i][j] = 268626954341170 % Q[i]
			}

			// if i == 0 {
			// 	p1.Coeffs[i][j] = sampling.RandUint64() % Q[i]
			// } else {
			// 	p1.Coeffs[i][j] = sampling.RandUint64() % Q[i]
			// }

		}
	}

	be.ModSwitchQtoP(1, 3, p1, p2)
	fmt.Println()
	be.ModSwitchPtoQ(3, 1, p2, res)

	fmt.Println(p2)
	fmt.Println(p1)
	fmt.Println(res)
	fmt.Println("error check")

	for i := range p1.Coeffs {
		for j := range p1.Coeffs[i] {
			if p1.Coeffs[i][j] != res.Coeffs[i][j] {
				fmt.Println(i, ", ", j, " : ", p1.Coeffs[i][j], ", ", res.Coeffs[i][j])
			}
		}
	}

}

func Test_Basis3(t *testing.T) {
	//ckks parameter init

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            16,
		LogQ:            []int{48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48},
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
		value[i] = sampling.RandFloat64(-1, 1)
	}

	//24bit
	//21552221
	P := []uint64{14624959, 15092711, 16654291, 16108999, 13227197, 16099607, 14232433, 16704799, 15543343, 12965263, 13134193, 15297563, 13536821, 13918787, 12676193, 15142703, 14437559, 12631777, 13704083, 15632377, 14880601, 15491477, 16625353, 16231427, 13351381, 15831551, 15576611, 15039943, 16321373, 16651757, 16722103, 12801337, 13858841, 13097699, 13844113, 14952997, 14788847, 15081413, 15146069, 16552919, 15883789, 13399327, 13466251, 16003619, 14104499, 15536119, 16667741, 12891587, 13922939, 13375783, 14587351, 15993793, 13077359, 13881059, 14541113, 16076227, 15457859, 13534247, 16327063, 16321843, 15841601, 12901619, 12990127, 14647547, 13025123, 13413511, 15537433, 13879141, 15439847, 13149043}
	P = P[:5]

	fmt.Println(params.Q()[:2])
	fmt.Println(P)

	pt := hefloat.NewPlaintext(params, 1)
	pt.IsBatched = false
	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	ringQ, _ := ring.NewRing(params.N(), params.Q()[:3])
	ringP, _ := ring.NewRing(params.N(), P)
	be := NewBasisExtender(ringQ.AtLevel(1), ringP, []Key{{1, 4}}, []Key{{4, 1}})

	fmt.Println(ringQ.AtLevel(1).ModuliChain())
	fmt.Println(P)
	fmt.Println(be.dicConstantsQtoP[Key{1, 4}].qoverqiinvqi)
	fmt.Println(be.dicConstantsQtoP[Key{1, 4}].alphaimodp)
	fmt.Println(be.dicConstantsQtoP[Key{1, 4}].betaioverqi)
	fmt.Println(be.dicConstantsPtoQ[Key{4, 1}].qoverqiinvqi)
	fmt.Println(be.dicConstantsPtoQ[Key{4, 1}].alphaimodp)
	fmt.Println(be.dicConstantsPtoQ[Key{4, 1}].betaioverqi)

	ringQ.AtLevel(1).INTT(ct.Value[0], ct.Value[0])
	ringQ.AtLevel(1).INTT(ct.Value[1], ct.Value[1])

	for i := range ct.Value {
		fmt.Println("TT", ct.Value[i].Coeffs[0])
		fmt.Println("TT", ct.Value[i].Coeffs[1])

		p2 := ringP.NewPoly()
		starttime := time.Now()
		be.ModSwitchQtoP(1, 4, ct.Value[i], p2)
		// fmt.Println("TT", p2.Coeffs)
		be.ModSwitchPtoQ(4, 1, p2, ct.Value[i])
		elapse := time.Since(starttime)
		fmt.Println(elapse)

		fmt.Println("TT", ct.Value[i].Coeffs[0])
		fmt.Println("TT", ct.Value[i].Coeffs[1])
	}

	ringQ.AtLevel(1).NTT(ct.Value[0], ct.Value[0])
	ringQ.AtLevel(1).NTT(ct.Value[1], ct.Value[1])

	values := make([]float64, n)
	dept := decryptor.DecryptNew(ct)
	encoder.Decode(dept, values)
	fmt.Println(ct.Level())
	fmt.Println(ct.LogScale())
	fmt.Println(value)
	fmt.Println(values)

}

func Test_Basis4(t *testing.T) {
	//ckks parameter init

	//ckks parameter init
	SchemeParams := hefloat.ParametersLiteral{
		// logN = 13, full slots
		// # special modulus = 1
		// # available levels = 4
		LogN:            16,
		LogQ:            []int{48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48},
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
	P := []uint64{14624959, 15092711, 16654291, 21552221, 16108999, 13227197, 16099607, 14232433, 16704799, 15543343, 12965263, 13134193, 15297563, 13536821, 13918787, 12676193, 15142703, 14437559, 12631777, 13704083, 15632377, 14880601, 15491477, 16625353, 16231427, 13351381, 15831551, 15576611, 15039943, 16321373, 16651757, 16722103, 12801337, 13858841, 13097699, 13844113, 14952997, 14788847, 15081413, 15146069, 16552919, 15883789, 13399327, 13466251, 16003619, 14104499, 15536119, 16667741, 12891587, 13922939, 13375783, 14587351, 15993793, 13077359, 13881059, 14541113, 16076227, 15457859, 13534247, 16327063, 16321843, 15841601, 12901619, 12990127, 14647547, 13025123, 13413511, 15537433, 13879141, 15439847, 13149043}
	P = P[:4]

	pt := hefloat.NewPlaintext(params, 1)
	pt.IsBatched = false
	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	ringQ := params.RingQ().AtLevel(ct.Level())
	ringP, _ := ring.NewRing(params.N(), P)
	be := ring.NewBasisExtender(ringQ, ringP)

	ringQ.INTT(ct.Value[0], ct.Value[0])
	ringQ.INTT(ct.Value[1], ct.Value[1])

	for i := range ct.Value {
		fmt.Println("TT", ct.Value[i].Coeffs[0])
		fmt.Println("TT", ct.Value[i].Coeffs[1])

		p2 := ringP.NewPoly()
		starttime := time.Now()
		be.ModUpQtoP(1, 3, ct.Value[i], p2)
		// fmt.Println("TT", p2.Coeffs)
		be.ModUpPtoQ(3, 1, p2, ct.Value[i])
		elapse := time.Since(starttime)
		fmt.Println(elapse)

		fmt.Println("TT", ct.Value[i].Coeffs[0])
		fmt.Println("TT", ct.Value[i].Coeffs[1])
	}

	ringQ.NTT(ct.Value[0], ct.Value[0])
	ringQ.NTT(ct.Value[1], ct.Value[1])
	values := make([]float64, n)
	dept := decryptor.DecryptNew(ct)
	encoder.Decode(dept, values)
	fmt.Println(ct.Level())
	fmt.Println(ct.LogScale())
	fmt.Println(values)
}
