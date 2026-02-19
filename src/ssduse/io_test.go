package ssduse

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"

	"github.com/lifejade/mm/src/matmult"
	"github.com/lifejade/mm/src/util"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
)

func Test_SaveLoadCipher(t *testing.T) {
	logN := 10

	// hefloat(CKKS-like) parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            logN,
		LogQ:            []int{48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}

	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		t.Fatalf("NewParametersFromLiteral: %v", err)
	}

	// generate keys
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	N := 1 << params.LogN()

	galLen := 1
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, galLen)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk := make([]*rlwe.GaloisKey, len(galEls))
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
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	_ = hefloat.NewEvaluator(params, evk)
	_, _, _ = encryptor, decryptor, encoder

	size := N
	// data
	values := make([][]float64, size)
	for i := range values {
		values[i] = make([]float64, N)
		for j := range values[i] {
			values[i][j] = sampling.RandFloat64(-1, 1)
		}
	}

	level := params.MaxLevel()
	name := "test4"
	AddWriters(name, 1<<20, N)
	for i := range values {
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		pt.IsBatched = false
		encoder.Encode(values[i], pt)
		ct, _ := encryptor.EncryptNew(pt)
		AppendPoly(name, ct.Value[0], level)
		AppendPoly(name, ct.Value[1], level)
	}
	FlushPoly(name)
	runtime.GC()
	util.PrintMemUsage()

	maxerr := 0.0
	v := make([]float64, N)
	cttmep := hefloat.NewCiphertext(params, 1, params.MaxLevel())
	for i := range values {
		GetPoly(name, 2*i, cttmep.Value[0], level)
		GetPoly(name, 2*i+1, cttmep.Value[1], level)
		ptres := decryptor.DecryptNew(cttmep)
		ptres.IsBatched = false
		encoder.Decode(ptres, v)
		for j := range values[i] {
			e := math.Abs(v[j] - values[i][j])
			if e > maxerr {
				maxerr = e
			}
		}
	}

	util.PrintMemUsage()
	fmt.Println((maxerr))
	fmt.Println(-math.Log2(maxerr))
}

func Test_Switch(t *testing.T) {
	logN := 10

	// hefloat(CKKS-like) parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            logN,
		LogQ:            []int{48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}

	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		t.Fatalf("NewParametersFromLiteral: %v", err)
	}

	// generate keys
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	N := 1 << params.LogN()

	galLen := 1
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, galLen)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk := make([]*rlwe.GaloisKey, len(galEls))
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
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	_ = hefloat.NewEvaluator(params, evk)
	_, _, _ = encryptor, decryptor, encoder

	P := []uint32{3422539, 3370361, 3231143, 3545881, 3577031, 3832931, 4064197, 3617099, 3651497, 3711319, 3439693, 3502001, 3555509, 3552013, 4031179, 4115407, 3167453, 3365393, 3291143, 3204973, 4182419, 3495781, 3315883, 3403391, 3529153, 3390899, 3453773, 3705469, 3180337, 4091993, 3503221, 3598949, 3822277, 3277853, 3547249, 3278053, 3696257, 3849409, 3725257, 3239449, 3730721, 3393619, 3361363, 3732997, 3661573, 3158971, 3516031, 3737039, 3882649, 3614969, 3518491, 3169759, 3326417, 4165333, 3853097, 3845357, 3721603, 3494831, 3255467, 3442987, 3381641, 4188433, 3960053, 3825473, 3269713, 3373781, 3403843, 4177609, 3265337, 3382231, 3342137, 3330179, 3272629, 3725357, 3667453, 3960049, 3435323, 3664249, 3632423, 3515269, 3784733, 3377657, 4064143, 3702119, 3835367, 3564937, 3507397, 3345877, 4169129, 3206783, 3397769, 4145293, 3773477, 3229319, 3161617, 3517427, 3456743, 3687163, 3389423, 3553541}
	PLevel := 33
	P = P[:PLevel+1]
	fmt.Println(P, PLevel)
	ringQ := params.RingQ()
	ringP, _ := matmult.NewRing(N, P)
	be := matmult.NewBasisExtender(ringQ, ringP, []matmult.Key{{From: params.MaxLevel(), To: PLevel}}, []matmult.Key{{From: PLevel, To: params.MaxLevel()}})

	// data
	values := make([]float64, N)
	for j := range values {
		values[j] = sampling.RandFloat64(-1, 1)
	}

	level := params.MaxLevel()
	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false
	encoder.Encode(values, pt)
	ct, _ := encryptor.EncryptNew(pt)
	ringQ.INTT(ct.Value[0], ct.Value[0])
	ringQ.INTT(ct.Value[1], ct.Value[1])

	temppoly := ringP.NewPoly()
	temppoly2 := ringP.NewPoly()
	be.ModSwitchQtoP(params.MaxLevel(), PLevel, ct.Value[0], temppoly)
	be.ModSwitchPtoQ(PLevel, level, temppoly, ct.Value[0])
	be.ModSwitchQtoP(params.MaxLevel(), PLevel, ct.Value[1], temppoly2)
	be.ModSwitchPtoQ(PLevel, level, temppoly2, ct.Value[1])

	ringQ.NTT(ct.Value[0], ct.Value[0])
	ringQ.NTT(ct.Value[1], ct.Value[1])

	maxerr := 0.0
	v := make([]float64, N)
	ptres := decryptor.DecryptNew(ct)
	ptres.IsBatched = false
	encoder.Decode(ptres, v)

	fmt.Println(v)

	for j := range values {
		e := math.Abs(v[j] - values[j])
		if e > maxerr {
			maxerr = e
		}
	}
	util.PrintMemUsage()
	fmt.Println((maxerr))
	fmt.Println(-math.Log2(maxerr))
}

func Test_SLCipherSwitch(t *testing.T) {
	logN := 10

	// hefloat(CKKS-like) parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            logN,
		LogQ:            []int{48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}

	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		t.Fatalf("NewParametersFromLiteral: %v", err)
	}

	// generate keys
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	N := 1 << params.LogN()

	galLen := 1
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, galLen)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk := make([]*rlwe.GaloisKey, len(galEls))
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
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	_ = hefloat.NewEvaluator(params, evk)
	_, _, _ = encryptor, decryptor, encoder

	P := []uint32{3422539, 3370361, 3231143, 3545881, 3577031, 3832931, 4064197, 3617099, 3651497, 3711319, 3439693, 3502001, 3555509, 3552013, 4031179, 4115407, 3167453, 3365393, 3291143, 3204973, 4182419, 3495781, 3315883, 3403391, 3529153, 3390899, 3453773, 3705469, 3180337, 4091993, 3503221, 3598949, 3822277, 3277853, 3547249, 3278053, 3696257, 3849409, 3725257, 3239449, 3730721, 3393619, 3361363, 3732997, 3661573, 3158971, 3516031, 3737039, 3882649, 3614969, 3518491, 3169759, 3326417, 4165333, 3853097, 3845357, 3721603, 3494831, 3255467, 3442987, 3381641, 4188433, 3960053, 3825473, 3269713, 3373781, 3403843, 4177609, 3265337, 3382231, 3342137, 3330179, 3272629, 3725357, 3667453, 3960049, 3435323, 3664249, 3632423, 3515269, 3784733, 3377657, 4064143, 3702119, 3835367, 3564937, 3507397, 3345877, 4169129, 3206783, 3397769, 4145293, 3773477, 3229319, 3161617, 3517427, 3456743, 3687163, 3389423, 3553541}
	PLevel := 33
	P = P[:PLevel+1]
	fmt.Println(P, PLevel)
	ringQ := params.RingQ()
	ringP, _ := matmult.NewRing(N, P)
	be := matmult.NewBasisExtender(ringQ, ringP, []matmult.Key{{From: params.MaxLevel(), To: PLevel}}, []matmult.Key{{From: PLevel, To: params.MaxLevel()}})
	temppoly := ringP.NewPoly()

	size := N
	// data
	values := make([][]float64, size)
	for i := range values {
		values[i] = make([]float64, N)
		for j := range values[i] {
			values[i][j] = sampling.RandFloat64(-1, 1)
		}
	}

	level := params.MaxLevel()
	name := "test4"
	AddWriters(name, 1<<20, N)
	for i := range values {
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		pt.IsBatched = false
		encoder.Encode(values[i], pt)
		ct, _ := encryptor.EncryptNew(pt)

		ringQ.INTT(ct.Value[0], ct.Value[0])
		be.ModSwitchQtoP(params.MaxLevel(), PLevel, ct.Value[0], temppoly)
		AppendPoly32(name, temppoly, PLevel)

		ringQ.INTT(ct.Value[1], ct.Value[1])
		be.ModSwitchQtoP(params.MaxLevel(), PLevel, ct.Value[1], temppoly)
		AppendPoly32(name, temppoly, PLevel)

		// AppendPoly(name, ct.Value[0], level)
		// AppendPoly(name, ct.Value[1], level)
	}
	FlushPoly(name)
	runtime.GC()
	util.PrintMemUsage()

	maxerr := 0.0
	v := make([]float64, N)
	cttmep := hefloat.NewCiphertext(params, 1, params.MaxLevel())
	for i := range values {
		GetPoly32(name, 2*i, temppoly, PLevel)
		be.ModSwitchPtoQ(PLevel, level, temppoly, cttmep.Value[0])
		ringQ.NTT(cttmep.Value[0], cttmep.Value[0])
		GetPoly32(name, 2*i+1, temppoly, PLevel)
		be.ModSwitchPtoQ(PLevel, level, temppoly, cttmep.Value[1])
		ringQ.NTT(cttmep.Value[1], cttmep.Value[1])

		ptres := decryptor.DecryptNew(cttmep)
		ptres.IsBatched = false
		encoder.Decode(ptres, v)
		for j := range values[i] {
			e := math.Abs(v[j] - values[i][j])
			if e > maxerr {
				maxerr = e
			}
		}
	}

	util.PrintMemUsage()
	fmt.Println((maxerr))
	fmt.Println(-math.Log2(maxerr))
}

func Test_SLCipherSwitchPPMM(t *testing.T) {
	logN := 10

	// hefloat(CKKS-like) parameter init
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            logN,
		LogQ:            []int{48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}

	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		t.Fatalf("NewParametersFromLiteral: %v", err)
	}

	// generate keys
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	N := 1 << params.LogN()

	galLen := 1
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// generate keys - Rotating key
	galEls := make([]uint64, galLen)
	for i := range galEls {
		galEls[i] = uint64(2*i + 1)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk := make([]*rlwe.GaloisKey, len(galEls))
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
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	_ = hefloat.NewEvaluator(params, evk)
	_, _, _ = encryptor, decryptor, encoder

	P := []uint32{3422539, 3370361, 3231143, 3545881, 3577031, 3832931, 4064197, 3617099, 3651497, 3711319, 3439693, 3502001, 3555509, 3552013, 4031179, 4115407, 3167453, 3365393, 3291143, 3204973, 4182419, 3495781, 3315883, 3403391, 3529153, 3390899, 3453773, 3705469, 3180337, 4091993, 3503221, 3598949, 3822277, 3277853, 3547249, 3278053, 3696257, 3849409, 3725257, 3239449, 3730721, 3393619, 3361363, 3732997, 3661573, 3158971, 3516031, 3737039, 3882649, 3614969, 3518491, 3169759, 3326417, 4165333, 3853097, 3845357, 3721603, 3494831, 3255467, 3442987, 3381641, 4188433, 3960053, 3825473, 3269713, 3373781, 3403843, 4177609, 3265337, 3382231, 3342137, 3330179, 3272629, 3725357, 3667453, 3960049, 3435323, 3664249, 3632423, 3515269, 3784733, 3377657, 4064143, 3702119, 3835367, 3564937, 3507397, 3345877, 4169129, 3206783, 3397769, 4145293, 3773477, 3229319, 3161617, 3517427, 3456743, 3687163, 3389423, 3553541}
	PLevel := 33
	P = P[:PLevel+1]
	fmt.Println(P, PLevel)
	ringQ := params.RingQ()
	ringP, _ := matmult.NewRing(N, P)
	be := matmult.NewBasisExtender(ringQ, ringP, []matmult.Key{{From: params.MaxLevel(), To: PLevel}}, []matmult.Key{{From: PLevel, To: params.MaxLevel()}})
	temppoly := ringP.NewPoly()

	size := N
	// data
	values := make([][]float64, size)
	for i := range values {
		values[i] = make([]float64, N)
		for j := range values[i] {
			values[i][j] = 0.001
		}
	}

	level := params.MaxLevel()
	name := "test4"
	AddWriters(name, 1<<20, N)
	for i := range values {
		pt := hefloat.NewPlaintext(params, params.MaxLevel())
		pt.IsBatched = false
		encoder.Encode(values[i], pt)
		ct, _ := encryptor.EncryptNew(pt)

		ringQ.INTT(ct.Value[0], ct.Value[0])
		be.ModSwitchQtoP(params.MaxLevel(), PLevel, ct.Value[0], temppoly)
		AppendPoly32(name, temppoly, PLevel)

		ringQ.INTT(ct.Value[1], ct.Value[1])
		be.ModSwitchQtoP(params.MaxLevel(), PLevel, ct.Value[1], temppoly)
		AppendPoly32(name, temppoly, PLevel)

		// AppendPoly(name, ct.Value[0], level)
		// AppendPoly(name, ct.Value[1], level)
	}
	FlushPoly(name)

	runtime.GC()
	util.PrintMemUsage()

	v := make([]float64, N)
	cttmep := hefloat.NewCiphertext(params, 1, params.MaxLevel())
	for i := range values {
		GetPoly32(name, 2*i, temppoly, PLevel)
		be.ModSwitchPtoQ(PLevel, level, temppoly, cttmep.Value[0])
		ringQ.NTT(cttmep.Value[0], cttmep.Value[0])
		GetPoly32(name, 2*i+1, temppoly, PLevel)
		be.ModSwitchPtoQ(PLevel, level, temppoly, cttmep.Value[1])
		ringQ.NTT(cttmep.Value[1], cttmep.Value[1])

		ptres := decryptor.DecryptNew(cttmep)
		ptres.IsBatched = false
		encoder.Decode(ptres, v)
		fmt.Println(v)
	}

}
