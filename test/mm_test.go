package test

// import (
// 	"fmt"
// 	"runtime"
// 	"sync"
// 	"testing"
// 	"time"

// 	"github.com/lifejade/mm/src/matmult"
// 	cwrappingflint "github.com/lifejade/mm/src/matmult/cwrapping_flint"
// 	"github.com/lifejade/mm/src/transpose"
// 	"github.com/lifejade/mm/src/util"
// 	"github.com/tuneinsight/lattigo/v5/core/rlwe"
// 	"github.com/tuneinsight/lattigo/v5/he/hefloat"
// 	"github.com/tuneinsight/lattigo/v5/ring"
// )

// func Test_ppmm_time(t *testing.T) {
// 	//CPU full power
// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

// 	//ckks parameter init
// 	SchemeParams := hefloat.ParametersLiteral{
// 		// logN = 13, full slots
// 		// # special modulus = 1
// 		// # available levels = 4
// 		LogN:            12,
// 		LogQ:            []int{50, 36, 36, 36},
// 		LogP:            []int{50},
// 		Xs:              ring.Ternary{H: 256},
// 		LogDefaultScale: 28,
// 	}

// 	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
// 	if err != nil {
// 		panic(err)
// 	}
// 	fmt.Println("ckks parameter init end")

// 	// generate keys
// 	//fmt.Println("generate keys")
// 	//keytime := time.Now()
// 	kgen := rlwe.NewKeyGenerator(params)
// 	sk := kgen.GenSecretKeyNew()

// 	var pk *rlwe.PublicKey
// 	var rlk *rlwe.RelinearizationKey
// 	var rtk []*rlwe.GaloisKey

// 	fmt.Println("generated bootstrapper end")
// 	pk = kgen.GenPublicKeyNew(sk)
// 	rlk = kgen.GenRelinearizationKeyNew(sk)

// 	// generate keys - Rotating key
// 	convRot := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
// 	galEls := make([]uint64, len(convRot))
// 	for i, x := range convRot {
// 		galEls[i] = params.GaloisElement(x)
// 	}
// 	galEls = append(galEls, params.GaloisElementForComplexConjugation())

// 	for i := range 16 {
// 		galEls = append(galEls, params.GaloisElement((1<<i)/2))
// 	}

// 	rtk = make([]*rlwe.GaloisKey, len(galEls))
// 	starttime := time.Now()
// 	var wg sync.WaitGroup
// 	wg.Add(len(galEls))
// 	for i := range galEls {
// 		i := i
// 		go func() {
// 			defer wg.Done()
// 			kgen_ := rlwe.NewKeyGenerator(params)
// 			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
// 		}()
// 	}
// 	wg.Wait()
// 	elapse := time.Since(starttime)
// 	fmt.Println(elapse)
// 	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)

// 	//generate -er
// 	encryptor := rlwe.NewEncryptor(params, pk)
// 	decryptor := rlwe.NewDecryptor(params, sk)
// 	encoder := hefloat.NewEncoder(params)
// 	evaluator := hefloat.NewEvaluator(params, evk)
// 	fmt.Println("generate Evaluator end")

// 	_, _, _, _ = encoder, encryptor, decryptor, evaluator

// 	n := 1 << params.LogN()
// 	fmt.Println("N is this  : ", n)
// 	value := make([]float64, n)
// 	for i, _ := range value {
// 		value[i] = 0.1
// 	}

// 	pt := hefloat.NewPlaintext(params, 1)
// 	pt.IsBatched = false

// 	encoder.Encode(value, pt)
// 	cts := make([]*rlwe.Ciphertext, n)
// 	for i := range cts {
// 		cts[i], _ = encryptor.EncryptNew(pt.CopyNew())
// 	}

// 	u := make([][]uint64, n)
// 	for i := range n {
// 		u[i] = make([]uint64, n)
// 		for j := range n {
// 			u[i][j] = uint64(j % 11)
// 		}
// 	}
// 	fmt.Println("ppmm time : ", elapse)

// 	fmt.Println("start flint")
// 	starttime = time.Now()
// 	result2 := matmult.PPMM_Flint(cts, u, params, n)
// 	elapse = time.Since(starttime)
// 	fmt.Println("ppmm time : ", elapse)

// 	values := make([][]float64, n)

// 	for i := range values {
// 		values[i] = make([]float64, n)
// 		dept := decryptor.DecryptNew(result2[i])
// 		encoder.Decode(dept, values[i])
// 	}

// 	fmt.Println(values[0][0])

// 	fmt.Println(values[200][700])

// 	fmt.Println(values[800][3])
// }

// func Test_MM(t *testing.T) {
// 	//CPU full power
// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

// 	n := 1 << 5
// 	Q := uint64(1) << 50

// 	u := make([][]uint64, n)
// 	a := make([][]uint64, n)

// 	for i := range u {
// 		u[i] = make([]uint64, n)
// 		for j := range u[i] {
// 			u[i][j] = (uint64(1) << 40) - 5
// 		}
// 	}

// 	for i := range a {
// 		a[i] = make([]uint64, n)
// 		for j := range a[i] {
// 			a[i][j] = (uint64(1) << 40) - 5
// 		}
// 	}
// 	result := cwrappingflint.Mult_mod_mat(u, a, n, n, n, Q)
// 	for i := range result {
// 		for j := range result[i] {
// 			fmt.Print(result[i][j], " ")
// 		}
// 		fmt.Println()
// 	}
// }

// func Test_PCMM(t *testing.T) {
// 	//CPU full power
// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

// 	//ckks parameter init
// 	SchemeParams := hefloat.ParametersLiteral{
// 		// logN = 13, full slots
// 		// # special modulus = 1
// 		// # available levels = 4
// 		LogN:            10,
// 		LogQ:            []int{32, 32, 40, 40, 30, 30},
// 		LogP:            []int{50},
// 		Xs:              ring.Ternary{H: 256},
// 		LogDefaultScale: 28,
// 	}

// 	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
// 	if err != nil {
// 		panic(err)
// 	}
// 	fmt.Println("ckks parameter init end")

// 	// generate keys
// 	//fmt.Println("generate keys")
// 	//keytime := time.Now()
// 	kgen := rlwe.NewKeyGenerator(params)
// 	sk := kgen.GenSecretKeyNew()

// 	var pk *rlwe.PublicKey
// 	var rlk *rlwe.RelinearizationKey
// 	var rtk []*rlwe.GaloisKey

// 	fmt.Println("generated bootstrapper end")
// 	n := 1 << params.LogN()
// 	pk = kgen.GenPublicKeyNew(sk)
// 	rlk = kgen.GenRelinearizationKeyNew(sk)

// 	// generate keys - Rotating key
// 	galEls := make([]uint64, n)
// 	for i := range galEls {
// 		galEls[i] = uint64(2*i + 1)
// 	}
// 	galEls = append(galEls, params.GaloisElementForComplexConjugation())

// 	rtk = make([]*rlwe.GaloisKey, len(galEls))
// 	starttime := time.Now()
// 	var wg sync.WaitGroup
// 	wg.Add(len(galEls))
// 	for i := range galEls {
// 		i := i
// 		go func() {
// 			defer wg.Done()
// 			kgen_ := rlwe.NewKeyGenerator(params)
// 			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
// 		}()
// 	}
// 	wg.Wait()
// 	elapse := time.Since(starttime)
// 	fmt.Println(elapse)
// 	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)

// 	//generate -er
// 	encryptor := rlwe.NewEncryptor(params, pk)
// 	decryptor := rlwe.NewDecryptor(params, sk)
// 	encoder := hefloat.NewEncoder(params)
// 	evaluator := hefloat.NewEvaluator(params, evk)
// 	fmt.Println("generate Evaluator end")

// 	_, _, _, _ = encoder, encryptor, decryptor, evaluator

// 	fmt.Println("N is this  : ", n)
// 	value := make([]float64, n)
// 	for i, _ := range value {
// 		value[i] = 0.0001 * float64(i)
// 	}

// 	pt := hefloat.NewPlaintext(params, 1)
// 	pt.IsBatched = false

// 	encoder.Encode(value, pt)
// 	cts := make([]*rlwe.Ciphertext, n)
// 	for i := range cts {
// 		cts[i], _ = encryptor.EncryptNew(pt.CopyNew())
// 	}
// 	ctT := transpose.Transpose(cts, params, evaluator, encoder, n)
// 	_ = ctT

// 	k := float64(1 << 28)
// 	u := make([][][]uint64, len(params.Q()))
// 	for q := range u {
// 		u[q] = make([][]uint64, n)
// 		fmt.Println("q : ", params.Q()[q])
// 		for i := range n {
// 			u[q][i] = make([]uint64, n)
// 			for j := range n {
// 				u[q][i][j] = params.Q()[q] - uint64(0.5*k)
// 				//u[q][i][j] = uint64(1 * k)
// 				fmt.Print(u[q][i][j], " ")
// 			}
// 			fmt.Println()
// 		}

// 	}

// 	fmt.Println("ppmm time : ", elapse)

// 	fmt.Println("start flint")
// 	starttime = time.Now()
// 	util.DebugCTS(cts, params, encoder, decryptor)
// 	result2 := matmult.PPMM_Flint_CRT(cts, u, params, n)
// 	util.DebugCTS(result2, params, encoder, decryptor)
// 	elapse = time.Since(starttime)
// 	fmt.Println("ppmm time : ", elapse)

// 	for i := range result2 {
// 		evaluator.Mul(result2[i], 1/k, result2[i])
// 		evaluator.Rescale(result2[i], result2[i])
// 	}

// 	values := make([][]float64, n)

// 	for i := range values {
// 		values[i] = make([]float64, n)
// 		dept := decryptor.DecryptNew(result2[i])
// 		encoder.Decode(dept, values[i])

// 		fmt.Println(values[i])
// 	}
// 	fmt.Println(result2[0].LogScale())

// 	fmt.Println(MaxUint64Slice(result2[0].Value[1].Coeffs[0]))
// }

// func Test_MMTime(t *testing.T) {
// 	runtime.GOMAXPROCS(1) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정

// 	// N := 1 << 16
// 	// sparseN := 1 << 7
// 	arrN := []int{1 << 16}
// 	arrSparseN := []int{32, 64, 128, 256}
// 	for j := range arrN {
// 		N := arrN[j]
// 		fmt.Println("N :", N)
// 		for i := range arrSparseN {
// 			sparseN := arrSparseN[i]
// 			fmt.Println("N :", N, " sparseN :", sparseN)

// 			u := make([][]uint64, sparseN)
// 			a := make([][]uint64, sparseN)

// 			for i := range u {
// 				u[i] = make([]uint64, sparseN)
// 				for j := range u[i] {
// 					u[i][j] = 1
// 				}
// 			}
// 			for i := range a {
// 				a[i] = make([]uint64, N)
// 				for j := range a[i] {
// 					a[i][j] = 1
// 				}
// 			}

// 			fmt.Println("Time check")
// 			starttime := time.Now()
// 			result := cwrappingflint.Mult_mod_mat_Blas(u, a, sparseN, sparseN, N, 11)
// 			_ = result
// 			elapse := time.Since(starttime)
// 			fmt.Println("MM time : ", elapse)
// 			fmt.Println()
// 		}
// 		fmt.Print("////////////////////////////////////////////////////////////////////////////////////")
// 	}
// }

// func Test_PCMMTime(t *testing.T) {
// 	runtime.GOMAXPROCS(1) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	SchemeParams := hefloat.ParametersLiteral{
// 		// logN = 13, full slots
// 		// # special modulus = 1
// 		// # available levels = 4
// 		LogN:            5,
// 		LogQ:            []int{60, 40, 40, 40, 40, 40},
// 		LogP:            []int{50},
// 		Xs:              ring.Ternary{H: 256},
// 		LogDefaultScale: 40,
// 	}
// 	//not use
// 	params, _ := hefloat.NewParametersFromLiteral(SchemeParams)

// 	P := []uint64{4107427, 3868699, 4073143, 3639397, 3835109, 3377447, 3338903, 3314141, 3816173, 3731251, 3925091, 3500261, 3507403, 3368353, 3598601, 3637573, 3387523, 3489259, 3804751, 4002811, 3417251, 3245357, 3659177, 4047647, 3367981, 3984439, 3621473, 3565147, 3789193, 3174547, 3293959, 3567803, 3856499, 3299617, 3939619, 4004683, 3803347, 3501467, 3518719, 3631919}
// 	N := 1 << 16
// 	arrLenP := []int{1, 36}
// 	arrSparseN := []int{32, 64, 128, 256}
// 	for l := range arrLenP {
// 		lenP := arrLenP[l]
// 		fmt.Println("lenP : ", lenP)
// 		ringQ, _ := ring.NewRing(1<<16, P[:lenP])
// 		for sn := range arrSparseN {
// 			sparseN := arrSparseN[sn]

// 			fmt.Println("N :", N, " sparseN :", sparseN)

// 			cts := make([]ring.Poly, sparseN)
// 			for i := range cts {
// 				cts[i] = ringQ.NewPoly()
// 				for q := range cts[i].Coeffs {
// 					for n_ := range cts[i].Coeffs[q] {
// 						cts[i].Coeffs[q][n_] = uint64(1)
// 					}
// 				}
// 			}
// 			u := make([][][]uint64, len(P[:lenP]))
// 			for q := range u {
// 				u[q] = make([][]uint64, sparseN)
// 				for i := range sparseN {
// 					u[q][i] = make([]uint64, sparseN)
// 					for j := range sparseN {
// 						u[q][i][j] = uint64(1)
// 					}
// 				}
// 			}
// 			res := make([]ring.Poly, sparseN)
// 			for i := range res {
// 				res[i] = ringQ.NewPoly()
// 			}

// 			starttime := time.Now()
// 			matmult.PPMM_Blas_CRT(cts, u, params, sparseN, sparseN, N, lenP, ringQ, res)
// 			elapse := time.Since(starttime)
// 			fmt.Println("PCMM time : ", elapse)
// 			fmt.Println()
// 		}
// 	}
// }

// func Test_PCMM_Inplace(t *testing.T) {

// 	//CPU full power
// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
// 	SchemeParams := hefloat.ParametersLiteral{
// 		LogN:            5,
// 		LogQ:            []int{48, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56},
// 		LogP:            []int{52, 52},
// 		LogDefaultScale: 40,
// 	}
// 	//parameter init
// 	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
// 	if err != nil {
// 		panic(err)
// 	}

// 	fmt.Println("ckks parameter init end")

// 	// generate keys
// 	//fmt.Println("generate keys")
// 	//keytime := time.Now()
// 	kgen := rlwe.NewKeyGenerator(params)
// 	sk := kgen.GenSecretKeyNew()

// 	N := 1 << params.LogN()

// 	var pk *rlwe.PublicKey
// 	var rlk *rlwe.RelinearizationKey
// 	var rtk []*rlwe.GaloisKey

// 	fmt.Println("generated bootstrapper end")
// 	pk = kgen.GenPublicKeyNew(sk)
// 	rlk = kgen.GenRelinearizationKeyNew(sk)

// 	// generate keys - Rotating key
// 	galEls := make([]uint64, N)
// 	for i := range galEls {
// 		galEls[i] = uint64(2*i + 1)
// 	}
// 	galEls = append(galEls, params.GaloisElementForComplexConjugation())

// 	rtk = make([]*rlwe.GaloisKey, len(galEls))
// 	starttime := time.Now()
// 	var wg sync.WaitGroup
// 	wg.Add(len(galEls))
// 	for i := range galEls {
// 		go func() {
// 			defer wg.Done()
// 			kgen_ := rlwe.NewKeyGenerator(params)
// 			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
// 		}()
// 	}
// 	wg.Wait()
// 	elapse := time.Since(starttime)
// 	fmt.Println(elapse)
// 	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
// 	//generate -er
// 	encryptor := rlwe.NewEncryptor(params, pk)
// 	decryptor := rlwe.NewDecryptor(params, sk)
// 	encoder := hefloat.NewEncoder(params)
// 	evaluator := hefloat.NewEvaluator(params, evk)
// 	fmt.Println("generate Evaluator end")
// 	_ = evaluator
// 	fmt.Println("ckks log degree : ", params.LogN())

// 	ringQ := params.RingQ().AtLevel(params.MaxLevel())

// 	value := make([]float64, N)
// 	for i := range value {
// 		value[i] = 0.01
// 	}

// 	pt := hefloat.NewPlaintext(params, params.MaxLevel())
// 	pt.IsBatched = false
// 	encoder.Encode(value, pt)
// 	cts := make([]*rlwe.Ciphertext, N)
// 	for i := range cts {
// 		cts[i], _ = encryptor.EncryptNew(pt)
// 		ringQ.INTT(cts[i].Value[0], cts[i].Value[0])
// 		ringQ.INTT(cts[i].Value[1], cts[i].Value[1])
// 	}

// 	P := []uint64{4107427, 3868699, 4073143, 3639397, 3835109, 3377447, 3338903, 3314141, 3816173, 3731251, 3925091, 3500261, 3507403, 3368353, 3598601, 3637573, 3387523, 3489259, 3804751, 4002811, 3417251, 3245357, 3659177, 4047647, 3367981, 3984439, 3621473, 3565147, 3789193, 3174547, 3293959, 3567803, 3856499, 3299617, 3939619, 4004683, 3803347, 3501467, 3518719, 3631919}
// 	PLevel := 38
// 	P = P[:PLevel+1]
// 	ringP, _ := ring.NewRing(N, P)
// 	be := matmult.NewBasisExtender(ringQ, ringP, []matmult.Key{{params.MaxLevel(), PLevel}}, []matmult.Key{{PLevel, params.MaxLevel()}})

// 	polys := make([][]ring.Poly, 2)
// 	for i := range polys {
// 		polys[i] = make([]ring.Poly, N)
// 		for j := range polys[i] {
// 			polys[i][j] = ringP.NewPoly()
// 		}
// 	}
// 	for d := range 2 {
// 		for i := range N {
// 			be.ModSwitchQtoP_Old(params.MaxLevel(), PLevel, cts[i].Value[d], polys[d][i])
// 		}
// 	}

// 	scale := 1 << 10
// 	mat := make([]float64, (PLevel+1)*N*N)
// 	for i := range mat {
// 		mat[i] = float64(int(0.1 * float64(scale)))
// 	}

// 	buff1 := make([]float64, (PLevel+1)*N*N)
// 	buff2 := make([]float64, (PLevel+1)*N*N)

// 	matmult.PPMM_Blas_CRT_Inplace(polys, mat, N, N, N, PLevel+1, 2, ringP, buff1, buff2)
// 	matmult.PPMM_Blas_CRT_Inplace(polys, mat, N, N, N, PLevel+1, 2, ringP, buff1, buff2)

// 	// mat := make([][][]uint64, PLevel+1)
// 	// for i := range mat {
// 	// 	mat[i] = make([][]uint64, N)
// 	// 	for j := range mat[i] {
// 	// 		mat[i][j] = make([]uint64, N)
// 	// 		for k := range mat[i][j] {
// 	// 			mat[i][j][k] = 1
// 	// 		}
// 	// 	}
// 	// }
// 	// for i := range 2 {
// 	// 	matmult.PPMM_Blas_CRT(polys[i], mat, params, N, N, N, PLevel+1, ringP, polys[i])
// 	// }

// 	for d := range 2 {
// 		for i := range N {
// 			be.ModSwitchPtoQ_Old(PLevel, params.MaxLevel(), polys[d][i], cts[i].Value[d])

// 		}
// 	}

// 	for i := range cts {
// 		ringQ.NTT(cts[i].Value[0], cts[i].Value[0])
// 		ringQ.NTT(cts[i].Value[1], cts[i].Value[1])
// 		evaluator.Mul(cts[i], 1.0/float64(scale), cts[i])
// 		evaluator.Rescale(cts[i], cts[i])
// 		evaluator.Mul(cts[i], 1.0/float64(scale), cts[i])
// 		evaluator.Rescale(cts[i], cts[i])
// 		ptres := decryptor.DecryptNew(cts[i])
// 		encoder.Decode(ptres, value)
// 		fmt.Println(value[:10])
// 	}
// }

// func Test_PCMMTime_Inplace(t *testing.T) {
// 	runtime.GOMAXPROCS(1) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	SchemeParams := hefloat.ParametersLiteral{
// 		// logN = 13, full slots
// 		// # special modulus = 1
// 		// # available levels = 4
// 		LogN:            5,
// 		LogQ:            []int{60, 40, 40, 40, 40, 40},
// 		LogP:            []int{50},
// 		Xs:              ring.Ternary{H: 256},
// 		LogDefaultScale: 40,
// 	}
// 	//not use
// 	params, _ := hefloat.NewParametersFromLiteral(SchemeParams)

// 	P := []uint64{4107427, 3868699, 4073143, 3639397, 3835109, 3377447, 3338903, 3314141, 3816173, 3731251, 3925091, 3500261, 3507403, 3368353, 3598601, 3637573, 3387523, 3489259, 3804751, 4002811, 3417251, 3245357, 3659177, 4047647, 3367981, 3984439, 3621473, 3565147, 3789193, 3174547, 3293959, 3567803, 3856499, 3299617, 3939619, 4004683, 3803347, 3501467, 3518719, 3631919}
// 	N := 1 << 16
// 	arrLenP := []int{1, 36}
// 	arrSparseN := []int{32, 64, 128, 256}
// 	for l := range arrLenP {
// 		lenP := arrLenP[l]
// 		fmt.Println("lenP : ", lenP)
// 		ringQ, _ := ring.NewRing(1<<16, P[:lenP])
// 		for sn := range arrSparseN {
// 			sparseN := arrSparseN[sn]

// 			fmt.Println("N :", N, " sparseN :", sparseN)

// 			cts := make([]ring.Poly, sparseN)
// 			for i := range cts {
// 				cts[i] = ringQ.NewPoly()
// 				for q := range cts[i].Coeffs {
// 					for n_ := range cts[i].Coeffs[q] {
// 						cts[i].Coeffs[q][n_] = uint64(1)
// 					}
// 				}
// 			}
// 			u := make([][][]uint64, len(P[:lenP]))
// 			for q := range u {
// 				u[q] = make([][]uint64, sparseN)
// 				for i := range sparseN {
// 					u[q][i] = make([]uint64, sparseN)
// 					for j := range sparseN {
// 						u[q][i][j] = uint64(1)
// 					}
// 				}
// 			}
// 			res := make([]ring.Poly, sparseN)
// 			for i := range res {
// 				res[i] = ringQ.NewPoly()
// 			}

// 			starttime := time.Now()
// 			matmult.PPMM_Blas_CRT(cts, u, params, sparseN, sparseN, N, lenP, ringQ, res)
// 			elapse := time.Since(starttime)
// 			fmt.Println("PCMM time : ", elapse)
// 			fmt.Println()
// 		}
// 	}
// }
// func Test_MMTime_CWrapping(t *testing.T) {
// 	runtime.GOMAXPROCS(1) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정

// 	// N := 1 << 16
// 	// sparseN := 1 << 7
// 	arrN := []int{1 << 16}
// 	arrSparseN := []int{32, 64, 128, 256}
// 	for j := range arrN {
// 		N := arrN[j]
// 		fmt.Println("N :", N)
// 		for i := range arrSparseN {
// 			sparseN := arrSparseN[i]
// 			fmt.Println("N :", N, " sparseN :", sparseN)

// 			u := make([]float64, sparseN*sparseN)
// 			a := make([]float64, sparseN*N)
// 			r := make([]uint64, sparseN*N)

// 			for i := range u {
// 				u[i] = 1
// 			}
// 			for i := range a {
// 				a[i] = 1
// 			}

// 			fmt.Println("Time check")
// 			starttime := time.Now()
// 			cwrappingflint.Mult_mod_mat_Blas_ForTest(u, a, r, sparseN, sparseN, N, 11)
// 			elapse := time.Since(starttime)
// 			fmt.Println("MM time : ", elapse)
// 			fmt.Println()
// 		}
// 		fmt.Print("////////////////////////////////////////////////////////////////////////////////////")
// 	}

// }

// func Test_MMTime_CWrapping2(t *testing.T) {
// 	runtime.GOMAXPROCS(1) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정

// 	// N := 1 << 16
// 	// sparseN := 1 << 7
// 	arrN := []int{1 << 16}
// 	arrSparseN := []int{32, 64, 128, 256}
// 	level := 11
// 	for j := range arrN {
// 		N := arrN[j]
// 		fmt.Println("N :", N)

// 		for i := range arrSparseN {
// 			sparseN := arrSparseN[i]
// 			fmt.Println("N :", N, " sparseN :", sparseN)

// 			u := make([]float64, level*sparseN*sparseN)
// 			a := make([]float64, level*sparseN*N)
// 			r := make([]float64, level*sparseN*N)

// 			for i := range u {
// 				u[i] = float64(int(i / (sparseN * sparseN)))
// 			}
// 			for i := range a {
// 				a[i] = 1
// 			}

// 			fmt.Println("Time check")
// 			starttime := time.Now()
// 			cwrappingflint.Mult_mod_mat_Blas_Inplace(u, a, r, sparseN, sparseN, N, level)
// 			elapse := time.Since(starttime)
// 			fmt.Println("MM time : ", elapse)
// 			fmt.Println(r[:100])
// 			fmt.Println(r[sparseN*N : sparseN*N+100])
// 			fmt.Println(r[sparseN*N*2 : sparseN*N*2+100])
// 			fmt.Println()
// 		}
// 		fmt.Print("////////////////////////////////////////////////////////////////////////////////////")
// 	}

// }

// func Test_PPMM_Blas_CRT_Inplace(t *testing.T) {
// 	runtime.GOMAXPROCS(1) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정

// 	P := []uint64{4107427, 3868699, 4073143, 3639397, 3835109, 3377447, 3338903, 3314141, 3816173, 3731251, 3925091, 3500261, 3507403, 3368353, 3598601, 3637573, 3387523, 3489259, 3804751, 4002811, 3417251, 3245357, 3659177, 4047647, 3367981, 3984439, 3621473, 3565147, 3789193, 3174547, 3293959, 3567803, 3856499, 3299617, 3939619, 4004683, 3803347, 3501467, 3518719, 3631919}
// 	N := 1 << 5
// 	arrLenP := []int{36}
// 	arrSparseN := []int{32, 64, 128, 256}
// 	for l := range arrLenP {
// 		lenP := arrLenP[l]
// 		fmt.Println("lenP : ", lenP)
// 		ringQ, _ := ring.NewRing(1<<5, P[:lenP])
// 		for sn := range arrSparseN {
// 			sparseN := arrSparseN[sn]

// 			fmt.Println("N :", N, " sparseN :", sparseN)

// 			cts := make([][]ring.Poly, 2)
// 			for i := range cts {
// 				cts[i] = make([]ring.Poly, sparseN)
// 				for j := range cts[i] {
// 					cts[i][j] = ringQ.NewPoly()
// 					for q := range cts[i][j].Coeffs {
// 						for n_ := range cts[i][j].Coeffs[q] {
// 							cts[i][j].Coeffs[q][n_] = uint64(n_)
// 						}
// 					}
// 				}

// 			}
// 			u := make([]float64, len(P[:lenP])*sparseN*sparseN)
// 			for q := range len(P[:lenP]) {
// 				for i := range sparseN {
// 					for j := range sparseN {
// 						u[q*sparseN*sparseN+i*sparseN+j] = float64(1)
// 					}
// 				}
// 			}
// 			buffer1 := make([]float64, lenP*N*sparseN)
// 			buffer2 := make([]float64, lenP*N*sparseN)

// 			starttime := time.Now()
// 			matmult.PPMM_Blas_CRT_Inplace(cts, u, sparseN, sparseN, N, lenP, 2, ringQ, buffer1, buffer2)
// 			elapse := time.Since(starttime)
// 			fmt.Println("PCMM time : ", elapse)
// 			fmt.Println(cts[0][0].Coeffs[2])
// 			fmt.Println(cts[1][0].Coeffs[2])
// 			fmt.Println()
// 		}
// 	}

// }

// func Test_NegMultConst(t *testing.T) {
// 	//CPU full power
// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

// 	//ckks parameter init
// 	SchemeParams := hefloat.ParametersLiteral{
// 		// logN = 13, full slots
// 		// # special modulus = 1
// 		// # available levels = 4
// 		LogN:            5,
// 		LogQ:            []int{60, 40, 40, 40, 40, 40},
// 		LogP:            []int{50},
// 		Xs:              ring.Ternary{H: 256},
// 		LogDefaultScale: 40,
// 	}

// 	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
// 	if err != nil {
// 		panic(err)
// 	}
// 	fmt.Println("ckks parameter init end")

// 	// generate keys
// 	//fmt.Println("generate keys")
// 	//keytime := time.Now()
// 	kgen := rlwe.NewKeyGenerator(params)
// 	sk := kgen.GenSecretKeyNew()

// 	var pk *rlwe.PublicKey
// 	var rlk *rlwe.RelinearizationKey
// 	var rtk []*rlwe.GaloisKey

// 	fmt.Println("generated bootstrapper end")
// 	n := 1 << params.LogN()
// 	pk = kgen.GenPublicKeyNew(sk)
// 	rlk = kgen.GenRelinearizationKeyNew(sk)

// 	// generate keys - Rotating key
// 	galEls := make([]uint64, n)
// 	for i := range galEls {
// 		galEls[i] = uint64(2*i + 1)
// 	}
// 	galEls = append(galEls, params.GaloisElementForComplexConjugation())

// 	rtk = make([]*rlwe.GaloisKey, len(galEls))
// 	starttime := time.Now()
// 	var wg sync.WaitGroup
// 	wg.Add(len(galEls))
// 	for i := range galEls {
// 		i := i
// 		go func() {
// 			defer wg.Done()
// 			kgen_ := rlwe.NewKeyGenerator(params)
// 			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
// 		}()
// 	}
// 	wg.Wait()
// 	elapse := time.Since(starttime)
// 	fmt.Println(elapse)
// 	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)

// 	//generate -er
// 	encryptor := rlwe.NewEncryptor(params, pk)
// 	decryptor := rlwe.NewDecryptor(params, sk)
// 	encoder := hefloat.NewEncoder(params)
// 	evaluator := hefloat.NewEvaluator(params, evk)
// 	fmt.Println("generate Evaluator end")

// 	_, _, _, _ = encoder, encryptor, decryptor, evaluator

// 	fmt.Println("N is this  : ", n)
// 	value := make([]float64, n)
// 	for i, _ := range value {
// 		value[i] = -0.0016 + 0.0001*float64(i)
// 	}

// 	pt := hefloat.NewPlaintext(params, 1)
// 	pt.IsBatched = false

// 	encoder.Encode(value, pt)
// 	ct, _ := encryptor.EncryptNew(pt)

// 	k := float64(1 << 20)
// 	u := make([]uint64, len(params.Q()))
// 	for q := range u {
// 		u[q] = params.Q()[q] - uint64(0.5*k)
// 	}
// 	ringQ := params.RingQ().AtLevel(ct.Level())
// 	ringQ.INTT(ct.Value[0], ct.Value[0])
// 	ringQ.INTT(ct.Value[1], ct.Value[1])

// 	ringQ.MForm(ct.Value[0], ct.Value[0])
// 	ringQ.MForm(ct.Value[1], ct.Value[1])

// 	ringQ.MulRNSScalarMontgomery(ct.Value[0], u, ct.Value[0])
// 	ringQ.MulRNSScalarMontgomery(ct.Value[1], u, ct.Value[1])

// 	ringQ.NTT(ct.Value[0], ct.Value[0])
// 	ringQ.NTT(ct.Value[1], ct.Value[1])

// 	evaluator.Mul(ct, 1/k, ct)
// 	evaluator.Rescale(ct, ct)

// 	values := make([]float64, n)

// 	dept := decryptor.DecryptNew(ct)
// 	encoder.Decode(dept, values)

// 	fmt.Println(values)
// }

// func Test_NegMultVec(t *testing.T) {
// 	//CPU full power
// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

// 	//ckks parameter init
// 	SchemeParams := hefloat.ParametersLiteral{
// 		// logN = 13, full slots
// 		// # special modulus = 1
// 		// # available levels = 4
// 		LogN:            5,
// 		LogQ:            []int{60, 40, 40, 40, 40, 40},
// 		LogP:            []int{50},
// 		Xs:              ring.Ternary{H: 256},
// 		LogDefaultScale: 40,
// 	}

// 	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
// 	if err != nil {
// 		panic(err)
// 	}
// 	fmt.Println("ckks parameter init end")

// 	// generate keys
// 	//fmt.Println("generate keys")
// 	//keytime := time.Now()
// 	kgen := rlwe.NewKeyGenerator(params)
// 	sk := kgen.GenSecretKeyNew()

// 	var pk *rlwe.PublicKey
// 	var rlk *rlwe.RelinearizationKey
// 	var rtk []*rlwe.GaloisKey

// 	fmt.Println("generated bootstrapper end")
// 	n := 1 << params.LogN()
// 	pk = kgen.GenPublicKeyNew(sk)
// 	rlk = kgen.GenRelinearizationKeyNew(sk)

// 	// generate keys - Rotating key
// 	galEls := make([]uint64, n)
// 	for i := range galEls {
// 		galEls[i] = uint64(2*i + 1)
// 	}
// 	galEls = append(galEls, params.GaloisElementForComplexConjugation())

// 	rtk = make([]*rlwe.GaloisKey, len(galEls))
// 	starttime := time.Now()
// 	var wg sync.WaitGroup
// 	wg.Add(len(galEls))
// 	for i := range galEls {
// 		i := i
// 		go func() {
// 			defer wg.Done()
// 			kgen_ := rlwe.NewKeyGenerator(params)
// 			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
// 		}()
// 	}
// 	wg.Wait()
// 	elapse := time.Since(starttime)
// 	fmt.Println(elapse)
// 	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)

// 	//generate -er
// 	encryptor := rlwe.NewEncryptor(params, pk)
// 	decryptor := rlwe.NewDecryptor(params, sk)
// 	encoder := hefloat.NewEncoder(params)
// 	evaluator := hefloat.NewEvaluator(params, evk)
// 	fmt.Println("generate Evaluator end")

// 	_, _, _, _ = encoder, encryptor, decryptor, evaluator

// 	fmt.Println("N is this  : ", n)
// 	value := make([]float64, n)
// 	for i, _ := range value {
// 		value[i] = 0.00001
// 	}

// 	pt := hefloat.NewPlaintext(params, 1)
// 	pt.IsBatched = false

// 	encoder.Encode(value, pt)
// 	ct, _ := encryptor.EncryptNew(pt)
// 	ringQ := params.RingQ().AtLevel(ct.Level())

// 	op := ringQ.NewPoly()
// 	k := float64(1 << 40)
// 	for q := range op.Coeffs {
// 		for n_ := range op.Coeffs[q] {
// 			op.Coeffs[q][n_] = params.Q()[q] - uint64(k)
// 			//op.Coeffs[q][n_] = uint64(k)
// 		}
// 	}
// 	fmt.Println(len(op.Coeffs))
// 	fmt.Println(len(op.Coeffs[0]))
// 	ringQ.NTT(op, op)

// 	// ringQ.INTT(ct.Value[0], ct.Value[0])
// 	// ringQ.INTT(ct.Value[1], ct.Value[1])

// 	ringQ.MForm(ct.Value[0], ct.Value[0])
// 	ringQ.MForm(ct.Value[1], ct.Value[1])

// 	ringQ.MulCoeffsMontgomery(ct.Value[0], op, ct.Value[0])
// 	ringQ.MulCoeffsMontgomery(ct.Value[1], op, ct.Value[1])

// 	// ringQ.NTT(ct.Value[0], ct.Value[0])
// 	// ringQ.NTT(ct.Value[1], ct.Value[1])

// 	evaluator.Mul(ct, 1/k, ct)
// 	evaluator.Rescale(ct, ct)

// 	values := make([]float64, n)

// 	dept := decryptor.DecryptNew(ct)
// 	encoder.Decode(dept, values)

// 	fmt.Println(values)
// }

// func MaxUint64Slice(arr []uint64) uint64 {
// 	if len(arr) == 0 {
// 		panic("slice is empty")
// 	}
// 	max := arr[0]
// 	for _, v := range arr[1:] {
// 		if v > max {
// 			max = v
// 		}
// 	}
// 	return max
// }
