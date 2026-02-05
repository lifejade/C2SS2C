package test

// import (
// 	"fmt"
// 	"math/cmplx"
// 	"runtime"
// 	"sync"
// 	"testing"
// 	"time"

// 	"github.com/lifejade/mm/src/matmult"
// 	"github.com/lifejade/mm/src/transpose"
// 	"github.com/lifejade/mm/src/util"
// 	"github.com/tuneinsight/lattigo/v5/core/rlwe"
// 	"github.com/tuneinsight/lattigo/v5/he/hefloat"
// 	"github.com/tuneinsight/lattigo/v5/ring"
// 	"github.com/tuneinsight/lattigo/v5/schemes/ckks"
// 	"github.com/tuneinsight/lattigo/v5/utils/sampling"
// )

// func Test_FFT(t *testing.T) {

// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
// 	logn := 5
// 	n := 1 << logn

// 	roots := ckks.GetRootsBigComplex(n<<2, 53)
// 	roots_complex := make([]complex128, 4*n)

// 	for i := range roots_complex {
// 		roots_complex[i] = roots[i].Complex128()
// 	}
// 	fmt.Println()

// 	pow5 := make([]int, (n<<1)+1)
// 	pow5[0] = 1
// 	for i := 1; i < (n<<1)+1; i++ {
// 		pow5[i] = pow5[i-1] * 5
// 		pow5[i] &= (n << 2) - 1
// 	}

// 	SF := make([][]complex128, n)

// 	for i := range SF {
// 		SF[i] = make([]complex128, n)
// 		for j := range SF[i] {
// 			idx := (pow5[i] * j) & ((n << 2) - 1)
// 			SF[i][j] = roots_complex[idx]
// 		}
// 	}

// 	SF_ := make([][][]complex128, logn)
// 	for idx := 0; idx < logn; idx++ {
// 		SF_[idx] = make([][]complex128, n)
// 		for i := range n {
// 			SF_[idx][i] = make([]complex128, n)
// 		}

// 		m := 1 << (idx + 1)
// 		for i := 0; i < n; i += m {
// 			pow5v := 1
// 			for j := 0; j < (m >> 1); j++ {
// 				k := pow5v * n / m

// 				SF_[idx][i+j][i+j] = 1
// 				SF_[idx][i+j][i+j+(m>>1)] = roots_complex[k]

// 				SF_[idx][i+j+(m>>1)][i+j] = 1
// 				SF_[idx][i+j+(m>>1)][i+j+(m>>1)] = -roots_complex[k]

// 				pow5v *= 5
// 				pow5v = pow5v & ((m << 2) - 1)
// 			}
// 		}
// 	}

// 	var SF__ [][]complex128
// 	SF__ = matmult.BitReversePermutationMatrix(n)
// 	// SF__ = make([][]complex128, n)
// 	// for i := range SF__ {
// 	// 	SF__[i] = make([]complex128, n)
// 	// 	SF__[i][i] = 1
// 	// }

// 	for i := range SF_ {
// 		SF__ = mul(SF_[i], SF__)
// 	}

// 	fmt.Println("is same ? : ", closeMat(SF, SF__, 0.00001))

// 	fmt.Println("origin SF")
// 	for i := range SF {
// 		fmt.Println(SF[i])
// 	}

// 	fmt.Println("/////////////////////////////////")
// 	fmt.Println("new SF")
// 	for i := range SF__ {
// 		fmt.Println(SF__[i])
// 	}

// 	fmt.Println("/////////////////////////////////")

// 	for i := range SF_ {
// 		for j := range SF_[i] {
// 			fmt.Println(SF_[i][j])
// 		}
// 		fmt.Println("")
// 	}

// 	// fmt.Println("//////////")
// 	// test := mul(SF_[0], matmult.BitReversePermutationMatrix(n))
// 	// for i := range test {
// 	// 	fmt.Println(test[i])
// 	// }
// }

// func Test_FFT_CL(t *testing.T) {

// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
// 	logn := 10
// 	n := 1 << logn

// 	roots := ckks.GetRootsBigComplex(n<<2, 53)
// 	roots_complex := make([]complex128, 4*n)

// 	for i := range roots_complex {
// 		roots_complex[i] = roots[i].Complex128()
// 	}
// 	fmt.Println()

// 	pow5 := make([]int, (n<<1)+1)
// 	pow5[0] = 1
// 	for i := 1; i < (n<<1)+1; i++ {
// 		pow5[i] = pow5[i-1] * 5
// 		pow5[i] &= (n << 2) - 1
// 	}

// 	SF := make([][]complex128, n)

// 	for i := range SF {
// 		SF[i] = make([]complex128, n)
// 		for j := range SF[i] {
// 			idx := (pow5[i] * j) & ((n << 2) - 1)
// 			SF[i][j] = roots_complex[idx]
// 		}
// 	}

// 	SF_ := make([][][]complex128, logn)
// 	for idx := 0; idx < logn; idx++ {
// 		SF_[idx] = make([][]complex128, n)
// 		for i := range n {
// 			SF_[idx][i] = make([]complex128, n)
// 		}

// 		m := 1 << (idx + 1)
// 		for i := 0; i < n; i += m {
// 			pow5v := 1
// 			for j := 0; j < (m >> 1); j++ {
// 				k := pow5v * n / m

// 				SF_[idx][i+j][i+j] = 1
// 				SF_[idx][i+j][i+j+(m>>1)] = roots_complex[k]

// 				SF_[idx][i+j+(m>>1)][i+j] = 1
// 				SF_[idx][i+j+(m>>1)][i+j+(m>>1)] = -roots_complex[k]

// 				pow5v *= 5
// 				pow5v = pow5v & ((m << 2) - 1)
// 			}
// 		}
// 	}

// 	var SF__ [][]complex128
// 	SF__ = matmult.BitReversePermutationMatrix(n)
// 	// SF__ = make([][]complex128, n)
// 	// for i := range SF__ {
// 	// 	SF__[i] = make([]complex128, n)
// 	// 	SF__[i][i] = 1
// 	// }

// 	for i := range SF_ {
// 		SF__ = mul(SF_[i], SF__)
// 	}

// 	fmt.Println("is same ? : ", closeMat(SF, SF__, 0.00001))

// 	fmt.Println("origin SF")
// 	for i := range SF {
// 		fmt.Println(SF[i])
// 	}

// 	fmt.Println("/////////////////////////////////")
// 	fmt.Println("new SF")
// 	for i := range SF__ {
// 		fmt.Println(SF__[i])
// 	}

// 	fmt.Println("/////////////////////////////////")

// 	l := 2
// 	pl := logn / l
// 	SF_CL := make([][][]complex128, l)
// 	for i := range SF_CL {
// 		SF_CL[i] = SF_[i*pl]
// 		for j := range pl {
// 			if j == 0 {
// 				continue
// 			}
// 			SF_CL[i] = mul(SF_[i*pl+j], SF_CL[i])
// 		}
// 	}

// 	fmt.Println("new SF_CL")
// 	for i := range SF_CL {
// 		for j := range SF_CL[i] {
// 			fmt.Println(SF_CL[i][j])
// 		}
// 		fmt.Println()
// 	}

// 	fmt.Println("check")
// 	SF_CL_ := matmult.BitReversePermutationMatrix(n)
// 	for i := range SF_CL {
// 		SF_CL_ = mul(SF_CL[i], SF_CL_)
// 	}
// 	fmt.Println("is same ? : ", closeMat(SF, SF_CL_, 0.00001))
// 	for i := range SF_CL_ {
// 		fmt.Println(SF_CL_[i])
// 	}
// }

// func Test_IFFT(t *testing.T) {

// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
// 	logn := 2
// 	n := 1 << logn

// 	roots := ckks.GetRootsBigComplex(n<<2, 53)
// 	roots_complex := make([]complex128, 4*n)

// 	for i := range roots_complex {
// 		roots_complex[i] = roots[i].Complex128()
// 	}
// 	fmt.Println()

// 	pow5 := make([]int, (n<<1)+1)
// 	pow5[0] = 1
// 	for i := 1; i < (n<<1)+1; i++ {
// 		pow5[i] = pow5[i-1] * 5
// 		pow5[i] &= (n << 2) - 1
// 	}

// 	SF := make([][]complex128, n)

// 	for i := range SF {
// 		SF[i] = make([]complex128, n)
// 		for j := range SF[i] {
// 			idx := (pow5[i] * j) & ((n << 2) - 1)
// 			SF[i][j] = roots_complex[idx]
// 		}
// 	}

// 	SFI := make([][]complex128, n)
// 	for i := range SFI {
// 		SFI[i] = make([]complex128, n)
// 	}
// 	for i := range SFI {
// 		for j := range SFI[i] {
// 			idx := (pow5[i] * j) & ((n << 2) - 1)
// 			SFI[j][i] = cmplx.Conj(roots_complex[idx]) / complex((float64(n)), 0)
// 		}
// 	}

// 	SFI_ := make([][][]complex128, logn)
// 	for idx := 0; idx < logn; idx++ {
// 		SFI_[idx] = make([][]complex128, n)
// 		for i := range n {
// 			SFI_[idx][i] = make([]complex128, n)
// 		}

// 		m := n >> (idx)
// 		for i := 0; i < n; i += m {
// 			pow5v := 1
// 			for j := 0; j < (m >> 1); j++ {
// 				k := pow5v * n / m

// 				SFI_[idx][i+j][i+j] = 1
// 				SFI_[idx][i+j][i+j+(m>>1)] = 1

// 				SFI_[idx][i+j+(m>>1)][i+j] = cmplx.Conj(roots_complex[k])
// 				SFI_[idx][i+j+(m>>1)][i+j+(m>>1)] = -cmplx.Conj(roots_complex[k])

// 				pow5v *= 5
// 				pow5v = pow5v & ((m << 2) - 1)
// 			}
// 		}
// 	}

// 	var SFI__ [][]complex128
// 	SFI__ = matmult.BitReversePermutationMatrix(n)
// 	for i := range SFI__ {
// 		for j := range SFI__ {
// 			SFI__[i][j] /= complex(float64(n), 0)
// 		}
// 	}
// 	// SF__ = make([][]complex128, n)
// 	// for i := range SF__ {
// 	// 	SF__[i] = make([]complex128, n)
// 	// 	SF__[i][i] = 1
// 	// }

// 	for i := range SFI_ {
// 		SFI__ = mul(SFI__, SFI_[logn-i-1])
// 	}

// 	fmt.Println("is same ? : ", closeMat(SFI, SFI__, 0.0001))

// 	fmt.Println("origin SF")
// 	for i := range SFI {
// 		fmt.Println(SFI[i])
// 	}

// 	fmt.Println("/////////////////////////////////")
// 	fmt.Println("new SF")
// 	for i := range SFI__ {
// 		fmt.Println(SFI__[i])
// 	}

// 	fmt.Println("/////////////////////////////////")

// 	// l := 2
// 	// pl := logn / l
// 	// SF_CL := make([][][]complex128, l)
// 	// for i := range SF_CL {
// 	// 	SF_CL[i] = SF_[i*pl]
// 	// 	for j := range pl {
// 	// 		if j == 0 {
// 	// 			continue
// 	// 		}
// 	// 		SF_CL[i] = mul(SF_[i*pl+j], SF_CL[i])
// 	// 	}
// 	// }

// 	// fmt.Println("new SF_CL")
// 	// for i := range SF_CL {
// 	// 	for j := range SF_CL[i] {
// 	// 		fmt.Println(SF_CL[i][j])
// 	// 	}
// 	// 	fmt.Println()
// 	// }

// 	// fmt.Println("check")
// 	// SF_CL_ := matmult.BitReversePermutationMatrix(n)
// 	// for i := range SF_CL {
// 	// 	SF_CL_ = mul(SF_CL[i], SF_CL_)
// 	// }
// 	// fmt.Println("is same ? : ", closeMat(SF, SF_CL_, 0.00001))
// 	// for i := range SF_CL_ {
// 	// 	fmt.Println(SF_CL_[i])
// 	// }
// }

// func Test_SFLC(t *testing.T) {

// 	//CPU full power
// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
// 	SchemeParams := hefloat.ParametersLiteral{
// 		LogN:            5,
// 		LogQ:            []int{50, 50, 50, 32, 32},
// 		LogP:            []int{52},
// 		LogDefaultScale: 40,
// 	}
// 	//parameter init
// 	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
// 	if err != nil {
// 		panic(err)
// 	}

// 	SF_LC, SFI_LC := matmult.GenSFMat_CL(params, []int{2, 2}, []int{2, 2})

// 	n := params.MaxSlots()

// 	pow5 := make([]int, (n<<1)+1)
// 	pow5[0] = 1
// 	for i := 1; i < (n<<1)+1; i++ {
// 		pow5[i] = pow5[i-1] * 5
// 		pow5[i] &= (n << 2) - 1
// 	}
// 	roots := ckks.GetRootsBigComplex(n<<2, params.EncodingPrecision())
// 	roots_complex := make([]complex128, 4*n)

// 	for i := range roots_complex {
// 		roots_complex[i] = roots[i].Complex128()
// 	}

// 	SF := make([][]complex128, n)

// 	for i := range SF {
// 		SF[i] = make([]complex128, n)
// 		for j := range SF[i] {
// 			idx := (pow5[i] * j) & ((n << 2) - 1)
// 			SF[i][j] = roots_complex[idx]
// 		}
// 	}

// 	SFI := make([][]complex128, n)
// 	for i := range SFI {
// 		SFI[i] = make([]complex128, n)
// 	}
// 	for i := range SFI {
// 		for j := range SFI[i] {
// 			idx := (pow5[i] * j) & ((n << 2) - 1)
// 			SFI[j][i] = cmplx.Conj(roots_complex[idx]) / complex((float64(n)), 0)
// 		}
// 	}

// 	SF_ := matmult.BitReversePermutationMatrix(n)
// 	for i := range SF_LC {
// 		SF_ = mul(SF_LC[i], SF_)
// 	}

// 	SFI_ := matmult.BitReversePermutationMatrix(n)
// 	// for i := range SFI_ {
// 	// 	for j := range SFI_[i] {
// 	// 		SFI_[i][j] /= complex(float64(n), 0)
// 	// 	}
// 	// }
// 	fmt.Println()
// 	for i := range SFI_LC {
// 		SFI_ = mul(SFI_, SFI_LC[len(SFI_LC)-i-1])
// 	}

// 	for i := range SFI {
// 		fmt.Println(SFI[i])
// 	}
// 	fmt.Println()
// 	for i := range SFI_ {
// 		fmt.Println(SFI_[i])
// 	}
// 	fmt.Println(len(SF_LC), len(SFI_LC))
// 	fmt.Println("is same ? : ", closeMat(SF, SF_, 0.0001))
// 	fmt.Println("is same ? : ", closeMat(SFI, SFI_, 0.0001))

// }

// func Test_C2SLC(t *testing.T) {

// 	//CPU full power
// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
// 	SchemeParams := hefloat.ParametersLiteral{
// 		LogN:            5,
// 		LogQ:            []int{48, 40, 40, 48},
// 		LogP:            []int{52},
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

// 	n := 1 << params.LogMaxSlots()

// 	var pk *rlwe.PublicKey
// 	var rlk *rlwe.RelinearizationKey
// 	var rtk []*rlwe.GaloisKey

// 	fmt.Println("generated bootstrapper end")
// 	pk = kgen.GenPublicKeyNew(sk)
// 	rlk = kgen.GenRelinearizationKeyNew(sk)

// 	// generate keys - Rotating key
// 	galEls := make([]uint64, 2*n)
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
// 	// btpevk, _, _ := btpParams.GenEvaluationKeys(sk)

// 	// btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
// 	// if err != nil {
// 	// 	panic(err)
// 	// }
// 	fmt.Println("generate Evaluator end")

// 	_, SFI := matmult.GenSFMat_CL(params, []int{2, 2}, []int{2, 2})
// 	scale := float64(1 << 40)
// 	mat0 := make([][][][]uint64, len(SFI))
// 	mat1 := make([][][][]uint64, len(SFI))
// 	mat0i := make([][][][]uint64, len(SFI))
// 	mat1i := make([][][][]uint64, len(SFI))

// 	for l := range SFI {
// 		mat0[l] = make([][][]uint64, len(params.Q()))
// 		mat1[l] = make([][][]uint64, len(params.Q()))
// 		mat0i[l] = make([][][]uint64, len(params.Q()))
// 		mat1i[l] = make([][][]uint64, len(params.Q()))

// 		for q := range mat0[l] {
// 			mat0[l][q] = make([][]uint64, 2*n)
// 			mat1[l][q] = make([][]uint64, 2*n)
// 			mat0i[l][q] = make([][]uint64, 2*n)
// 			mat1i[l][q] = make([][]uint64, 2*n)
// 			for i := range 2 * n {
// 				mat0[l][q][i] = make([]uint64, 2*n)
// 				mat1[l][q][i] = make([]uint64, 2*n)
// 				mat0i[l][q][i] = make([]uint64, 2*n)
// 				mat1i[l][q][i] = make([]uint64, 2*n)
// 				for j := range 2 * n {
// 					if i < n && j < n {
// 						if real(SFI[l][i][j]) >= 0 {
// 							mat0[l][q][i][j] = uint64(real(SFI[l][i][j]) * scale)
// 						} else {
// 							mat0[l][q][i][j] = params.Q()[q] - uint64(-real(SFI[l][i][j])*scale)
// 						}
// 						if imag(SFI[l][i][j]) >= 0 {
// 							mat0i[l][q][i][j] = uint64(imag(SFI[l][i][j]) * scale)
// 						} else {
// 							mat0i[l][q][i][j] = params.Q()[q] - uint64(-imag(SFI[l][i][j])*scale)
// 						}
// 					}
// 					if i >= n && j >= n {
// 						if real(SFI[l][i%n][j%n]) >= 0 {
// 							mat1[l][q][i][j] = uint64(real(SFI[l][i%n][j%n]) * scale)
// 						} else {
// 							mat1[l][q][i][j] = params.Q()[q] - uint64(-real(SFI[l][i%n][j%n])*scale)
// 						}
// 						if imag(SFI[l][i%n][j%n]) >= 0 {
// 							mat1i[l][q][i][j] = uint64(imag(SFI[l][i%n][j%n]) * scale)
// 						} else {
// 							mat1i[l][q][i][j] = params.Q()[q] - uint64(-imag(SFI[l][i%n][j%n])*scale)
// 						}

// 					}
// 				}
// 			}
// 		}
// 	}

// 	for l := range mat0 {
// 		fmt.Println("l is : ", l)
// 		for i := range mat0[l][0] {
// 			fmt.Println(mat0[l][0][i])
// 		}
// 		fmt.Println()
// 		fmt.Println()
// 		for i := range mat0[l][0] {
// 			fmt.Println(mat0i[l][0][i])
// 		}
// 		fmt.Println()
// 	}

// 	value := make([]float64, 2*n)
// 	for i := range value {
// 		value[i] = 0.001 * float64(i)
// 	}

// 	pt := hefloat.NewPlaintext(params, params.MaxLevel())
// 	pt.IsBatched = false

// 	encoder.Encode(value, pt)
// 	ct, _ := encryptor.EncryptNew(pt)
// 	cts := make([]*rlwe.Ciphertext, 2*n)
// 	for i := range cts {
// 		cts[i] = ct.CopyNew()
// 	}

// 	fmt.Println("start c2s")
// 	starttime = time.Now()
// 	ctT := transpose.Transpose(cts, params, evaluator, encoder, 2*n)
// 	// rev := matmult.BitReversePermutationMatrix(n)
// 	// matrev := make([][]uint64, 2*n)
// 	// for i := range matrev {
// 	// 	matrev[i] = make([]uint64, 2*n)
// 	// 	for j := range matrev[i] {
// 	// 		if (i < n && j < n) || (i >= n && j >= n) {
// 	// 			matrev[i][j] = uint64(real(rev[i%n][j%n]))
// 	// 		}
// 	// 	}
// 	// }
// 	// ctT = matmult.PPMM_Flint(ctT, matrev, params, 2*n)

// 	ctTC := make([]*rlwe.Ciphertext, 2*n)
// 	fmt.Println("ctT ctTC")
// 	for i := range ctTC {
// 		if i < n {
// 			ctTC[i], _ = evaluator.MulNew(ctT[i+n], -1)
// 		} else {
// 			ctTC[i] = ctT[i-n].CopyNew()
// 		}
// 	}
// 	// for l := range mat0 {
// 	// 	for q := range mat0[l] {
// 	// 		for i := range mat0[l][q] {
// 	// 			fmt.Println(mat0i[l][q][i])
// 	// 		}
// 	// 		fmt.Println()
// 	// 	}
// 	// }

// 	var res00, res01, res10, res11, res00i, res01i, res10i, res11i []*rlwe.Ciphertext
// 	for l := range len(SFI) {
// 		if l == 0 {
// 			res00 = matmult.PPMM_Flint_CRT(ctT, mat0[l], params, 2*n)
// 			res01 = matmult.PPMM_Flint_CRT(ctTC, mat1[l], params, 2*n)
// 			res00i = matmult.PPMM_Flint_CRT(ctT, mat0i[l], params, 2*n)
// 			res01i = matmult.PPMM_Flint_CRT(ctTC, mat1i[l], params, 2*n)

// 			res10 = matmult.PPMM_Flint_CRT(ctTC, mat0[l], params, 2*n)
// 			res11 = matmult.PPMM_Flint_CRT(ctT, mat1[l], params, 2*n)
// 			res10i = matmult.PPMM_Flint_CRT(ctTC, mat0i[l], params, 2*n)
// 			res11i = matmult.PPMM_Flint_CRT(ctT, mat1i[l], params, 2*n)

// 			for i := range 2 * n {
// 				evaluator.Mul(res00[i], 1.0/scale, res00[i])
// 				evaluator.Rescale(res00[i], res00[i])
// 				evaluator.Mul(res00i[i], 1.0/scale, res00i[i])
// 				evaluator.Rescale(res00i[i], res00i[i])

// 				evaluator.Mul(res01[i], 1.0/scale, res01[i])
// 				evaluator.Rescale(res01[i], res01[i])
// 				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
// 				evaluator.Rescale(res01i[i], res01i[i])

// 				evaluator.Mul(res10[i], 1.0/scale, res10[i])
// 				evaluator.Rescale(res10[i], res10[i])
// 				evaluator.Mul(res10i[i], 1.0/scale, res10i[i])
// 				evaluator.Rescale(res10i[i], res10i[i])

// 				evaluator.Mul(res11[i], 1.0/scale, res11[i])
// 				evaluator.Rescale(res11[i], res11[i])
// 				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
// 				evaluator.Rescale(res11i[i], res11i[i])
// 			}

// 		} else if l == len(SFI)-1 {
// 			temp1 := matmult.PPMM_Flint_CRT(res00, mat0[l], params, 2*n)
// 			temp1_ := matmult.PPMM_Flint_CRT(res00i, mat0i[l], params, 2*n)
// 			temp2 := matmult.PPMM_Flint_CRT(res01, mat1i[l], params, 2*n)
// 			temp2_ := matmult.PPMM_Flint_CRT(res01i, mat1[l], params, 2*n)
// 			for i := range 2 * n {
// 				evaluator.Sub(temp1[i], temp1_[i], res00[i])
// 				evaluator.Add(temp2[i], temp2_[i], res01i[i])
// 			}

// 			temp1 = matmult.PPMM_Flint_CRT(res10, mat0[l], params, 2*n)
// 			temp1_ = matmult.PPMM_Flint_CRT(res10i, mat0i[l], params, 2*n)
// 			temp2 = matmult.PPMM_Flint_CRT(res11, mat1i[l], params, 2*n)
// 			temp2_ = matmult.PPMM_Flint_CRT(res11i, mat1[l], params, 2*n)
// 			for i := range 2 * n {
// 				evaluator.Sub(temp1[i], temp1_[i], res10[i])
// 				evaluator.Add(temp2[i], temp2_[i], res11i[i])
// 			}
// 			for i := range 2 * n {
// 				evaluator.Mul(res00[i], 1.0/scale, res00[i])
// 				evaluator.Rescale(res00[i], res00[i])

// 				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
// 				evaluator.Rescale(res01i[i], res01i[i])

// 				evaluator.Mul(res10[i], 1.0/scale, res10[i])
// 				evaluator.Rescale(res10[i], res10[i])

// 				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
// 				evaluator.Rescale(res11i[i], res11i[i])
// 			}
// 		} else {
// 			temp1 := matmult.PPMM_Flint_CRT(res00, mat0[l], params, 2*n)
// 			temp1_ := matmult.PPMM_Flint_CRT(res00i, mat0i[l], params, 2*n)
// 			temp2 := matmult.PPMM_Flint_CRT(res00, mat0i[l], params, 2*n)
// 			temp2_ := matmult.PPMM_Flint_CRT(res00i, mat0[l], params, 2*n)
// 			for i := range 2 * n {
// 				evaluator.Sub(temp1[i], temp1_[i], res00[i])
// 				evaluator.Add(temp2[i], temp2_[i], res00i[i])
// 			}

// 			temp1 = matmult.PPMM_Flint_CRT(res01, mat1[l], params, 2*n)
// 			temp1_ = matmult.PPMM_Flint_CRT(res01i, mat1i[l], params, 2*n)
// 			temp2 = matmult.PPMM_Flint_CRT(res01, mat1i[l], params, 2*n)
// 			temp2_ = matmult.PPMM_Flint_CRT(res01i, mat1[l], params, 2*n)
// 			for i := range 2 * n {
// 				evaluator.Sub(temp1[i], temp1_[i], res01[i])
// 				evaluator.Add(temp2[i], temp2_[i], res01i[i])
// 			}

// 			temp1 = matmult.PPMM_Flint_CRT(res10, mat0[l], params, 2*n)
// 			temp1_ = matmult.PPMM_Flint_CRT(res10i, mat0i[l], params, 2*n)
// 			temp2 = matmult.PPMM_Flint_CRT(res10, mat0i[l], params, 2*n)
// 			temp2_ = matmult.PPMM_Flint_CRT(res10i, mat0[l], params, 2*n)
// 			for i := range 2 * n {
// 				evaluator.Sub(temp1[i], temp1_[i], res10[i])
// 				evaluator.Add(temp2[i], temp2_[i], res10i[i])
// 			}

// 			temp1 = matmult.PPMM_Flint_CRT(res11, mat1[l], params, 2*n)
// 			temp1_ = matmult.PPMM_Flint_CRT(res11i, mat1i[l], params, 2*n)
// 			temp2 = matmult.PPMM_Flint_CRT(res11, mat1i[l], params, 2*n)
// 			temp2_ = matmult.PPMM_Flint_CRT(res11i, mat1[l], params, 2*n)
// 			for i := range 2 * n {
// 				evaluator.Sub(temp1[i], temp1_[i], res11[i])
// 				evaluator.Add(temp2[i], temp2_[i], res11i[i])
// 			}

// 			for i := range 2 * n {
// 				evaluator.Mul(res00[i], 1.0/scale, res00[i])
// 				evaluator.Rescale(res00[i], res00[i])
// 				evaluator.Mul(res00i[i], 1.0/scale, res00i[i])
// 				evaluator.Rescale(res00i[i], res00i[i])

// 				evaluator.Mul(res01[i], 1.0/scale, res01[i])
// 				evaluator.Rescale(res01[i], res01[i])
// 				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
// 				evaluator.Rescale(res01i[i], res01i[i])

// 				evaluator.Mul(res10[i], 1.0/scale, res10[i])
// 				evaluator.Rescale(res10[i], res10[i])
// 				evaluator.Mul(res10i[i], 1.0/scale, res10i[i])
// 				evaluator.Rescale(res10i[i], res10i[i])

// 				evaluator.Mul(res11[i], 1.0/scale, res11[i])
// 				evaluator.Rescale(res11[i], res11[i])
// 				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
// 				evaluator.Rescale(res11i[i], res11i[i])
// 			}
// 		}
// 	}

// 	res0 := make([]*rlwe.Ciphertext, 2*n)
// 	for i := range res0 {
// 		res0[i], _ = evaluator.AddNew(res00[i], res01i[i])
// 	}
// 	res1 := make([]*rlwe.Ciphertext, 2*n)
// 	for i := range res1 {
// 		evaluator.Mul(res10[i], -1, res10[i])
// 		res1[i], _ = evaluator.AddNew(res10[i], res11i[i])
// 	}

// 	rev := matmult.BitReversePermutationMatrix(n)
// 	matrev := make([][]uint64, 2*n)
// 	for i := range matrev {
// 		matrev[i] = make([]uint64, 2*n)
// 		for j := range matrev[i] {
// 			if (i < n && j < n) || (i >= n && j >= n) {
// 				matrev[i][j] = uint64(real(rev[i%n][j%n]))
// 			}
// 		}
// 	}
// 	res0 = matmult.PPMM_Flint(res0, matrev, params, 2*n)
// 	res1 = matmult.PPMM_Flint(res1, matrev, params, 2*n)

// 	result0 := transpose.Transpose(res0, params, evaluator, encoder, 2*n)
// 	result1 := transpose.Transpose(res1, params, evaluator, encoder, 2*n)

// 	elapse = time.Since(starttime)
// 	fmt.Println(elapse)
// 	fmt.Println(result0[0].LogScale())
// 	fmt.Println(result1[0].LogScale())

// 	resvalue := make([]complex128, n)
// 	for i := range result0 {
// 		result0[i].IsBatched = true
// 		dept := decryptor.DecryptNew(result0[i])
// 		encoder.Decode(dept, resvalue)

// 		fmt.Println(resvalue)
// 	}
// 	fmt.Println()
// 	fmt.Println()
// 	fmt.Println()
// 	for i := range result1 {
// 		result1[i].IsBatched = true
// 		dept := decryptor.DecryptNew(result1[i])
// 		encoder.Decode(dept, resvalue)

// 		fmt.Println(resvalue)
// 	}
// 	fmt.Println(result0[0].LogScale())

// }

// func Test_S2CLC(t *testing.T) {

// 	//CPU full power
// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
// 	SchemeParams := hefloat.ParametersLiteral{
// 		LogN:            5,
// 		LogQ:            []int{48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
// 		LogP:            []int{52},
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

// 	n := 1 << params.LogMaxSlots()

// 	var pk *rlwe.PublicKey
// 	var rlk *rlwe.RelinearizationKey
// 	var rtk []*rlwe.GaloisKey

// 	fmt.Println("generated bootstrapper end")
// 	pk = kgen.GenPublicKeyNew(sk)
// 	rlk = kgen.GenRelinearizationKeyNew(sk)

// 	// generate keys - Rotating key
// 	galEls := make([]uint64, 2*n)
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
// 	// btpevk, _, _ := btpParams.GenEvaluationKeys(sk)

// 	// btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
// 	// if err != nil {
// 	// 	panic(err)
// 	// }
// 	fmt.Println("generate Evaluator end")

// 	SF, _ := matmult.GenSFMat_CL(params, []int{2, 2}, []int{2, 2})
// 	scale := float64(1 << 25)
// 	mat0 := make([][][][]uint64, len(SF))
// 	mat1 := make([][][][]uint64, len(SF))
// 	mat0i := make([][][][]uint64, len(SF))
// 	mat1i := make([][][][]uint64, len(SF))

// 	for l := range SF {
// 		mat0[l] = make([][][]uint64, len(params.Q()))
// 		mat1[l] = make([][][]uint64, len(params.Q()))
// 		mat0i[l] = make([][][]uint64, len(params.Q()))
// 		mat1i[l] = make([][][]uint64, len(params.Q()))

// 		for q := range mat0[l] {
// 			mat0[l][q] = make([][]uint64, 2*n)
// 			mat1[l][q] = make([][]uint64, 2*n)
// 			mat0i[l][q] = make([][]uint64, 2*n)
// 			mat1i[l][q] = make([][]uint64, 2*n)
// 			for i := range 2 * n {
// 				mat0[l][q][i] = make([]uint64, 2*n)
// 				mat1[l][q][i] = make([]uint64, 2*n)
// 				mat0i[l][q][i] = make([]uint64, 2*n)
// 				mat1i[l][q][i] = make([]uint64, 2*n)
// 				for j := range 2 * n {
// 					if i < n && j < n {
// 						if real(SF[l][i][j]) >= 0 {
// 							mat0[l][q][i][j] = uint64(real(SF[l][i][j]) * scale)
// 						} else {
// 							mat0[l][q][i][j] = params.Q()[q] - uint64(-real(SF[l][i][j])*scale)
// 						}
// 						if imag(SF[l][i][j]) >= 0 {
// 							mat0i[l][q][i][j] = uint64(imag(SF[l][i][j]) * scale)
// 						} else {
// 							mat0i[l][q][i][j] = params.Q()[q] - uint64(-imag(SF[l][i][j])*scale)
// 						}
// 					}
// 					if i >= n && j >= n {
// 						if real(SF[l][i%n][j%n]) >= 0 {
// 							mat1[l][q][i][j] = uint64(real(SF[l][i%n][j%n]) * scale)
// 						} else {
// 							mat1[l][q][i][j] = params.Q()[q] - uint64(-real(SF[l][i%n][j%n])*scale)
// 						}
// 						if imag(SF[l][i%n][j%n]) >= 0 {
// 							mat1i[l][q][i][j] = uint64(imag(SF[l][i%n][j%n]) * scale)
// 						} else {
// 							mat1i[l][q][i][j] = params.Q()[q] - uint64(-imag(SF[l][i%n][j%n])*scale)
// 						}

// 					}
// 				}
// 			}
// 		}
// 	}

// 	value := make([]complex128, n)
// 	for i := range value {
// 		value[i] = complex(0.001*float64(i), 0.001*float64(n-i))
// 	}

// 	pt := hefloat.NewPlaintext(params, params.MaxLevel())
// 	pt.IsBatched = true

// 	encoder.Encode(value, pt)
// 	ct, _ := encryptor.EncryptNew(pt)
// 	cts := make([]*rlwe.Ciphertext, 2*n)
// 	for i := range cts {
// 		cts[i] = ct.CopyNew()
// 	}

// 	fmt.Println("start c2s")
// 	starttime = time.Now()
// 	ctT := transpose.Transpose(cts, params, evaluator, encoder, 2*n)
// 	rev := matmult.BitReversePermutationMatrix(n)
// 	matrev := make([][]uint64, 2*n)
// 	for i := range matrev {
// 		matrev[i] = make([]uint64, 2*n)
// 		for j := range matrev[i] {
// 			if (i < n && j < n) || (i >= n && j >= n) {
// 				matrev[i][j] = uint64(real(rev[i%n][j%n]))
// 			}
// 		}
// 	}
// 	ctT = matmult.PPMM_Flint(ctT, matrev, params, 2*n)

// 	ctTC := make([]*rlwe.Ciphertext, 2*n)
// 	fmt.Println("ctT ctTC")
// 	for i := range ctTC {
// 		if i < n {
// 			ctTC[i], _ = evaluator.MulNew(ctT[i+n], -1)
// 		} else {
// 			ctTC[i] = ctT[i-n].CopyNew()
// 		}
// 	}

// 	var res00, res01, res10, res11, res00i, res01i, res10i, res11i []*rlwe.Ciphertext
// 	for l := range len(SF) {
// 		if l == 0 {
// 			res00 = matmult.PPMM_Flint_CRT(ctT, mat0[l], params, 2*n)
// 			res00i = matmult.PPMM_Flint_CRT(ctT, mat0i[l], params, 2*n)
// 			res01 = matmult.PPMM_Flint_CRT(ctTC, mat0[l], params, 2*n)
// 			res01i = matmult.PPMM_Flint_CRT(ctTC, mat0i[l], params, 2*n)

// 			res10 = matmult.PPMM_Flint_CRT(ctT, mat1[l], params, 2*n)
// 			res10i = matmult.PPMM_Flint_CRT(ctT, mat1i[l], params, 2*n)
// 			res11 = matmult.PPMM_Flint_CRT(ctTC, mat1[l], params, 2*n)
// 			res11i = matmult.PPMM_Flint_CRT(ctTC, mat1i[l], params, 2*n)

// 			for i := range 2 * n {
// 				evaluator.Mul(res00[i], 1.0/scale, res00[i])
// 				evaluator.Rescale(res00[i], res00[i])
// 				evaluator.Mul(res00i[i], 1.0/scale, res00i[i])
// 				evaluator.Rescale(res00i[i], res00i[i])

// 				evaluator.Mul(res01[i], 1.0/scale, res01[i])
// 				evaluator.Rescale(res01[i], res01[i])
// 				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
// 				evaluator.Rescale(res01i[i], res01i[i])

// 				evaluator.Mul(res10[i], 1.0/scale, res10[i])
// 				evaluator.Rescale(res10[i], res10[i])
// 				evaluator.Mul(res10i[i], 1.0/scale, res10i[i])
// 				evaluator.Rescale(res10i[i], res10i[i])

// 				evaluator.Mul(res11[i], 1.0/scale, res11[i])
// 				evaluator.Rescale(res11[i], res11[i])
// 				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
// 				evaluator.Rescale(res11i[i], res11i[i])
// 			}

// 		} else if l == len(SF)-1 {
// 			temp1 := matmult.PPMM_Flint_CRT(res00, mat0[l], params, 2*n)
// 			temp1_ := matmult.PPMM_Flint_CRT(res00i, mat0i[l], params, 2*n)
// 			temp2 := matmult.PPMM_Flint_CRT(res01, mat0i[l], params, 2*n)
// 			temp2_ := matmult.PPMM_Flint_CRT(res01i, mat0[l], params, 2*n)
// 			for i := range 2 * n {
// 				evaluator.Sub(temp1[i], temp1_[i], res00[i])
// 				evaluator.Add(temp2[i], temp2_[i], res01i[i])
// 			}

// 			temp1 = matmult.PPMM_Flint_CRT(res10, mat1[l], params, 2*n)
// 			temp1_ = matmult.PPMM_Flint_CRT(res10i, mat1i[l], params, 2*n)
// 			temp2 = matmult.PPMM_Flint_CRT(res11, mat1i[l], params, 2*n)
// 			temp2_ = matmult.PPMM_Flint_CRT(res11i, mat1[l], params, 2*n)
// 			for i := range 2 * n {
// 				evaluator.Sub(temp1[i], temp1_[i], res10[i])
// 				evaluator.Add(temp2[i], temp2_[i], res11i[i])
// 			}
// 			for i := range 2 * n {
// 				evaluator.Mul(res00[i], 1.0/scale, res00[i])
// 				evaluator.Rescale(res00[i], res00[i])

// 				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
// 				evaluator.Rescale(res01i[i], res01i[i])

// 				evaluator.Mul(res10[i], 1.0/scale, res10[i])
// 				evaluator.Rescale(res10[i], res10[i])

// 				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
// 				evaluator.Rescale(res11i[i], res11i[i])
// 			}
// 		} else {
// 			temp1 := matmult.PPMM_Flint_CRT(res00, mat0[l], params, 2*n)
// 			temp1_ := matmult.PPMM_Flint_CRT(res00i, mat0i[l], params, 2*n)
// 			temp2 := matmult.PPMM_Flint_CRT(res00, mat0i[l], params, 2*n)
// 			temp2_ := matmult.PPMM_Flint_CRT(res00i, mat0[l], params, 2*n)
// 			for i := range 2 * n {
// 				evaluator.Sub(temp1[i], temp1_[i], res00[i])
// 				evaluator.Add(temp2[i], temp2_[i], res00i[i])
// 			}

// 			temp1 = matmult.PPMM_Flint_CRT(res01, mat0[l], params, 2*n)
// 			temp1_ = matmult.PPMM_Flint_CRT(res01i, mat0i[l], params, 2*n)
// 			temp2 = matmult.PPMM_Flint_CRT(res01, mat0i[l], params, 2*n)
// 			temp2_ = matmult.PPMM_Flint_CRT(res01i, mat0[l], params, 2*n)
// 			for i := range 2 * n {
// 				evaluator.Sub(temp1[i], temp1_[i], res01[i])
// 				evaluator.Add(temp2[i], temp2_[i], res01i[i])
// 			}

// 			temp1 = matmult.PPMM_Flint_CRT(res10, mat1[l], params, 2*n)
// 			temp1_ = matmult.PPMM_Flint_CRT(res10i, mat1i[l], params, 2*n)
// 			temp2 = matmult.PPMM_Flint_CRT(res10, mat1i[l], params, 2*n)
// 			temp2_ = matmult.PPMM_Flint_CRT(res10i, mat1[l], params, 2*n)
// 			for i := range 2 * n {
// 				evaluator.Sub(temp1[i], temp1_[i], res10[i])
// 				evaluator.Add(temp2[i], temp2_[i], res10i[i])
// 			}

// 			temp1 = matmult.PPMM_Flint_CRT(res11, mat1[l], params, 2*n)
// 			temp1_ = matmult.PPMM_Flint_CRT(res11i, mat1i[l], params, 2*n)
// 			temp2 = matmult.PPMM_Flint_CRT(res11, mat1i[l], params, 2*n)
// 			temp2_ = matmult.PPMM_Flint_CRT(res11i, mat1[l], params, 2*n)
// 			for i := range 2 * n {
// 				evaluator.Sub(temp1[i], temp1_[i], res11[i])
// 				evaluator.Add(temp2[i], temp2_[i], res11i[i])
// 			}

// 			for i := range 2 * n {
// 				evaluator.Mul(res00[i], 1.0/scale, res00[i])
// 				evaluator.Rescale(res00[i], res00[i])
// 				evaluator.Mul(res00i[i], 1.0/scale, res00i[i])
// 				evaluator.Rescale(res00i[i], res00i[i])

// 				evaluator.Mul(res01[i], 1.0/scale, res01[i])
// 				evaluator.Rescale(res01[i], res01[i])
// 				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
// 				evaluator.Rescale(res01i[i], res01i[i])

// 				evaluator.Mul(res10[i], 1.0/scale, res10[i])
// 				evaluator.Rescale(res10[i], res10[i])
// 				evaluator.Mul(res10i[i], 1.0/scale, res10i[i])
// 				evaluator.Rescale(res10i[i], res10i[i])

// 				evaluator.Mul(res11[i], 1.0/scale, res11[i])
// 				evaluator.Rescale(res11[i], res11[i])
// 				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
// 				evaluator.Rescale(res11i[i], res11i[i])
// 			}
// 		}
// 	}

// 	res := make([]*rlwe.Ciphertext, 2*n)
// 	for i := range res {
// 		res[i], _ = evaluator.AddNew(res00[i], res01i[i])
// 		evaluator.Add(res[i], res10[i], res[i])
// 		evaluator.Add(res[i], res11i[i], res[i])
// 	}

// 	// result0 := transpose.Transpose(res, params, evaluator, encoder, 2*n)
// 	result0 := res

// 	elapse = time.Since(starttime)
// 	fmt.Println(elapse)
// 	fmt.Println(result0[0].LogScale())

// 	resvalue := make([]float64, 2*n)
// 	for i := range result0 {
// 		result0[i].IsBatched = false
// 		dept := decryptor.DecryptNew(result0[i])
// 		encoder.Decode(dept, resvalue)

// 		fmt.Println(resvalue)
// 	}

// }

// func Test_C2SLC_Opt(t *testing.T) {

// 	//CPU full power
// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
// 	SchemeParams := hefloat.ParametersLiteral{
// 		LogN:            5,
// 		LogQ:            []int{48, 40, 40, 48},
// 		LogP:            []int{52},
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

// 	n := 1 << params.LogMaxSlots()

// 	var pk *rlwe.PublicKey
// 	var rlk *rlwe.RelinearizationKey
// 	var rtk []*rlwe.GaloisKey

// 	fmt.Println("generated bootstrapper end")
// 	pk = kgen.GenPublicKeyNew(sk)
// 	rlk = kgen.GenRelinearizationKeyNew(sk)

// 	// generate keys - Rotating key
// 	galEls := make([]uint64, n*2)
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
// 	// btpevk, _, _ := btpParams.GenEvaluationKeys(sk)

// 	// btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
// 	// if err != nil {
// 	// 	panic(err)
// 	// }
// 	fmt.Println("generate Evaluator end")

// 	fmt.Println("ckks log degree : ", params.LogN())

// 	CL_arr := []int{2, 2}
// 	_, SFI := matmult.GenSFMat_CL_Slow(params, CL_arr, CL_arr)
// 	scale := float64(1 << 10)
// 	mat0 := make([][][][][]uint64, len(SFI))
// 	mat0i := make([][][][][]uint64, len(SFI))
// 	mat0si := make([][][][][]uint64, len(SFI))

// 	inter_it := n
// 	for l := range SFI {
// 		inter := inter_it >> CL_arr[l]
// 		llen := (1 << CL_arr[l])
// 		mat0[l] = make([][][][]uint64, n/llen)
// 		mat0i[l] = make([][][][]uint64, n/llen)
// 		mat0si[l] = make([][][][]uint64, n/llen)
// 		for t := range n / llen {
// 			mat0[l][t] = make([][][]uint64, len(params.Q()))
// 			mat0i[l][t] = make([][][]uint64, len(params.Q()))
// 			mat0si[l][t] = make([][][]uint64, len(params.Q()))
// 			stpoint := inter_it*int(t/inter) + (t % inter)
// 			for q := range len(params.Q()) {
// 				mat0[l][t][q] = make([][]uint64, llen)
// 				mat0i[l][t][q] = make([][]uint64, llen)
// 				mat0si[l][t][q] = make([][]uint64, llen)
// 				for i := range llen {
// 					mat0[l][t][q][i] = make([]uint64, llen)
// 					mat0i[l][t][q][i] = make([]uint64, llen)
// 					mat0si[l][t][q][i] = make([]uint64, llen)
// 					for j := range llen {
// 						if real(SFI[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
// 							mat0[l][t][q][i][j] = uint64(real(SFI[l][stpoint+inter*i][stpoint+inter*j]) * scale)
// 						} else {
// 							mat0[l][t][q][i][j] = params.Q()[q] - uint64(-real(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale)
// 						}
// 						if imag(SFI[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
// 							mat0i[l][t][q][i][j] = uint64(imag(SFI[l][stpoint+inter*i][stpoint+inter*j]) * scale)
// 						} else {
// 							mat0i[l][t][q][i][j] = params.Q()[q] - uint64(-imag(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale)
// 						}
// 						mat0si[l][t][q][i][j] = (mat0[l][t][q][i][j] + mat0i[l][t][q][i][j]) % params.Q()[q]
// 					}
// 				}
// 			}
// 		}

// 		inter_it = inter
// 	}
// 	fmt.Println(params.Q())
// 	fmt.Println(mat0)
// 	fmt.Println()
// 	fmt.Println(mat0i)

// 	value := make([]float64, 2*n)
// 	for i := range value {
// 		value[i] = 0.001 * float64(i)
// 	}

// 	pt := hefloat.NewPlaintext(params, params.MaxLevel())
// 	pt.IsBatched = false

// 	encoder.Encode(value, pt)
// 	ct, _ := encryptor.EncryptNew(pt)
// 	cts := make([]*rlwe.Ciphertext, 2*n)
// 	for i := range cts {
// 		cts[i] = ct.CopyNew()
// 	}

// 	res00 := make([]*rlwe.Ciphertext, 2*n)
// 	res00i := make([]*rlwe.Ciphertext, 2*n)
// 	res01 := make([]*rlwe.Ciphertext, 2*n)
// 	res01i := make([]*rlwe.Ciphertext, 2*n)

// 	res10 := make([]*rlwe.Ciphertext, 2*n)
// 	res10i := make([]*rlwe.Ciphertext, 2*n)
// 	res11 := make([]*rlwe.Ciphertext, 2*n)
// 	res11i := make([]*rlwe.Ciphertext, 2*n)
// 	temp1 := make([]*rlwe.Ciphertext, 2*n)
// 	temp1_ := make([]*rlwe.Ciphertext, 2*n)
// 	temp2 := make([]*rlwe.Ciphertext, 2*n)
// 	temp2_ := make([]*rlwe.Ciphertext, 2*n)
// 	ctZero := util.CtZero(params, encoder, encryptor)
// 	for i := range 2 * n {
// 		res00[i] = util.CtZero(params, encoder, encryptor)
// 		res00i[i] = util.CtZero(params, encoder, encryptor)
// 		res01[i] = util.CtZero(params, encoder, encryptor)
// 		res01i[i] = util.CtZero(params, encoder, encryptor)

// 		res10[i] = util.CtZero(params, encoder, encryptor)
// 		res10i[i] = util.CtZero(params, encoder, encryptor)
// 		res11[i] = util.CtZero(params, encoder, encryptor)
// 		res11i[i] = util.CtZero(params, encoder, encryptor)

// 		temp1[i] = util.CtZero(params, encoder, encryptor)
// 		temp1_[i] = util.CtZero(params, encoder, encryptor)
// 		temp2[i] = util.CtZero(params, encoder, encryptor)
// 		temp2_[i] = util.CtZero(params, encoder, encryptor)
// 	}

// 	fmt.Println("start c2s")
// 	starttime = time.Now()
// 	ctT := transpose.Transpose(cts, params, evaluator, encoder, 2*n)
// 	// ctT := cts
// 	ctTC := make([]*rlwe.Ciphertext, 2*n)
// 	fmt.Println("ctT ctTC")
// 	for i := range ctTC {
// 		if i < n {
// 			ctTC[i], _ = evaluator.MulNew(ctT[i+n], -1)
// 		} else {
// 			ctTC[i] = ctT[i-n].CopyNew()
// 		}
// 	}

// 	//var res00, res01, res10, res11, res00i, res01i, res10i, res11i []*rlwe.Ciphertext
// 	inter_it = n
// 	for l := range len(SFI) {
// 		inter := inter_it >> CL_arr[l]
// 		llen := (1 << CL_arr[l])
// 		if l == 0 {
// 			for t := range n / llen {
// 				cts_temp := make([]*rlwe.Ciphertext, llen)
// 				stpoint := (t % inter) + inter_it*int(t/inter)
// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = ctT[stpoint+inter*idx_ll]
// 				}
// 				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					res00[stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = ctT[stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					res00i[stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = ctTC[n+stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					res01[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = ctTC[n+stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					res01i[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				//10~11
// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = ctTC[stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					res10[stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = ctTC[stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					res10i[stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = ctT[n+stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					res11[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = ctT[n+stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					res11i[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}
// 			}
// 			for i := range temp1 {
// 				temp1[i] = nil
// 				temp1_[i] = nil
// 				temp2[i] = nil
// 				temp2_[i] = nil
// 			}
// 			for i := range 2 * n {
// 				evaluator.Mul(res00[i], 1.0/scale, res00[i])
// 				evaluator.Rescale(res00[i], res00[i])
// 				evaluator.Mul(res00i[i], 1.0/scale, res00i[i])
// 				evaluator.Rescale(res00i[i], res00i[i])

// 				evaluator.Mul(res01[i], 1.0/scale, res01[i])
// 				evaluator.Rescale(res01[i], res01[i])
// 				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
// 				evaluator.Rescale(res01i[i], res01i[i])

// 				evaluator.Mul(res10[i], 1.0/scale, res10[i])
// 				evaluator.Rescale(res10[i], res10[i])
// 				evaluator.Mul(res10i[i], 1.0/scale, res10i[i])
// 				evaluator.Rescale(res10i[i], res10i[i])

// 				evaluator.Mul(res11[i], 1.0/scale, res11[i])
// 				evaluator.Rescale(res11[i], res11[i])
// 				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
// 				evaluator.Rescale(res11i[i], res11i[i])

// 			}

// 		} else if l == len(SFI)-1 {
// 			for t := range n / llen {
// 				cts_temp := make([]*rlwe.Ciphertext, llen)
// 				stpoint := (t % inter) + inter_it*int(t/inter)

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res00[stpoint+inter*idx_ll]
// 				}
// 				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					temp1[stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res00i[stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					temp1_[stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res01[n+stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					temp2[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res01i[n+stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					temp2_[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}
// 			}
// 			res00 = matmult.SubMany(temp1, temp1_, evaluator)
// 			res01i = matmult.AddMany(temp2, temp2_, evaluator)
// 			for i := range temp1 {
// 				temp1[i] = nil
// 				temp1_[i] = nil
// 				temp2[i] = nil
// 				temp2_[i] = nil
// 			}

// 			for t := range n / llen {
// 				cts_temp := make([]*rlwe.Ciphertext, llen)
// 				stpoint := (t % inter) + inter_it*int(t/inter)

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res10[stpoint+inter*idx_ll]
// 				}
// 				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					temp1[stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res10i[stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					temp1_[stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res11[n+stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					temp2[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res11i[n+stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					temp2_[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}
// 			}
// 			res10 = matmult.SubMany(temp1_, temp1, evaluator)
// 			res11i = matmult.AddMany(temp2, temp2_, evaluator)

// 			for i := range 2 * n {
// 				if res00[i] != nil {
// 					evaluator.Mul(res00[i], 1.0/scale, res00[i])
// 					evaluator.Rescale(res00[i], res00[i])
// 				}
// 				if res01i[i] != nil {
// 					evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
// 					evaluator.Rescale(res01i[i], res01i[i])
// 				}
// 				if res10[i] != nil {
// 					evaluator.Mul(res10[i], 1.0/scale, res10[i])
// 					evaluator.Rescale(res10[i], res10[i])
// 				}
// 				if res11i[i] != nil {
// 					evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
// 					evaluator.Rescale(res11i[i], res11i[i])
// 				}
// 			}
// 		} else {
// 			m1 := make([]*rlwe.Ciphertext, 2*n)
// 			m2 := make([]*rlwe.Ciphertext, 2*n)
// 			temp := make([]*rlwe.Ciphertext, 2*n)
// 			m3 := make([]*rlwe.Ciphertext, 2*n)
// 			m4 := make([]*rlwe.Ciphertext, 2*n)
// 			for i := range 2 * n {
// 				m1[i] = ctZero.CopyNew()
// 				m2[i] = ctZero.CopyNew()
// 				m3[i] = ctZero.CopyNew()
// 				m4[i] = ctZero.CopyNew()
// 				temp[i] = ctZero.CopyNew()
// 			}
// 			for t := range n / llen {
// 				cts_temp := make([]*rlwe.Ciphertext, llen)
// 				stpoint := (t % inter) + inter_it*int(t/inter)

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res00[stpoint+inter*idx_ll]
// 				}
// 				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					m1[stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res00i[stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					m2[stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}
// 			}
// 			temp = matmult.AddMany(res00, res00i, evaluator)
// 			for t := range n / llen {
// 				cts_temp := make([]*rlwe.Ciphertext, llen)
// 				stpoint := (t % inter) + inter_it*int(t/inter)

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = temp[stpoint+inter*idx_ll]
// 				}
// 				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0si[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					m3[stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}
// 			}
// 			m4 = matmult.AddMany(m1, m2, evaluator)

// 			res00 = matmult.SubMany(m1, m2, evaluator)
// 			res00i = matmult.SubMany(m3, m4, evaluator)

// 			for i := range 2 * n {
// 				m1[i] = ctZero.CopyNew()
// 				m2[i] = ctZero.CopyNew()
// 				m3[i] = ctZero.CopyNew()
// 				m4[i] = ctZero.CopyNew()
// 				temp[i] = ctZero.CopyNew()
// 			}
// 			for t := range n / llen {
// 				cts_temp := make([]*rlwe.Ciphertext, llen)
// 				stpoint := (t % inter) + inter_it*int(t/inter)

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res01[n+stpoint+inter*idx_ll]
// 				}
// 				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					m1[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res01i[n+stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					m2[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}
// 			}
// 			temp = matmult.AddMany(res01, res01i, evaluator)
// 			for t := range n / llen {
// 				cts_temp := make([]*rlwe.Ciphertext, llen)
// 				stpoint := (t % inter) + inter_it*int(t/inter)

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = temp[n+stpoint+inter*idx_ll]
// 				}
// 				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0si[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					m3[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}
// 			}
// 			m4 = matmult.AddMany(m1, m2, evaluator)
// 			res01 = matmult.SubMany(m1, m2, evaluator)
// 			res01i = matmult.SubMany(m3, m4, evaluator)

// 			for i := range 2 * n {
// 				m1[i] = ctZero.CopyNew()
// 				m2[i] = ctZero.CopyNew()
// 				m3[i] = ctZero.CopyNew()
// 				m4[i] = ctZero.CopyNew()
// 				temp[i] = ctZero.CopyNew()
// 			}
// 			for t := range n / llen {
// 				cts_temp := make([]*rlwe.Ciphertext, llen)
// 				stpoint := (t % inter) + inter_it*int(t/inter)

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res10[stpoint+inter*idx_ll]
// 				}
// 				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					m1[stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res10i[stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					m2[stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}
// 			}
// 			temp = matmult.AddMany(res10, res10i, evaluator)
// 			for t := range n / llen {
// 				cts_temp := make([]*rlwe.Ciphertext, llen)
// 				stpoint := (t % inter) + inter_it*int(t/inter)

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = temp[stpoint+inter*idx_ll]
// 				}
// 				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0si[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					m3[stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}
// 			}
// 			m4 = matmult.AddMany(m1, m2, evaluator)
// 			res10 = matmult.SubMany(m1, m2, evaluator)
// 			res10i = matmult.SubMany(m3, m4, evaluator)

// 			for i := range 2 * n {
// 				m1[i] = ctZero.CopyNew()
// 				m2[i] = ctZero.CopyNew()
// 				m3[i] = ctZero.CopyNew()
// 				m4[i] = ctZero.CopyNew()
// 				temp[i] = ctZero.CopyNew()
// 			}
// 			for t := range n / llen {
// 				cts_temp := make([]*rlwe.Ciphertext, llen)
// 				stpoint := (t % inter) + inter_it*int(t/inter)

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res11[n+stpoint+inter*idx_ll]
// 				}
// 				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					m1[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = res11i[n+stpoint+inter*idx_ll]
// 				}
// 				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					m2[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}
// 			}
// 			temp = matmult.AddMany(res11, res11i, evaluator)
// 			for t := range n / llen {
// 				cts_temp := make([]*rlwe.Ciphertext, llen)
// 				stpoint := (t % inter) + inter_it*int(t/inter)

// 				for idx_ll := range llen {
// 					cts_temp[idx_ll] = temp[n+stpoint+inter*idx_ll]
// 				}
// 				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0si[l][t], llen, llen, 2*n, params)
// 				for idx_ll := range llen {
// 					m3[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
// 				}
// 			}
// 			m4 = matmult.AddMany(m1, m2, evaluator)
// 			res11 = matmult.SubMany(m1, m2, evaluator)
// 			res11i = matmult.SubMany(m3, m4, evaluator)

// 			for i := range 2 * n {
// 				evaluator.Mul(res00[i], 1.0/scale, res00[i])
// 				evaluator.Rescale(res00[i], res00[i])
// 				evaluator.Mul(res00i[i], 1.0/scale, res00i[i])
// 				evaluator.Rescale(res00i[i], res00i[i])

// 				evaluator.Mul(res01[i], 1.0/scale, res01[i])
// 				evaluator.Rescale(res01[i], res01[i])
// 				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
// 				evaluator.Rescale(res01i[i], res01i[i])

// 				evaluator.Mul(res10[i], 1.0/scale, res10[i])
// 				evaluator.Rescale(res10[i], res10[i])
// 				evaluator.Mul(res10i[i], 1.0/scale, res10i[i])
// 				evaluator.Rescale(res10i[i], res10i[i])

// 				evaluator.Mul(res11[i], 1.0/scale, res11[i])
// 				evaluator.Rescale(res11[i], res11[i])
// 				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
// 				evaluator.Rescale(res11i[i], res11i[i])
// 			}
// 		}
// 		inter_it = inter
// 	}

// 	res0 := matmult.AddMany(res00, res01i, evaluator)
// 	res1 := matmult.AddMany(res10, res11i, evaluator)
// 	// res0 := make([]*rlwe.Ciphertext, 2*n)
// 	// res1 := make([]*rlwe.Ciphertext, 2*n)
// 	// for i := range res0 {
// 	// 	fmt.Println(res00[i].Scale.Value.Float64())
// 	// 	fmt.Println(res01i[i].Scale.Value.Float64())
// 	// 	res0[i], _ = evaluator.AddNew(res00[i], res01i[i])
// 	// 	res1[i], _ = evaluator.AddNew(res10[i], res11i[i])
// 	// }

// 	rev := matmult.BitReversePermutationMatrix(n)
// 	matrev := make([][]uint64, 2*n)
// 	for i := range matrev {
// 		matrev[i] = make([]uint64, 2*n)
// 		for j := range matrev[i] {
// 			if (i < n && j < n) || (i >= n && j >= n) {
// 				matrev[i][j] = uint64(real(rev[i%n][j%n]))
// 			}
// 		}
// 	}
// 	res0 = matmult.PPMM_Flint(res0, matrev, params, 2*n)
// 	res1 = matmult.PPMM_Flint(res1, matrev, params, 2*n)

// 	result0 := transpose.Transpose(res0, params, evaluator, encoder, 2*n)
// 	result1 := transpose.Transpose(res1, params, evaluator, encoder, 2*n)
// 	// result0 := res0
// 	// result1 := res1
// 	elapse = time.Since(starttime)
// 	fmt.Println(elapse)
// 	fmt.Println(result0[0].LogScale())
// 	fmt.Println(result1[0].LogScale())

// 	resvalue := make([]complex128, n)
// 	for i := range result0 {
// 		result0[i].IsBatched = true
// 		dept := decryptor.DecryptNew(result0[i])
// 		encoder.Decode(dept, resvalue)

// 		fmt.Println(resvalue)
// 	}
// 	fmt.Println()
// 	fmt.Println()
// 	fmt.Println()
// 	for i := range result1 {
// 		result1[i].IsBatched = true
// 		dept := decryptor.DecryptNew(result1[i])
// 		encoder.Decode(dept, resvalue)

// 		fmt.Println(resvalue)
// 	}
// 	fmt.Println(result0[0].LogScale())

// }

// func Test_C2S_New(t *testing.T) {

// 	//CPU full power
// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
// 	SchemeParams := hefloat.ParametersLiteral{
// 		LogN:            10,
// 		LogQ:            []int{48, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
// 		LogP:            []int{52, 52, 52},
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
// 	n := N / 2
// 	sparseN := N

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
// 		i := i
// 		gal := galEls[i]
// 		go func() {
// 			defer wg.Done()
// 			kgen_ := rlwe.NewKeyGenerator(params)
// 			rtk[i] = kgen_.GenGaloisKeyNew(gal, sk)
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

// 	fmt.Println("ckks log degree : ", params.LogN())

// 	CL_arr := []int{5, 4}
// 	_, SFI := matmult.GenSFMat_CL(params, CL_arr, CL_arr)
// 	scale := float64(1 << 20)
// 	mat0 := make([][][]float64, len(SFI))
// 	mat0i := make([][][]float64, len(SFI))
// 	// mat0si := make([][][]float64, len(SFI))
// 	ringQ := params.RingQ()
// 	value := make([]float64, N)

// 	pt := hefloat.NewPlaintext(params, params.MaxLevel())
// 	pt.IsBatched = false

// 	encoder.Encode(value, pt)
// 	cts := make([]*rlwe.Ciphertext, N)
// 	for i := range cts {
// 		value := make([]float64, N)
// 		for j := range value {
// 			value[j] = 0.0001 * float64(j)
// 		}
// 		encoder.Encode(value, pt)
// 		cts[i], _ = encryptor.EncryptNew(pt)
// 		ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
// 		ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
// 	}

// 	P := []uint64{3422539, 3370361, 3231143, 3545881, 3577031, 3832931, 4064197, 3617099, 3651497, 3711319, 3439693, 3502001, 3555509, 3552013, 4031179, 4115407, 3167453, 3365393, 3291143, 3204973, 4182419, 3495781, 3315883, 3403391, 3529153, 3390899, 3453773, 3705469, 3180337, 4091993, 3503221, 3598949, 3822277, 3277853, 3547249, 3278053, 3696257, 3849409, 3725257, 3239449, 3730721, 3393619, 3361363, 3732997, 3661573, 3158971, 3516031, 3737039, 3882649, 3614969, 3518491, 3169759, 3326417, 4165333, 3853097, 3845357, 3721603, 3494831, 3255467, 3442987, 3381641, 4188433, 3960053, 3825473, 3269713, 3373781, 3403843, 4177609, 3265337, 3382231, 3342137, 3330179, 3272629, 3725357, 3667453, 3960049, 3435323, 3664249, 3632423, 3515269, 3784733, 3377657, 4064143, 3702119, 3835367, 3564937, 3507397, 3345877, 4169129, 3206783, 3397769, 4145293, 3773477, 3229319, 3161617, 3517427, 3456743, 3687163, 3389423, 3553541}
// 	PLevel := 30
// 	P = P[:PLevel+1]
// 	ringP, _ := ring.NewRing(N, P)
// 	be := matmult.NewBasisExtender(ringQ, ringP, []matmult.Key{{params.MaxLevel(), PLevel}}, []matmult.Key{{PLevel, params.MaxLevel()}})

// 	fmt.Println("mat gen start")
// 	inter_it := n
// 	for l := range SFI {
// 		inter := inter_it >> CL_arr[l]
// 		llen := (1 << CL_arr[l])
// 		mat0[l] = make([][]float64, n/llen)
// 		mat0i[l] = make([][]float64, n/llen)

// 		// mat0si[l] = make([][]float64, n/llen)
// 		for t := range n / llen {
// 			mat0[l][t] = make([]float64, len(P)*llen*llen)
// 			mat0i[l][t] = make([]float64, len(P)*llen*llen)
// 			// mat0si[l][t] = make([]float64, len(params.Q())*llen*llen)
// 			stpoint := inter_it*int(t/inter) + (t % inter)
// 			for q := range len(P) {
// 				for i := range llen {
// 					for j := range llen {
// 						idx := q*llen*llen + i*llen + j
// 						if real(SFI[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
// 							mat0[l][t][idx] = float64(int64(real(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q]))
// 						} else {
// 							mat0[l][t][idx] = float64(int64(P[q]) - (int64(-real(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q])))
// 						}
// 						if imag(SFI[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
// 							mat0i[l][t][idx] = float64(int64(imag(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q]))
// 						} else {
// 							mat0i[l][t][idx] = float64(int64(P[q]) - (int64(-imag(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q])))
// 						}
// 						// mat0si[l][t][idx] = (mat0[l][t][idx] + mat0i[l][t][idx])
// 					}
// 				}
// 			}
// 		}

// 		inter_it = inter
// 	}
// 	fmt.Println("mat gen end")

// 	inputPolys := make([][]ring.Poly, 2)
// 	for i := range inputPolys {
// 		inputPolys[i] = make([]ring.Poly, sparseN)
// 		for j := range inputPolys[i] {
// 			inputPolys[i][j] = ringP.NewPoly()
// 		}
// 	}

// 	inputPolysC := make([][]ring.Poly, 2)
// 	for i := range inputPolysC {
// 		inputPolysC[i] = make([]ring.Poly, sparseN)
// 		for j := range inputPolysC[i] {
// 			inputPolysC[i][j] = ringP.NewPoly()
// 		}
// 	}
// 	result := make([]*rlwe.Ciphertext, N)
// 	result2 := make([]*rlwe.Ciphertext, N)
// 	pttemp := hefloat.NewPlaintext(params, params.MaxLevel())
// 	pttemp.IsBatched = false
// 	encoder.Encode(value, pttemp)
// 	fmt.Println("start ct res pre allocate")
// 	for i := range N {
// 		cttemp, _ := encryptor.EncryptNew(pttemp)
// 		result[i] = cttemp.CopyNew()
// 		result2[i] = cttemp.CopyNew()
// 	}
// 	fmt.Println("prealloc end")
// 	fmt.Println(result[0].LogScale())

// 	// sc := rlwe.NewScale(1)
// 	// for i := range 2 {
// 	// 	q := rlwe.NewScale(params.Q()[params.MaxLevel()-i])
// 	// 	sc = sc.Mul(q)
// 	// }

// 	ppmmbuffer1 := make([]float64, len(P)*N*N)
// 	ppmmbuffer2 := make([]float64, len(P)*N*N)

// 	resPolys00 := make([][]ring.Poly, 2)
// 	resPolys00i := make([][]ring.Poly, 2)
// 	resPolys01 := make([][]ring.Poly, 2)
// 	resPolys01i := make([][]ring.Poly, 2)
// 	resPolys10 := make([][]ring.Poly, 2)
// 	resPolys10i := make([][]ring.Poly, 2)
// 	resPolys11 := make([][]ring.Poly, 2)
// 	resPolys11i := make([][]ring.Poly, 2)
// 	for i := range 2 {
// 		resPolys00[i] = make([]ring.Poly, N)
// 		resPolys00i[i] = make([]ring.Poly, N)
// 		resPolys01[i] = make([]ring.Poly, N)
// 		resPolys01i[i] = make([]ring.Poly, N)
// 		resPolys10[i] = make([]ring.Poly, N)
// 		resPolys10i[i] = make([]ring.Poly, N)
// 		resPolys11[i] = make([]ring.Poly, N)
// 		resPolys11i[i] = make([]ring.Poly, N)
// 		for j := range N {
// 			resPolys00[i][j] = ringP.NewPoly()
// 			resPolys00i[i][j] = ringP.NewPoly()
// 			resPolys01[i][j] = ringP.NewPoly()
// 			resPolys01i[i][j] = ringP.NewPoly()
// 			resPolys10[i][j] = ringP.NewPoly()
// 			resPolys10i[i][j] = ringP.NewPoly()
// 			resPolys11[i][j] = ringP.NewPoly()
// 			resPolys11i[i][j] = ringP.NewPoly()
// 		}
// 	}
// 	work := make([]*rlwe.Ciphertext, N)
// 	for i := range work {
// 		work[i] = encryptor.EncryptZeroNew(params.MaxLevel())
// 	}
// 	aux := make([]*rlwe.Ciphertext, N)
// 	for i := range aux {
// 		aux[i] = encryptor.EncryptZeroNew(params.MaxLevel())
// 	}

// 	fmt.Println("start c2s")
// 	starttime = time.Now()
// 	util.PrintMemUsage()

// 	transpose.Transpose3(cts, params, evaluator, ringQ.AtLevel(cts[0].Level()), N, N, work, aux, cts)

// 	fmt.Println("mod switch start")

// 	for i := range sparseN {
// 		for d := range 2 {
// 			be.ModSwitchQtoP_Old(params.MaxLevel(), PLevel, cts[i].Value[d], inputPolys[d][i])
// 		}
// 		if i < n {
// 			ringQ.Neg(cts[i+n].Value[0], work[0].Value[0])
// 			ringQ.Neg(cts[i+n].Value[1], work[0].Value[1])
// 		} else {
// 			work[0] = cts[i-n]
// 		}
// 		for d := range 2 {
// 			be.ModSwitchQtoP_Old(params.MaxLevel(), PLevel, work[0].Value[d], inputPolysC[d][i])
// 		}
// 	}
// 	fmt.Println("mod switch end")
// 	fmt.Println("ppmm start")
// 	inter_it = n
// 	for l := range len(SFI) {
// 		inter := inter_it >> CL_arr[l]
// 		llen := (1 << CL_arr[l])
// 		if l == 0 {
// 			for t := range n / llen {
// 				//00 ~ 01
// 				// fmt.Println(inputPolys[0][0].Coeffs[0][:100])
// 				stpoint := (t % inter) + inter_it*int(t/inter)

// 				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00i, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01i, ppmmbuffer1, ppmmbuffer2)

// 				//10~11
// 				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10i, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11i, ppmmbuffer1, ppmmbuffer2)

// 			}

// 		} else if l == len(SFI)-1 {
// 			for t := range n / llen {
// 				stpoint := (t % inter) + inter_it*int(t/inter)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys00, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys00i, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00i, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys01, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys01i, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01i, ppmmbuffer1, ppmmbuffer2)
// 			}
// 			matmult.SubManyRing(ringP, resPolys00, resPolys00i, resPolys00)
// 			matmult.AddManyRing(ringP, resPolys01, resPolys01i, resPolys01i)

// 			for t := range n / llen {
// 				stpoint := (t % inter) + inter_it*int(t/inter)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys10, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys10i, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10i, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys11, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys11i, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11i, ppmmbuffer1, ppmmbuffer2)
// 			}
// 			matmult.SubManyRing(ringP, resPolys10i, resPolys10, resPolys10)
// 			matmult.AddManyRing(ringP, resPolys11, resPolys11i, resPolys11i)

// 		}
// 		inter_it = inter
// 	}
// 	matmult.AddManyRing(ringP, resPolys00, resPolys01i, resPolys00)
// 	matmult.AddManyRing(ringP, resPolys10, resPolys11i, resPolys10)

// 	rev := matmult.BitReversePermutationMatrix(n)
// 	matrev := make([]float64, len(P)*N*N)
// 	for p := range len(P) {
// 		for i := range N {
// 			for j := range N {
// 				if (i < n && j < n) || (i >= n && j >= n) {
// 					matrev[p*N*N+i*N+j] = (real(rev[i%n][j%n]))
// 				}
// 			}
// 		}
// 	}

// 	matmult.PPMM_Blas_CRT_Inplace(resPolys00, matrev, N, N, N, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 	matmult.PPMM_Blas_CRT_Inplace(resPolys10, matrev, N, N, N, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)

// 	for i := range N {
// 		for idx := range 2 {
// 			be.ModSwitchPtoQ_Old(PLevel, params.MaxLevel(), resPolys00[idx][i], result[i].Value[idx])
// 			be.ModSwitchPtoQ_Old(PLevel, params.MaxLevel(), resPolys10[idx][i], result2[i].Value[idx])
// 		}

// 		q := rlwe.NewScale(params.Q()[result[i].Level()])
// 		util.Mul_ScaleExact(evaluator, result[i], 1.0/(scale), result[i], q)
// 		util.Rescale_NonNTT(evaluator, result[i], result[i])
// 		util.Mul_ScaleExact(evaluator, result2[i], 1.0/(scale), result2[i], q)
// 		util.Rescale_NonNTT(evaluator, result2[i], result2[i])

// 		q = rlwe.NewScale(params.Q()[result[i].Level()])
// 		util.Mul_ScaleExact(evaluator, result[i], 1.0/(scale), result[i], q)
// 		util.Rescale_NonNTT(evaluator, result[i], result[i])
// 		util.Mul_ScaleExact(evaluator, result2[i], 1.0/(scale), result2[i], q)
// 		util.Rescale_NonNTT(evaluator, result2[i], result2[i])
// 	}
// 	fmt.Println(result[0].Level())

// 	transpose.Transpose3(result, params, evaluator, ringQ.AtLevel(result[0].Level()), N, sparseN, work, aux, result)
// 	transpose.Transpose3(result2, params, evaluator, ringQ.AtLevel(result2[0].Level()), N, sparseN, work, aux, result2)
// 	elapse = time.Since(starttime)
// 	fmt.Println(elapse)

// 	resvalue := make([]complex128, n)
// 	for i := range result {

// 		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[0], result[i].Value[0])
// 		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[1], result[i].Value[1])
// 		result[i].IsBatched = true
// 		dept := decryptor.DecryptNew(result[i])
// 		encoder.Decode(dept, resvalue)

// 		fmt.Println(resvalue)
// 		if i > 100 {
// 			break
// 		}
// 	}
// 	fmt.Println()
// 	fmt.Println()
// 	fmt.Println()
// 	for i := range result2 {

// 		ringQ.AtLevel(result2[i].Level()).NTT(result2[i].Value[0], result2[i].Value[0])
// 		ringQ.AtLevel(result2[i].Level()).NTT(result2[i].Value[1], result2[i].Value[1])
// 		result2[i].IsBatched = true
// 		dept := decryptor.DecryptNew(result2[i])
// 		encoder.Decode(dept, resvalue)

// 		fmt.Println(resvalue)
// 		if i > 100 {
// 			break
// 		}
// 	}
// 	fmt.Println(result2[0].LogScale())

// }

// func Test_CheckPrecTrans(t *testing.T) {
// 	runtime.GOMAXPROCS(runtime.NumCPU())
// 	//ckks parameter init
// 	SchemeParams := hefloat.ParametersLiteral{
// 		LogN:            5,
// 		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46},
// 		LogP:            []int{51},
// 		LogDefaultScale: 46,
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

// 	n := 1 << params.LogN()

// 	var pk *rlwe.PublicKey
// 	var rlk *rlwe.RelinearizationKey
// 	var rtk []*rlwe.GaloisKey

// 	fmt.Println("generated bootstrapper end")
// 	pk = kgen.GenPublicKeyNew(sk)
// 	rlk = kgen.GenRelinearizationKeyNew(sk)

// 	galLen := n
// 	fmt.Println("galLen : ", galLen)

// 	// generate keys - Rotating key
// 	galEls := make([]uint64, galLen)
// 	for i := range galEls {
// 		galEls[i] = uint64(2*i + 1)
// 	}
// 	galEls = append(galEls, params.GaloisElementForComplexConjugation())

// 	rtk = make([]*rlwe.GaloisKey, len(galEls))
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

// 	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
// 	//generate -er
// 	encryptor := rlwe.NewEncryptor(params, pk)
// 	decryptor := rlwe.NewDecryptor(params, sk)
// 	encoder := hefloat.NewEncoder(params)
// 	evaluator := hefloat.NewEvaluator(params, evk)

// 	fmt.Println("generate Evaluator end")
// 	runtime.GOMAXPROCS(1)

// 	_, _, _, _ = encoder, encryptor, decryptor, evaluator

// 	value := make([]float64, n)
// 	for i := range value {
// 		value[i] = sampling.RandFloat64(-1, 1)
// 	}
// 	pt := hefloat.NewPlaintext(params, params.MaxLevel())
// 	pt.IsBatched = false
// 	encoder.Encode(value, pt)

// 	cts := make([]*rlwe.Ciphertext, n)
// 	for i := range cts {
// 		cts[i], _ = encryptor.EncryptNew(pt)
// 	}

// 	res1 := transpose.Transpose(cts, params, evaluator, encoder, n)

// 	for i := range res1 {
// 		value := make([]float64, n)
// 		pt_ := decryptor.DecryptNew(res1[i])
// 		encoder.Decode(pt_, value)
// 		fmt.Println(value)
// 	}
// 	fmt.Println()
// 	fmt.Println()
// 	fmt.Println()

// 	for i := range cts {
// 		params.RingQ().INTT(cts[i].Value[0], cts[i].Value[0])
// 		params.RingQ().INTT(cts[i].Value[1], cts[i].Value[1])
// 	}

// 	work := make([]*rlwe.Ciphertext, n)
// 	for i := range work {
// 		work[i], _ = encryptor.EncryptNew(pt)
// 	}
// 	aux := make([]*rlwe.Ciphertext, n)
// 	for i := range aux {
// 		aux[i], _ = encryptor.EncryptNew(pt)
// 	}

// 	res2 := make([]*rlwe.Ciphertext, n)
// 	for i := range res2 {
// 		res2[i], _ = encryptor.EncryptNew(pt)
// 	}

// 	transpose.Transpose3(cts, params, evaluator, params.RingQ().AtLevel(cts[0].Level()), n, n, work, aux, cts)
// 	res2 = cts
// 	for i := range res2 {
// 		params.RingQ().NTT(res2[i].Value[0], res2[i].Value[0])
// 		params.RingQ().NTT(res2[i].Value[1], res2[i].Value[1])
// 	}

// 	for i := range res2 {
// 		value := make([]float64, n)
// 		pt_ := decryptor.DecryptNew(res2[i])
// 		encoder.Decode(pt_, value)
// 		fmt.Println(value)
// 	}
// }

// func Test_C2S_New_Check(t *testing.T) {

// 	//CPU full power
// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
// 	SchemeParams := hefloat.ParametersLiteral{
// 		LogN:            5,
// 		LogQ:            []int{48, 56, 56, 56},
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
// 	n := N / 2
// 	sparseN := N

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

// 	fmt.Println("ckks log degree : ", params.LogN())

// 	CL_arr := []int{2, 2}
// 	_, SFI := matmult.GenSFMat_CL(params, CL_arr, CL_arr)
// 	scale := float64(1 << 40)
// 	mat0 := make([][][]float64, len(SFI))
// 	mat0i := make([][][]float64, len(SFI))
// 	// mat0si := make([][][]float64, len(SFI))
// 	ringQ := params.RingQ()
// 	value := make([]float64, N)
// 	for i := range value {
// 		value[i] = 0.001
// 	}

// 	pt := hefloat.NewPlaintext(params, params.MaxLevel())
// 	pt.IsBatched = false

// 	encoder.Encode(value, pt)
// 	ct, _ := encryptor.EncryptNew(pt)
// 	cts := make([]*rlwe.Ciphertext, 2*n)
// 	for i := range cts {
// 		cts[i], _ = encryptor.EncryptNew(pt)
// 		ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
// 		ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
// 	}

// 	P := []uint64{4107427, 3868699, 4073143, 3639397, 3835109, 3377447, 3338903, 3314141, 3816173, 3731251, 3925091, 3500261, 3507403, 3368353, 3598601, 3637573, 3387523, 3489259, 3804751, 4002811, 3417251, 3245357, 3659177, 4047647, 3367981, 3984439, 3621473, 3565147, 3789193, 3174547, 3293959, 3567803, 3856499, 3299617, 3939619, 4004683, 3803347, 3501467, 3518719, 3631919}
// 	PLevel := 10
// 	P = P[:PLevel+1]
// 	ringP, _ := ring.NewRing(N, P)
// 	be := matmult.NewBasisExtender(ringQ, ringP, []matmult.Key{{params.MaxLevel(), PLevel}}, []matmult.Key{{PLevel, params.MaxLevel()}})

// 	fmt.Println("mat gen start")
// 	inter_it := n
// 	for l := range SFI {
// 		inter := inter_it >> CL_arr[l]
// 		llen := (1 << CL_arr[l])
// 		mat0[l] = make([][]float64, n/llen)
// 		mat0i[l] = make([][]float64, n/llen)

// 		// mat0si[l] = make([][]float64, n/llen)
// 		for t := range n / llen {
// 			mat0[l][t] = make([]float64, len(P)*llen*llen)
// 			mat0i[l][t] = make([]float64, len(P)*llen*llen)
// 			// mat0si[l][t] = make([]float64, len(params.Q())*llen*llen)
// 			stpoint := inter_it*int(t/inter) + (t % inter)
// 			for q := range len(P) {
// 				for i := range llen {
// 					for j := range llen {
// 						idx := q*llen*llen + i*llen + j
// 						if real(SFI[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
// 							mat0[l][t][idx] = float64(int64(real(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q]))
// 						} else {
// 							mat0[l][t][idx] = float64(int64(P[q]) - (int64(-real(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q])))
// 						}
// 						if imag(SFI[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
// 							mat0i[l][t][idx] = float64(int64(imag(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q]))
// 						} else {
// 							mat0i[l][t][idx] = float64(int64(P[q]) - (int64(-imag(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale) % int64(P[q])))
// 						}
// 						// mat0si[l][t][idx] = (mat0[l][t][idx] + mat0i[l][t][idx])
// 					}
// 				}
// 			}
// 		}

// 		inter_it = inter
// 	}
// 	fmt.Println("mat gen end")
// 	fmt.Println(P)
// 	fmt.Println(mat0)
// 	fmt.Println()
// 	fmt.Println(mat0i)

// 	inputPolys := make([][]ring.Poly, 2)
// 	for i := range inputPolys {
// 		inputPolys[i] = make([]ring.Poly, sparseN)
// 		for j := range inputPolys[i] {
// 			inputPolys[i][j] = ringP.NewPoly()
// 		}
// 	}

// 	inputPolysC := make([][]ring.Poly, 2)
// 	for i := range inputPolysC {
// 		inputPolysC[i] = make([]ring.Poly, sparseN)
// 		for j := range inputPolysC[i] {
// 			inputPolysC[i][j] = ringP.NewPoly()
// 		}
// 	}
// 	result := make([]*rlwe.Ciphertext, N)
// 	result2 := make([]*rlwe.Ciphertext, N)
// 	pttemp := hefloat.NewPlaintext(params, params.MaxLevel())
// 	pttemp.IsBatched = false
// 	encoder.Encode(value, pttemp)
// 	fmt.Println("start ct res pre allocate")
// 	for i := range N {
// 		cttemp, _ := encryptor.EncryptNew(pttemp)
// 		result[i] = cttemp.CopyNew()
// 		result2[i] = cttemp.CopyNew()
// 	}
// 	fmt.Println("prealloc end")
// 	fmt.Println(result[0].LogScale())

// 	// sc := rlwe.NewScale(1)
// 	// for i := range 2 {
// 	// 	q := rlwe.NewScale(params.Q()[params.MaxLevel()-i])
// 	// 	sc = sc.Mul(q)
// 	// }

// 	ppmmbuffer1 := make([]float64, len(P)*N*N)
// 	ppmmbuffer2 := make([]float64, len(P)*N*N)

// 	resPolys00 := make([][]ring.Poly, 2)
// 	resPolys00i := make([][]ring.Poly, 2)
// 	resPolys01 := make([][]ring.Poly, 2)
// 	resPolys01i := make([][]ring.Poly, 2)
// 	resPolys10 := make([][]ring.Poly, 2)
// 	resPolys10i := make([][]ring.Poly, 2)
// 	resPolys11 := make([][]ring.Poly, 2)
// 	resPolys11i := make([][]ring.Poly, 2)
// 	for i := range 2 {
// 		resPolys00[i] = make([]ring.Poly, N)
// 		resPolys00i[i] = make([]ring.Poly, N)
// 		resPolys01[i] = make([]ring.Poly, N)
// 		resPolys01i[i] = make([]ring.Poly, N)
// 		resPolys10[i] = make([]ring.Poly, N)
// 		resPolys10i[i] = make([]ring.Poly, N)
// 		resPolys11[i] = make([]ring.Poly, N)
// 		resPolys11i[i] = make([]ring.Poly, N)
// 		for j := range N {
// 			resPolys00[i][j] = ringP.NewPoly()
// 			resPolys00i[i][j] = ringP.NewPoly()
// 			resPolys01[i][j] = ringP.NewPoly()
// 			resPolys01i[i][j] = ringP.NewPoly()
// 			resPolys10[i][j] = ringP.NewPoly()
// 			resPolys10i[i][j] = ringP.NewPoly()
// 			resPolys11[i][j] = ringP.NewPoly()
// 			resPolys11i[i][j] = ringP.NewPoly()
// 		}
// 	}
// 	work := make([]*rlwe.Ciphertext, sparseN)
// 	for i := range work {
// 		work[i] = ct.CopyNew()
// 	}
// 	aux := make([]*rlwe.Ciphertext, sparseN)
// 	for i := range aux {
// 		aux[i] = ct.CopyNew()
// 	}

// 	fmt.Println("start c2s")
// 	starttime = time.Now()
// 	util.PrintMemUsage()
// 	transpose.Transpose3(cts, params, evaluator, ringQ.AtLevel(cts[0].Level()), N, sparseN, work, aux, cts)
// 	fmt.Println("mod switch start")

// 	for i := range sparseN {
// 		for d := range 2 {
// 			be.ModSwitchQtoP_Old(params.MaxLevel(), PLevel, cts[i].Value[d], inputPolys[d][i])
// 		}
// 		if i < n {
// 			ringQ.NTT(cts[i+n].Value[0], work[0].Value[0])
// 			ringQ.NTT(cts[i+n].Value[1], work[0].Value[1])
// 			evaluator.Mul(work[0], -1, work[0])
// 			ringQ.INTT(work[0].Value[0], work[0].Value[0])
// 			ringQ.INTT(work[0].Value[1], work[0].Value[1])
// 		} else {
// 			work[0] = cts[i-n]
// 		}
// 		for d := range 2 {
// 			be.ModSwitchQtoP_Old(params.MaxLevel(), PLevel, work[0].Value[d], inputPolysC[d][i])
// 		}

// 	}

// 	fmt.Println("mod switch end")
// 	fmt.Println("ppmm start")
// 	inter_it = n
// 	for l := range len(SFI) {
// 		inter := inter_it >> CL_arr[l]
// 		llen := (1 << CL_arr[l])
// 		if l == 0 {
// 			for t := range n / llen {
// 				stpoint := (t % inter) + inter_it*int(t/inter)
// 				rings_temp := make([][]ring.Poly, 2)
// 				for d := range rings_temp {
// 					rings_temp[d] = make([]ring.Poly, llen)
// 				}
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = inputPolys[d][stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys00[d][stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}

// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = inputPolys[d][stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0i[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys00i[d][stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}

// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = inputPolysC[d][n+stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys01[d][n+stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}

// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = inputPolysC[d][n+stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0i[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys01i[d][n+stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}

// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = inputPolysC[d][stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys10[d][stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}

// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = inputPolysC[d][stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0i[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys10i[d][stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}

// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = inputPolys[d][n+stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys11[d][n+stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}

// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = inputPolys[d][n+stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0i[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys11i[d][n+stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}
// 			}

// 		} else if l == len(SFI)-1 {
// 			for t := range n / llen {
// 				stpoint := (t % inter) + inter_it*int(t/inter)
// 				rings_temp := make([][]ring.Poly, 2)
// 				for d := range rings_temp {
// 					rings_temp[d] = make([]ring.Poly, llen)
// 				}

// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = resPolys00[d][stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys00[d][stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}

// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = resPolys00i[d][stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0i[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys00i[d][stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}

// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = resPolys01[d][n+stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0i[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys01[d][n+stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}

// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = resPolys01i[d][n+stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys01i[d][n+stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}
// 			}
// 			matmult.SubManyRing(ringP, resPolys00, resPolys00i, resPolys00)
// 			matmult.AddManyRing(ringP, resPolys01, resPolys01i, resPolys01i)

// 			for t := range n / llen {
// 				stpoint := (t % inter) + inter_it*int(t/inter)
// 				rings_temp := make([][]ring.Poly, 2)
// 				for d := range rings_temp {
// 					rings_temp[d] = make([]ring.Poly, llen)
// 				}

// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = resPolys10[d][stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys10[d][stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}

// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = resPolys10i[d][stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0i[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys10i[d][stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}

// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = resPolys11[d][n+stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0i[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys11[d][n+stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}

// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						rings_temp[d][idx_ll] = resPolys11i[d][n+stpoint+inter*idx_ll]
// 					}
// 				}
// 				matmult.PPMM_Blas_CRT_Inplace(rings_temp, mat0[l][t], llen, llen, 2*n, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 				for d := range 2 {
// 					for idx_ll := range llen {
// 						resPolys11i[d][n+stpoint+inter*idx_ll] = rings_temp[d][idx_ll]
// 					}
// 				}
// 			}
// 			matmult.SubManyRing(ringP, resPolys10, resPolys10i, resPolys10)
// 			matmult.AddManyRing(ringP, resPolys11, resPolys11i, resPolys11i)

// 		}
// 		inter_it = inter
// 	}
// 	matmult.AddManyRing(ringP, resPolys00, resPolys01i, resPolys00)
// 	matmult.AddManyRing(ringP, resPolys10, resPolys11i, resPolys10)

// 	rev := matmult.BitReversePermutationMatrix(n)
// 	matrev := make([]float64, len(P)*N*N)
// 	for p := range len(P) {
// 		for i := range N {
// 			for j := range N {
// 				if (i < n && j < n) || (i >= n && j >= n) {
// 					matrev[p*N*N+i*N+j] = (real(rev[i%n][j%n]))
// 				}
// 			}
// 		}
// 	}

// 	matmult.PPMM_Blas_CRT_Inplace(resPolys00, matrev, N, N, N, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 	matmult.PPMM_Blas_CRT_Inplace(resPolys10, matrev, N, N, N, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)

// 	for i := range N {
// 		for idx := range 2 {
// 			be.ModSwitchPtoQ_Old(PLevel, params.MaxLevel(), resPolys00[idx][i], result[i].Value[idx])
// 			be.ModSwitchPtoQ_Old(PLevel, params.MaxLevel(), resPolys10[idx][i], result2[i].Value[idx])
// 		}
// 		ringQ.NTT(result[i].Value[0], result[i].Value[0])
// 		ringQ.NTT(result[i].Value[1], result[i].Value[1])
// 		ringQ.NTT(result2[i].Value[0], result2[i].Value[0])
// 		ringQ.NTT(result2[i].Value[1], result2[i].Value[1])

// 		evaluator.Mul(result[i], 1.0/(scale), result[i])
// 		evaluator.Rescale(result[i], result[i])
// 		evaluator.Mul(result[i], 1.0/(scale), result[i])
// 		evaluator.Rescale(result[i], result[i])

// 		evaluator.Mul(result2[i], 1.0/(scale), result2[i])
// 		evaluator.Rescale(result2[i], result2[i])
// 		evaluator.Mul(result2[i], 1.0/(scale), result2[i])
// 		evaluator.Rescale(result2[i], result2[i])

// 		ringQ.AtLevel(result[i].Level()).INTT(result[i].Value[0], result[i].Value[0])
// 		ringQ.AtLevel(result[i].Level()).INTT(result[i].Value[1], result[i].Value[1])
// 		ringQ.AtLevel(result[i].Level()).INTT(result2[i].Value[0], result2[i].Value[0])
// 		ringQ.AtLevel(result[i].Level()).INTT(result2[i].Value[1], result2[i].Value[1])

// 		// q := rlwe.NewScale(params.Q()[result[i].Level()])
// 		// util.Mul_ScaleExact(evaluator, result[i], 1.0/(scale), result[i], q)
// 		// util.Rescale_NonNTT(evaluator, result[i], result[i])
// 		// util.Mul_ScaleExact(evaluator, result2[i], 1.0/(scale), result2[i], q)
// 		// util.Rescale_NonNTT(evaluator, result2[i], result2[i])

// 		// q = rlwe.NewScale(params.Q()[result[i].Level()])
// 		// util.Mul_ScaleExact(evaluator, result[i], 1.0/(scale), result[i], q)
// 		// util.Rescale_NonNTT(evaluator, result[i], result[i])
// 		// util.Mul_ScaleExact(evaluator, result2[i], 1.0/(scale), result2[i], q)
// 		// util.Rescale_NonNTT(evaluator, result2[i], result2[i])
// 	}
// 	fmt.Println(result[0].Level())

// 	transpose.Transpose3(result, params, evaluator, ringQ.AtLevel(result[0].Level()), N, sparseN, work, aux, result)
// 	transpose.Transpose3(result2, params, evaluator, ringQ.AtLevel(result2[0].Level()), N, sparseN, work, aux, result2)
// 	elapse = time.Since(starttime)
// 	fmt.Println(elapse)

// 	resvalue := make([]complex128, n)
// 	for i := range result {

// 		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[0], result[i].Value[0])
// 		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[1], result[i].Value[1])
// 		result[i].IsBatched = true
// 		dept := decryptor.DecryptNew(result[i])
// 		encoder.Decode(dept, resvalue)

// 		fmt.Println(resvalue[0:10])
// 		if i > 100 {
// 			break
// 		}
// 	}
// 	fmt.Println()
// 	fmt.Println()
// 	fmt.Println()
// 	for i := range result2 {

// 		ringQ.AtLevel(result2[i].Level()).NTT(result2[i].Value[0], result2[i].Value[0])
// 		ringQ.AtLevel(result2[i].Level()).NTT(result2[i].Value[1], result2[i].Value[1])
// 		result2[i].IsBatched = true
// 		dept := decryptor.DecryptNew(result2[i])
// 		encoder.Decode(dept, resvalue)

// 		fmt.Println(resvalue[0:10])
// 		if i > 100 {
// 			break
// 		}
// 	}
// 	fmt.Println(result2[0].LogScale())

// }
