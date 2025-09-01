package test

import (
	"fmt"
	"math/cmplx"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/lifejade/mm/src/matmult"
	"github.com/lifejade/mm/src/transpose"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/schemes/ckks"
)

func Test_FFT(t *testing.T) {

	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	logn := 5
	n := 1 << logn

	roots := ckks.GetRootsBigComplex(n<<2, 53)
	roots_complex := make([]complex128, 4*n)

	for i := range roots_complex {
		roots_complex[i] = roots[i].Complex128()
	}
	fmt.Println()

	pow5 := make([]int, (n<<1)+1)
	pow5[0] = 1
	for i := 1; i < (n<<1)+1; i++ {
		pow5[i] = pow5[i-1] * 5
		pow5[i] &= (n << 2) - 1
	}

	SF := make([][]complex128, n)

	for i := range SF {
		SF[i] = make([]complex128, n)
		for j := range SF[i] {
			idx := (pow5[i] * j) & ((n << 2) - 1)
			SF[i][j] = roots_complex[idx]
		}
	}

	SF_ := make([][][]complex128, logn)
	for idx := 0; idx < logn; idx++ {
		SF_[idx] = make([][]complex128, n)
		for i := range n {
			SF_[idx][i] = make([]complex128, n)
		}

		m := 1 << (idx + 1)
		for i := 0; i < n; i += m {
			pow5v := 1
			for j := 0; j < (m >> 1); j++ {
				k := pow5v * n / m

				SF_[idx][i+j][i+j] = 1
				SF_[idx][i+j][i+j+(m>>1)] = roots_complex[k]

				SF_[idx][i+j+(m>>1)][i+j] = 1
				SF_[idx][i+j+(m>>1)][i+j+(m>>1)] = -roots_complex[k]

				pow5v *= 5
				pow5v = pow5v & ((m << 2) - 1)
			}
		}
	}

	var SF__ [][]complex128
	SF__ = matmult.BitReversePermutationMatrix(n)
	// SF__ = make([][]complex128, n)
	// for i := range SF__ {
	// 	SF__[i] = make([]complex128, n)
	// 	SF__[i][i] = 1
	// }

	for i := range SF_ {
		SF__ = mul(SF_[i], SF__)
	}

	fmt.Println("is same ? : ", closeMat(SF, SF__, 0.00001))

	fmt.Println("origin SF")
	for i := range SF {
		fmt.Println(SF[i])
	}

	fmt.Println("/////////////////////////////////")
	fmt.Println("new SF")
	for i := range SF__ {
		fmt.Println(SF__[i])
	}

	fmt.Println("/////////////////////////////////")

	for i := range SF_ {
		for j := range SF_[i] {
			fmt.Println(SF_[i][j])
		}
		fmt.Println("")
	}

	// fmt.Println("//////////")
	// test := mul(SF_[0], matmult.BitReversePermutationMatrix(n))
	// for i := range test {
	// 	fmt.Println(test[i])
	// }
}

func Test_FFT_CL(t *testing.T) {

	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	logn := 10
	n := 1 << logn

	roots := ckks.GetRootsBigComplex(n<<2, 53)
	roots_complex := make([]complex128, 4*n)

	for i := range roots_complex {
		roots_complex[i] = roots[i].Complex128()
	}
	fmt.Println()

	pow5 := make([]int, (n<<1)+1)
	pow5[0] = 1
	for i := 1; i < (n<<1)+1; i++ {
		pow5[i] = pow5[i-1] * 5
		pow5[i] &= (n << 2) - 1
	}

	SF := make([][]complex128, n)

	for i := range SF {
		SF[i] = make([]complex128, n)
		for j := range SF[i] {
			idx := (pow5[i] * j) & ((n << 2) - 1)
			SF[i][j] = roots_complex[idx]
		}
	}

	SF_ := make([][][]complex128, logn)
	for idx := 0; idx < logn; idx++ {
		SF_[idx] = make([][]complex128, n)
		for i := range n {
			SF_[idx][i] = make([]complex128, n)
		}

		m := 1 << (idx + 1)
		for i := 0; i < n; i += m {
			pow5v := 1
			for j := 0; j < (m >> 1); j++ {
				k := pow5v * n / m

				SF_[idx][i+j][i+j] = 1
				SF_[idx][i+j][i+j+(m>>1)] = roots_complex[k]

				SF_[idx][i+j+(m>>1)][i+j] = 1
				SF_[idx][i+j+(m>>1)][i+j+(m>>1)] = -roots_complex[k]

				pow5v *= 5
				pow5v = pow5v & ((m << 2) - 1)
			}
		}
	}

	var SF__ [][]complex128
	SF__ = matmult.BitReversePermutationMatrix(n)
	// SF__ = make([][]complex128, n)
	// for i := range SF__ {
	// 	SF__[i] = make([]complex128, n)
	// 	SF__[i][i] = 1
	// }

	for i := range SF_ {
		SF__ = mul(SF_[i], SF__)
	}

	fmt.Println("is same ? : ", closeMat(SF, SF__, 0.00001))

	fmt.Println("origin SF")
	for i := range SF {
		fmt.Println(SF[i])
	}

	fmt.Println("/////////////////////////////////")
	fmt.Println("new SF")
	for i := range SF__ {
		fmt.Println(SF__[i])
	}

	fmt.Println("/////////////////////////////////")

	l := 2
	pl := logn / l
	SF_CL := make([][][]complex128, l)
	for i := range SF_CL {
		SF_CL[i] = SF_[i*pl]
		for j := range pl {
			if j == 0 {
				continue
			}
			SF_CL[i] = mul(SF_[i*pl+j], SF_CL[i])
		}
	}

	fmt.Println("new SF_CL")
	for i := range SF_CL {
		for j := range SF_CL[i] {
			fmt.Println(SF_CL[i][j])
		}
		fmt.Println()
	}

	fmt.Println("check")
	SF_CL_ := matmult.BitReversePermutationMatrix(n)
	for i := range SF_CL {
		SF_CL_ = mul(SF_CL[i], SF_CL_)
	}
	fmt.Println("is same ? : ", closeMat(SF, SF_CL_, 0.00001))
	for i := range SF_CL_ {
		fmt.Println(SF_CL_[i])
	}
}

func Test_IFFT(t *testing.T) {

	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	logn := 2
	n := 1 << logn

	roots := ckks.GetRootsBigComplex(n<<2, 53)
	roots_complex := make([]complex128, 4*n)

	for i := range roots_complex {
		roots_complex[i] = roots[i].Complex128()
	}
	fmt.Println()

	pow5 := make([]int, (n<<1)+1)
	pow5[0] = 1
	for i := 1; i < (n<<1)+1; i++ {
		pow5[i] = pow5[i-1] * 5
		pow5[i] &= (n << 2) - 1
	}

	SF := make([][]complex128, n)

	for i := range SF {
		SF[i] = make([]complex128, n)
		for j := range SF[i] {
			idx := (pow5[i] * j) & ((n << 2) - 1)
			SF[i][j] = roots_complex[idx]
		}
	}

	SFI := make([][]complex128, n)
	for i := range SFI {
		SFI[i] = make([]complex128, n)
	}
	for i := range SFI {
		for j := range SFI[i] {
			idx := (pow5[i] * j) & ((n << 2) - 1)
			SFI[j][i] = cmplx.Conj(roots_complex[idx]) / complex((float64(n)), 0)
		}
	}

	SFI_ := make([][][]complex128, logn)
	for idx := 0; idx < logn; idx++ {
		SFI_[idx] = make([][]complex128, n)
		for i := range n {
			SFI_[idx][i] = make([]complex128, n)
		}

		m := n >> (idx)
		for i := 0; i < n; i += m {
			pow5v := 1
			for j := 0; j < (m >> 1); j++ {
				k := pow5v * n / m

				SFI_[idx][i+j][i+j] = 1
				SFI_[idx][i+j][i+j+(m>>1)] = 1

				SFI_[idx][i+j+(m>>1)][i+j] = cmplx.Conj(roots_complex[k])
				SFI_[idx][i+j+(m>>1)][i+j+(m>>1)] = -cmplx.Conj(roots_complex[k])

				pow5v *= 5
				pow5v = pow5v & ((m << 2) - 1)
			}
		}
	}

	var SFI__ [][]complex128
	SFI__ = matmult.BitReversePermutationMatrix(n)
	for i := range SFI__ {
		for j := range SFI__ {
			SFI__[i][j] /= complex(float64(n), 0)
		}
	}
	// SF__ = make([][]complex128, n)
	// for i := range SF__ {
	// 	SF__[i] = make([]complex128, n)
	// 	SF__[i][i] = 1
	// }

	for i := range SFI_ {
		SFI__ = mul(SFI__, SFI_[logn-i-1])
	}

	fmt.Println("is same ? : ", closeMat(SFI, SFI__, 0.0001))

	fmt.Println("origin SF")
	for i := range SFI {
		fmt.Println(SFI[i])
	}

	fmt.Println("/////////////////////////////////")
	fmt.Println("new SF")
	for i := range SFI__ {
		fmt.Println(SFI__[i])
	}

	fmt.Println("/////////////////////////////////")

	// l := 2
	// pl := logn / l
	// SF_CL := make([][][]complex128, l)
	// for i := range SF_CL {
	// 	SF_CL[i] = SF_[i*pl]
	// 	for j := range pl {
	// 		if j == 0 {
	// 			continue
	// 		}
	// 		SF_CL[i] = mul(SF_[i*pl+j], SF_CL[i])
	// 	}
	// }

	// fmt.Println("new SF_CL")
	// for i := range SF_CL {
	// 	for j := range SF_CL[i] {
	// 		fmt.Println(SF_CL[i][j])
	// 	}
	// 	fmt.Println()
	// }

	// fmt.Println("check")
	// SF_CL_ := matmult.BitReversePermutationMatrix(n)
	// for i := range SF_CL {
	// 	SF_CL_ = mul(SF_CL[i], SF_CL_)
	// }
	// fmt.Println("is same ? : ", closeMat(SF, SF_CL_, 0.00001))
	// for i := range SF_CL_ {
	// 	fmt.Println(SF_CL_[i])
	// }
}

func Test_SFLC(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            5,
		LogQ:            []int{50, 50, 50, 32, 32},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}
	//parameter init
	params, err := hefloat.NewParametersFromLiteral(SchemeParams)
	if err != nil {
		panic(err)
	}

	SF_LC, SFI_LC := matmult.GenSFMat_CL(params, []int{2, 2}, []int{2, 2})

	n := params.MaxSlots()

	pow5 := make([]int, (n<<1)+1)
	pow5[0] = 1
	for i := 1; i < (n<<1)+1; i++ {
		pow5[i] = pow5[i-1] * 5
		pow5[i] &= (n << 2) - 1
	}
	roots := ckks.GetRootsBigComplex(n<<2, params.EncodingPrecision())
	roots_complex := make([]complex128, 4*n)

	for i := range roots_complex {
		roots_complex[i] = roots[i].Complex128()
	}

	SF := make([][]complex128, n)

	for i := range SF {
		SF[i] = make([]complex128, n)
		for j := range SF[i] {
			idx := (pow5[i] * j) & ((n << 2) - 1)
			SF[i][j] = roots_complex[idx]
		}
	}

	SFI := make([][]complex128, n)
	for i := range SFI {
		SFI[i] = make([]complex128, n)
	}
	for i := range SFI {
		for j := range SFI[i] {
			idx := (pow5[i] * j) & ((n << 2) - 1)
			SFI[j][i] = cmplx.Conj(roots_complex[idx]) / complex((float64(n)), 0)
		}
	}

	SF_ := matmult.BitReversePermutationMatrix(n)
	for i := range SF_LC {
		SF_ = mul(SF_LC[i], SF_)
	}

	SFI_ := matmult.BitReversePermutationMatrix(n)
	// for i := range SFI_ {
	// 	for j := range SFI_[i] {
	// 		SFI_[i][j] /= complex(float64(n), 0)
	// 	}
	// }
	for i := range SFI_LC {
		SFI_ = mul(SFI_, SFI_LC[len(SFI_LC)-i-1])
	}
	for i := range SFI {
		fmt.Println(SFI[i])
	}
	for i := range SFI_ {
		fmt.Println(SFI_[i])
	}
	fmt.Println(len(SF_LC), len(SFI_LC))
	fmt.Println("is same ? : ", closeMat(SF, SF_, 0.0001))
	fmt.Println("is same ? : ", closeMat(SFI, SFI_, 0.0001))

}

func Test_C2SLC(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            5,
		LogQ:            []int{48, 40, 40, 48},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}
	//parameter init
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

	n := 1 << params.LogMaxSlots()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

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
	// btpevk, _, _ := btpParams.GenEvaluationKeys(sk)

	// btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	// if err != nil {
	// 	panic(err)
	// }
	fmt.Println("generate Evaluator end")

	_, SFI := matmult.GenSFMat_CL(params, []int{2, 2}, []int{2, 2})
	scale := float64(1 << 40)
	mat0 := make([][][][]uint64, len(SFI))
	mat1 := make([][][][]uint64, len(SFI))
	mat0i := make([][][][]uint64, len(SFI))
	mat1i := make([][][][]uint64, len(SFI))

	for l := range SFI {
		mat0[l] = make([][][]uint64, len(params.Q()))
		mat1[l] = make([][][]uint64, len(params.Q()))
		mat0i[l] = make([][][]uint64, len(params.Q()))
		mat1i[l] = make([][][]uint64, len(params.Q()))

		for q := range mat0[l] {
			mat0[l][q] = make([][]uint64, 2*n)
			mat1[l][q] = make([][]uint64, 2*n)
			mat0i[l][q] = make([][]uint64, 2*n)
			mat1i[l][q] = make([][]uint64, 2*n)
			for i := range 2 * n {
				mat0[l][q][i] = make([]uint64, 2*n)
				mat1[l][q][i] = make([]uint64, 2*n)
				mat0i[l][q][i] = make([]uint64, 2*n)
				mat1i[l][q][i] = make([]uint64, 2*n)
				for j := range 2 * n {
					if i < n && j < n {
						if real(SFI[l][i][j]) >= 0 {
							mat0[l][q][i][j] = uint64(real(SFI[l][i][j]) * scale)
						} else {
							mat0[l][q][i][j] = params.Q()[q] - uint64(-real(SFI[l][i][j])*scale)
						}
						if imag(SFI[l][i][j]) >= 0 {
							mat0i[l][q][i][j] = uint64(imag(SFI[l][i][j]) * scale)
						} else {
							mat0i[l][q][i][j] = params.Q()[q] - uint64(-imag(SFI[l][i][j])*scale)
						}
					}
					if i >= n && j >= n {
						if real(SFI[l][i%n][j%n]) >= 0 {
							mat1[l][q][i][j] = uint64(real(SFI[l][i%n][j%n]) * scale)
						} else {
							mat1[l][q][i][j] = params.Q()[q] - uint64(-real(SFI[l][i%n][j%n])*scale)
						}
						if imag(SFI[l][i%n][j%n]) >= 0 {
							mat1i[l][q][i][j] = uint64(imag(SFI[l][i%n][j%n]) * scale)
						} else {
							mat1i[l][q][i][j] = params.Q()[q] - uint64(-imag(SFI[l][i%n][j%n])*scale)
						}

					}
				}
			}
		}
	}

	for l := range mat0 {
		fmt.Println("l is : ", l)
		for i := range mat0[l][0] {
			fmt.Println(mat0[l][0][i])
		}
		fmt.Println()
		fmt.Println()
		for i := range mat0[l][0] {
			fmt.Println(mat0i[l][0][i])
		}
		fmt.Println()
	}

	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	cts := make([]*rlwe.Ciphertext, 2*n)
	for i := range cts {
		cts[i] = ct.CopyNew()
	}

	fmt.Println("start c2s")
	starttime = time.Now()
	ctT := transpose.Transpose(cts, params, evaluator, encoder, 2*n)
	// rev := matmult.BitReversePermutationMatrix(n)
	// matrev := make([][]uint64, 2*n)
	// for i := range matrev {
	// 	matrev[i] = make([]uint64, 2*n)
	// 	for j := range matrev[i] {
	// 		if (i < n && j < n) || (i >= n && j >= n) {
	// 			matrev[i][j] = uint64(real(rev[i%n][j%n]))
	// 		}
	// 	}
	// }
	// ctT = matmult.PPMM_Flint(ctT, matrev, params, 2*n)

	ctTC := make([]*rlwe.Ciphertext, 2*n)
	fmt.Println("ctT ctTC")
	for i := range ctTC {
		if i < n {
			ctTC[i], _ = evaluator.MulNew(ctT[i+n], -1)
		} else {
			ctTC[i] = ctT[i-n].CopyNew()
		}
	}
	// for l := range mat0 {
	// 	for q := range mat0[l] {
	// 		for i := range mat0[l][q] {
	// 			fmt.Println(mat0i[l][q][i])
	// 		}
	// 		fmt.Println()
	// 	}
	// }

	var res00, res01, res10, res11, res00i, res01i, res10i, res11i []*rlwe.Ciphertext
	for l := range len(SFI) {
		if l == 0 {
			res00 = matmult.PPMM_Flint_CRT(ctT, mat0[l], params, 2*n)
			res01 = matmult.PPMM_Flint_CRT(ctTC, mat1[l], params, 2*n)
			res00i = matmult.PPMM_Flint_CRT(ctT, mat0i[l], params, 2*n)
			res01i = matmult.PPMM_Flint_CRT(ctTC, mat1i[l], params, 2*n)

			res10 = matmult.PPMM_Flint_CRT(ctTC, mat0[l], params, 2*n)
			res11 = matmult.PPMM_Flint_CRT(ctT, mat1[l], params, 2*n)
			res10i = matmult.PPMM_Flint_CRT(ctTC, mat0i[l], params, 2*n)
			res11i = matmult.PPMM_Flint_CRT(ctT, mat1i[l], params, 2*n)

			for i := range 2 * n {
				evaluator.Mul(res00[i], 1.0/scale, res00[i])
				evaluator.Rescale(res00[i], res00[i])
				evaluator.Mul(res00i[i], 1.0/scale, res00i[i])
				evaluator.Rescale(res00i[i], res00i[i])

				evaluator.Mul(res01[i], 1.0/scale, res01[i])
				evaluator.Rescale(res01[i], res01[i])
				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
				evaluator.Rescale(res01i[i], res01i[i])

				evaluator.Mul(res10[i], 1.0/scale, res10[i])
				evaluator.Rescale(res10[i], res10[i])
				evaluator.Mul(res10i[i], 1.0/scale, res10i[i])
				evaluator.Rescale(res10i[i], res10i[i])

				evaluator.Mul(res11[i], 1.0/scale, res11[i])
				evaluator.Rescale(res11[i], res11[i])
				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
				evaluator.Rescale(res11i[i], res11i[i])
			}

		} else if l == len(SFI)-1 {
			temp1 := matmult.PPMM_Flint_CRT(res00, mat0[l], params, 2*n)
			temp1_ := matmult.PPMM_Flint_CRT(res00i, mat0i[l], params, 2*n)
			temp2 := matmult.PPMM_Flint_CRT(res01, mat1i[l], params, 2*n)
			temp2_ := matmult.PPMM_Flint_CRT(res01i, mat1[l], params, 2*n)
			for i := range 2 * n {
				evaluator.Sub(temp1[i], temp1_[i], res00[i])
				evaluator.Add(temp2[i], temp2_[i], res01i[i])
			}

			temp1 = matmult.PPMM_Flint_CRT(res10, mat0[l], params, 2*n)
			temp1_ = matmult.PPMM_Flint_CRT(res10i, mat0i[l], params, 2*n)
			temp2 = matmult.PPMM_Flint_CRT(res11, mat1i[l], params, 2*n)
			temp2_ = matmult.PPMM_Flint_CRT(res11i, mat1[l], params, 2*n)
			for i := range 2 * n {
				evaluator.Sub(temp1[i], temp1_[i], res10[i])
				evaluator.Add(temp2[i], temp2_[i], res11i[i])
			}
			for i := range 2 * n {
				evaluator.Mul(res00[i], 1.0/scale, res00[i])
				evaluator.Rescale(res00[i], res00[i])

				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
				evaluator.Rescale(res01i[i], res01i[i])

				evaluator.Mul(res10[i], 1.0/scale, res10[i])
				evaluator.Rescale(res10[i], res10[i])

				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
				evaluator.Rescale(res11i[i], res11i[i])
			}
		} else {
			temp1 := matmult.PPMM_Flint_CRT(res00, mat0[l], params, 2*n)
			temp1_ := matmult.PPMM_Flint_CRT(res00i, mat0i[l], params, 2*n)
			temp2 := matmult.PPMM_Flint_CRT(res00, mat0i[l], params, 2*n)
			temp2_ := matmult.PPMM_Flint_CRT(res00i, mat0[l], params, 2*n)
			for i := range 2 * n {
				evaluator.Sub(temp1[i], temp1_[i], res00[i])
				evaluator.Add(temp2[i], temp2_[i], res00i[i])
			}

			temp1 = matmult.PPMM_Flint_CRT(res01, mat1[l], params, 2*n)
			temp1_ = matmult.PPMM_Flint_CRT(res01i, mat1i[l], params, 2*n)
			temp2 = matmult.PPMM_Flint_CRT(res01, mat1i[l], params, 2*n)
			temp2_ = matmult.PPMM_Flint_CRT(res01i, mat1[l], params, 2*n)
			for i := range 2 * n {
				evaluator.Sub(temp1[i], temp1_[i], res01[i])
				evaluator.Add(temp2[i], temp2_[i], res01i[i])
			}

			temp1 = matmult.PPMM_Flint_CRT(res10, mat0[l], params, 2*n)
			temp1_ = matmult.PPMM_Flint_CRT(res10i, mat0i[l], params, 2*n)
			temp2 = matmult.PPMM_Flint_CRT(res10, mat0i[l], params, 2*n)
			temp2_ = matmult.PPMM_Flint_CRT(res10i, mat0[l], params, 2*n)
			for i := range 2 * n {
				evaluator.Sub(temp1[i], temp1_[i], res10[i])
				evaluator.Add(temp2[i], temp2_[i], res10i[i])
			}

			temp1 = matmult.PPMM_Flint_CRT(res11, mat1[l], params, 2*n)
			temp1_ = matmult.PPMM_Flint_CRT(res11i, mat1i[l], params, 2*n)
			temp2 = matmult.PPMM_Flint_CRT(res11, mat1i[l], params, 2*n)
			temp2_ = matmult.PPMM_Flint_CRT(res11i, mat1[l], params, 2*n)
			for i := range 2 * n {
				evaluator.Sub(temp1[i], temp1_[i], res11[i])
				evaluator.Add(temp2[i], temp2_[i], res11i[i])
			}

			for i := range 2 * n {
				evaluator.Mul(res00[i], 1.0/scale, res00[i])
				evaluator.Rescale(res00[i], res00[i])
				evaluator.Mul(res00i[i], 1.0/scale, res00i[i])
				evaluator.Rescale(res00i[i], res00i[i])

				evaluator.Mul(res01[i], 1.0/scale, res01[i])
				evaluator.Rescale(res01[i], res01[i])
				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
				evaluator.Rescale(res01i[i], res01i[i])

				evaluator.Mul(res10[i], 1.0/scale, res10[i])
				evaluator.Rescale(res10[i], res10[i])
				evaluator.Mul(res10i[i], 1.0/scale, res10i[i])
				evaluator.Rescale(res10i[i], res10i[i])

				evaluator.Mul(res11[i], 1.0/scale, res11[i])
				evaluator.Rescale(res11[i], res11[i])
				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
				evaluator.Rescale(res11i[i], res11i[i])
			}
		}
	}

	res0 := make([]*rlwe.Ciphertext, 2*n)
	for i := range res0 {
		res0[i], _ = evaluator.AddNew(res00[i], res01i[i])
	}
	res1 := make([]*rlwe.Ciphertext, 2*n)
	for i := range res1 {
		evaluator.Mul(res10[i], -1, res10[i])
		res1[i], _ = evaluator.AddNew(res10[i], res11i[i])
	}

	rev := matmult.BitReversePermutationMatrix(n)
	matrev := make([][]uint64, 2*n)
	for i := range matrev {
		matrev[i] = make([]uint64, 2*n)
		for j := range matrev[i] {
			if (i < n && j < n) || (i >= n && j >= n) {
				matrev[i][j] = uint64(real(rev[i%n][j%n]))
			}
		}
	}
	res0 = matmult.PPMM_Flint(res0, matrev, params, 2*n)
	res1 = matmult.PPMM_Flint(res1, matrev, params, 2*n)

	result0 := transpose.Transpose(res0, params, evaluator, encoder, 2*n)
	result1 := transpose.Transpose(res1, params, evaluator, encoder, 2*n)

	elapse = time.Since(starttime)
	fmt.Println(elapse)
	fmt.Println(result0[0].LogScale())
	fmt.Println(result1[0].LogScale())

	resvalue := make([]complex128, n)
	for i := range result0 {
		result0[i].IsBatched = true
		dept := decryptor.DecryptNew(result0[i])
		encoder.Decode(dept, resvalue)

		fmt.Println(resvalue)
	}
	fmt.Println()
	fmt.Println()
	fmt.Println()
	for i := range result1 {
		result1[i].IsBatched = true
		dept := decryptor.DecryptNew(result1[i])
		encoder.Decode(dept, resvalue)

		fmt.Println(resvalue)
	}
	fmt.Println(result0[0].LogScale())

}

func Test_S2CLC(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            5,
		LogQ:            []int{48, 40, 40, 48, 48, 48, 48, 48, 48, 48, 48, 40, 40},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}
	//parameter init
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

	n := 1 << params.LogMaxSlots()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

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
	// btpevk, _, _ := btpParams.GenEvaluationKeys(sk)

	// btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	// if err != nil {
	// 	panic(err)
	// }
	fmt.Println("generate Evaluator end")

	SF, _ := matmult.GenSFMat_CL(params, []int{2, 1, 1}, []int{2, 1, 1})
	scale := float64(1 << 25)
	mat0 := make([][][][]uint64, len(SF))
	mat1 := make([][][][]uint64, len(SF))
	mat0i := make([][][][]uint64, len(SF))
	mat1i := make([][][][]uint64, len(SF))

	for l := range SF {
		mat0[l] = make([][][]uint64, len(params.Q()))
		mat1[l] = make([][][]uint64, len(params.Q()))
		mat0i[l] = make([][][]uint64, len(params.Q()))
		mat1i[l] = make([][][]uint64, len(params.Q()))

		for q := range mat0[l] {
			mat0[l][q] = make([][]uint64, 2*n)
			mat1[l][q] = make([][]uint64, 2*n)
			mat0i[l][q] = make([][]uint64, 2*n)
			mat1i[l][q] = make([][]uint64, 2*n)
			for i := range 2 * n {
				mat0[l][q][i] = make([]uint64, 2*n)
				mat1[l][q][i] = make([]uint64, 2*n)
				mat0i[l][q][i] = make([]uint64, 2*n)
				mat1i[l][q][i] = make([]uint64, 2*n)
				for j := range 2 * n {
					if i < n && j < n {
						if real(SF[l][i][j]) >= 0 {
							mat0[l][q][i][j] = uint64(real(SF[l][i][j]) * scale)
						} else {
							mat0[l][q][i][j] = params.Q()[q] - uint64(-real(SF[l][i][j])*scale)
						}
						if imag(SF[l][i][j]) >= 0 {
							mat0i[l][q][i][j] = uint64(imag(SF[l][i][j]) * scale)
						} else {
							mat0i[l][q][i][j] = params.Q()[q] - uint64(-imag(SF[l][i][j])*scale)
						}
					}
					if i >= n && j >= n {
						if real(SF[l][i%n][j%n]) >= 0 {
							mat1[l][q][i][j] = uint64(real(SF[l][i%n][j%n]) * scale)
						} else {
							mat1[l][q][i][j] = params.Q()[q] - uint64(-real(SF[l][i%n][j%n])*scale)
						}
						if imag(SF[l][i%n][j%n]) >= 0 {
							mat1i[l][q][i][j] = uint64(imag(SF[l][i%n][j%n]) * scale)
						} else {
							mat1i[l][q][i][j] = params.Q()[q] - uint64(-imag(SF[l][i%n][j%n])*scale)
						}

					}
				}
			}
		}
	}

	value := make([]complex128, n)
	for i := range value {
		value[i] = complex(0.001*float64(i), 0.001*float64(n-i))
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = true

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	cts := make([]*rlwe.Ciphertext, 2*n)
	for i := range cts {
		cts[i] = ct.CopyNew()
	}

	fmt.Println("start c2s")
	starttime = time.Now()
	ctT := transpose.Transpose(cts, params, evaluator, encoder, 2*n)
	rev := matmult.BitReversePermutationMatrix(n)
	matrev := make([][]uint64, 2*n)
	for i := range matrev {
		matrev[i] = make([]uint64, 2*n)
		for j := range matrev[i] {
			if (i < n && j < n) || (i >= n && j >= n) {
				matrev[i][j] = uint64(real(rev[i%n][j%n]))
			}
		}
	}
	ctT = matmult.PPMM_Flint(ctT, matrev, params, 2*n)

	ctTC := make([]*rlwe.Ciphertext, 2*n)
	fmt.Println("ctT ctTC")
	for i := range ctTC {
		if i < n {
			ctTC[i], _ = evaluator.MulNew(ctT[i+n], -1)
		} else {
			ctTC[i] = ctT[i-n].CopyNew()
		}
	}

	var res00, res01, res10, res11, res00i, res01i, res10i, res11i []*rlwe.Ciphertext
	for l := range len(SF) {
		if l == 0 {
			res00 = matmult.PPMM_Flint_CRT(ctT, mat0[l], params, 2*n)
			res00i = matmult.PPMM_Flint_CRT(ctT, mat0i[l], params, 2*n)
			res01 = matmult.PPMM_Flint_CRT(ctTC, mat0[l], params, 2*n)
			res01i = matmult.PPMM_Flint_CRT(ctTC, mat0i[l], params, 2*n)

			res10 = matmult.PPMM_Flint_CRT(ctT, mat1[l], params, 2*n)
			res10i = matmult.PPMM_Flint_CRT(ctT, mat1i[l], params, 2*n)
			res11 = matmult.PPMM_Flint_CRT(ctTC, mat1[l], params, 2*n)
			res11i = matmult.PPMM_Flint_CRT(ctTC, mat1i[l], params, 2*n)

			for i := range 2 * n {
				evaluator.Mul(res00[i], 1.0/scale, res00[i])
				evaluator.Rescale(res00[i], res00[i])
				evaluator.Mul(res00i[i], 1.0/scale, res00i[i])
				evaluator.Rescale(res00i[i], res00i[i])

				evaluator.Mul(res01[i], 1.0/scale, res01[i])
				evaluator.Rescale(res01[i], res01[i])
				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
				evaluator.Rescale(res01i[i], res01i[i])

				evaluator.Mul(res10[i], 1.0/scale, res10[i])
				evaluator.Rescale(res10[i], res10[i])
				evaluator.Mul(res10i[i], 1.0/scale, res10i[i])
				evaluator.Rescale(res10i[i], res10i[i])

				evaluator.Mul(res11[i], 1.0/scale, res11[i])
				evaluator.Rescale(res11[i], res11[i])
				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
				evaluator.Rescale(res11i[i], res11i[i])
			}

		} else if l == len(SF)-1 {
			temp1 := matmult.PPMM_Flint_CRT(res00, mat0[l], params, 2*n)
			temp1_ := matmult.PPMM_Flint_CRT(res00i, mat0i[l], params, 2*n)
			temp2 := matmult.PPMM_Flint_CRT(res01, mat0i[l], params, 2*n)
			temp2_ := matmult.PPMM_Flint_CRT(res01i, mat0[l], params, 2*n)
			for i := range 2 * n {
				evaluator.Sub(temp1[i], temp1_[i], res00[i])
				evaluator.Add(temp2[i], temp2_[i], res01i[i])
			}

			temp1 = matmult.PPMM_Flint_CRT(res10, mat1[l], params, 2*n)
			temp1_ = matmult.PPMM_Flint_CRT(res10i, mat1i[l], params, 2*n)
			temp2 = matmult.PPMM_Flint_CRT(res11, mat1i[l], params, 2*n)
			temp2_ = matmult.PPMM_Flint_CRT(res11i, mat1[l], params, 2*n)
			for i := range 2 * n {
				evaluator.Sub(temp1[i], temp1_[i], res10[i])
				evaluator.Add(temp2[i], temp2_[i], res11i[i])
			}
			for i := range 2 * n {
				evaluator.Mul(res00[i], 1.0/scale, res00[i])
				evaluator.Rescale(res00[i], res00[i])

				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
				evaluator.Rescale(res01i[i], res01i[i])

				evaluator.Mul(res10[i], 1.0/scale, res10[i])
				evaluator.Rescale(res10[i], res10[i])

				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
				evaluator.Rescale(res11i[i], res11i[i])
			}
		} else {
			temp1 := matmult.PPMM_Flint_CRT(res00, mat0[l], params, 2*n)
			temp1_ := matmult.PPMM_Flint_CRT(res00i, mat0i[l], params, 2*n)
			temp2 := matmult.PPMM_Flint_CRT(res00, mat0i[l], params, 2*n)
			temp2_ := matmult.PPMM_Flint_CRT(res00i, mat0[l], params, 2*n)
			for i := range 2 * n {
				evaluator.Sub(temp1[i], temp1_[i], res00[i])
				evaluator.Add(temp2[i], temp2_[i], res00i[i])
			}

			temp1 = matmult.PPMM_Flint_CRT(res01, mat0[l], params, 2*n)
			temp1_ = matmult.PPMM_Flint_CRT(res01i, mat0i[l], params, 2*n)
			temp2 = matmult.PPMM_Flint_CRT(res01, mat0i[l], params, 2*n)
			temp2_ = matmult.PPMM_Flint_CRT(res01i, mat0[l], params, 2*n)
			for i := range 2 * n {
				evaluator.Sub(temp1[i], temp1_[i], res01[i])
				evaluator.Add(temp2[i], temp2_[i], res01i[i])
			}

			temp1 = matmult.PPMM_Flint_CRT(res10, mat1[l], params, 2*n)
			temp1_ = matmult.PPMM_Flint_CRT(res10i, mat1i[l], params, 2*n)
			temp2 = matmult.PPMM_Flint_CRT(res10, mat1i[l], params, 2*n)
			temp2_ = matmult.PPMM_Flint_CRT(res10i, mat1[l], params, 2*n)
			for i := range 2 * n {
				evaluator.Sub(temp1[i], temp1_[i], res10[i])
				evaluator.Add(temp2[i], temp2_[i], res10i[i])
			}

			temp1 = matmult.PPMM_Flint_CRT(res11, mat1[l], params, 2*n)
			temp1_ = matmult.PPMM_Flint_CRT(res11i, mat1i[l], params, 2*n)
			temp2 = matmult.PPMM_Flint_CRT(res11, mat1i[l], params, 2*n)
			temp2_ = matmult.PPMM_Flint_CRT(res11i, mat1[l], params, 2*n)
			for i := range 2 * n {
				evaluator.Sub(temp1[i], temp1_[i], res11[i])
				evaluator.Add(temp2[i], temp2_[i], res11i[i])
			}

			for i := range 2 * n {
				evaluator.Mul(res00[i], 1.0/scale, res00[i])
				evaluator.Rescale(res00[i], res00[i])
				evaluator.Mul(res00i[i], 1.0/scale, res00i[i])
				evaluator.Rescale(res00i[i], res00i[i])

				evaluator.Mul(res01[i], 1.0/scale, res01[i])
				evaluator.Rescale(res01[i], res01[i])
				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
				evaluator.Rescale(res01i[i], res01i[i])

				evaluator.Mul(res10[i], 1.0/scale, res10[i])
				evaluator.Rescale(res10[i], res10[i])
				evaluator.Mul(res10i[i], 1.0/scale, res10i[i])
				evaluator.Rescale(res10i[i], res10i[i])

				evaluator.Mul(res11[i], 1.0/scale, res11[i])
				evaluator.Rescale(res11[i], res11[i])
				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
				evaluator.Rescale(res11i[i], res11i[i])
			}
		}
	}

	res := make([]*rlwe.Ciphertext, 2*n)
	for i := range res {
		res[i], _ = evaluator.AddNew(res00[i], res01i[i])
		evaluator.Add(res[i], res10[i], res[i])
		evaluator.Add(res[i], res11i[i], res[i])
	}

	result0 := transpose.Transpose(res, params, evaluator, encoder, 2*n)

	elapse = time.Since(starttime)
	fmt.Println(elapse)
	fmt.Println(result0[0].LogScale())

	resvalue := make([]float64, 2*n)
	for i := range result0 {
		result0[i].IsBatched = false
		dept := decryptor.DecryptNew(result0[i])
		encoder.Decode(dept, resvalue)

		fmt.Println(resvalue)
	}

}

func Test_C2SLC_Opt(t *testing.T) {

	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	SchemeParams := hefloat.ParametersLiteral{
		LogN:            5,
		LogQ:            []int{48, 40, 40, 48},
		LogP:            []int{52},
		LogDefaultScale: 40,
	}
	//parameter init
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

	n := 1 << params.LogMaxSlots()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)

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
	// btpevk, _, _ := btpParams.GenEvaluationKeys(sk)

	// btp, err := bootstrapping.NewEvaluator(btpParams, btpevk)
	// if err != nil {
	// 	panic(err)
	// }
	fmt.Println("generate Evaluator end")

	fmt.Println("ckks log degree : ", params.LogN())

	CL_arr := []int{2, 1, 1}
	_, SFI := matmult.GenSFMat_CL(params, CL_arr, CL_arr)
	scale := float64(1 << 40)
	mat0 := make([][][][][]uint64, len(SFI))
	mat0i := make([][][][][]uint64, len(SFI))
	mat0si := make([][][][][]uint64, len(SFI))

	inter_it := n
	for l := range SFI {
		inter := inter_it >> CL_arr[l]
		llen := (1 << CL_arr[l])
		mat0[l] = make([][][][]uint64, n/llen)
		mat0i[l] = make([][][][]uint64, n/llen)
		mat0si[l] = make([][][][]uint64, n/llen)
		for t := range n / llen {
			mat0[l][t] = make([][][]uint64, len(params.Q()))
			mat0i[l][t] = make([][][]uint64, len(params.Q()))
			mat0si[l][t] = make([][][]uint64, len(params.Q()))
			stpoint := inter_it*int(t/inter) + (t % inter)
			for q := range len(params.Q()) {
				mat0[l][t][q] = make([][]uint64, llen)
				mat0i[l][t][q] = make([][]uint64, llen)
				mat0si[l][t][q] = make([][]uint64, llen)
				for i := range llen {
					mat0[l][t][q][i] = make([]uint64, llen)
					mat0i[l][t][q][i] = make([]uint64, llen)
					mat0si[l][t][q][i] = make([]uint64, llen)
					for j := range llen {
						if real(SFI[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
							mat0[l][t][q][i][j] = uint64(real(SFI[l][stpoint+inter*i][stpoint+inter*j]) * scale)
						} else {
							mat0[l][t][q][i][j] = params.Q()[q] - uint64(-real(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale)
						}
						if imag(SFI[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
							mat0i[l][t][q][i][j] = uint64(imag(SFI[l][stpoint+inter*i][stpoint+inter*j]) * scale)
						} else {
							mat0i[l][t][q][i][j] = params.Q()[q] - uint64(-imag(SFI[l][stpoint+inter*i][stpoint+inter*j])*scale)
						}
						mat0si[l][t][q][i][j] = (mat0[l][t][q][i][j] + mat0i[l][t][q][i][j]) % params.Q()[q]
					}
				}
			}
		}

		inter_it = inter
	}

	value := make([]float64, 2*n)
	for i := range value {
		value[i] = 0.001 * float64(i)
	}

	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	pt.IsBatched = false

	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	cts := make([]*rlwe.Ciphertext, 2*n)
	for i := range cts {
		cts[i] = ct.CopyNew()
	}

	res00 := make([]*rlwe.Ciphertext, 2*n)
	res00i := make([]*rlwe.Ciphertext, 2*n)
	res01 := make([]*rlwe.Ciphertext, 2*n)
	res01i := make([]*rlwe.Ciphertext, 2*n)

	res10 := make([]*rlwe.Ciphertext, 2*n)
	res10i := make([]*rlwe.Ciphertext, 2*n)
	res11 := make([]*rlwe.Ciphertext, 2*n)
	res11i := make([]*rlwe.Ciphertext, 2*n)
	temp1 := make([]*rlwe.Ciphertext, 2*n)
	temp1_ := make([]*rlwe.Ciphertext, 2*n)
	temp2 := make([]*rlwe.Ciphertext, 2*n)
	temp2_ := make([]*rlwe.Ciphertext, 2*n)
	ctZero := matmult.CtZero(params, encoder, encryptor)
	for i := range 2 * n {
		res00[i] = matmult.CtZero(params, encoder, encryptor)
		res00i[i] = matmult.CtZero(params, encoder, encryptor)
		res01[i] = matmult.CtZero(params, encoder, encryptor)
		res01i[i] = matmult.CtZero(params, encoder, encryptor)

		res10[i] = matmult.CtZero(params, encoder, encryptor)
		res10i[i] = matmult.CtZero(params, encoder, encryptor)
		res11[i] = matmult.CtZero(params, encoder, encryptor)
		res11i[i] = matmult.CtZero(params, encoder, encryptor)

		temp1[i] = matmult.CtZero(params, encoder, encryptor)
		temp1_[i] = matmult.CtZero(params, encoder, encryptor)
		temp2[i] = matmult.CtZero(params, encoder, encryptor)
		temp2_[i] = matmult.CtZero(params, encoder, encryptor)
	}

	fmt.Println("start c2s")
	starttime = time.Now()
	ctT := transpose.Transpose(cts, params, evaluator, encoder, 2*n)

	ctTC := make([]*rlwe.Ciphertext, 2*n)
	fmt.Println("ctT ctTC")
	for i := range ctTC {
		if i < n {
			ctTC[i], _ = evaluator.MulNew(ctT[i+n], -1)
		} else {
			ctTC[i] = ctT[i-n].CopyNew()
		}
	}

	//var res00, res01, res10, res11, res00i, res01i, res10i, res11i []*rlwe.Ciphertext
	inter_it = n
	for l := range len(SFI) {
		inter := inter_it >> CL_arr[l]
		llen := (1 << CL_arr[l])
		if l == 0 {
			for t := range n / llen {
				cts_temp := make([]*rlwe.Ciphertext, llen)
				stpoint := (t % inter) + inter_it*int(t/inter)
				for idx_ll := range llen {
					cts_temp[idx_ll] = ctT[stpoint+inter*idx_ll]
				}
				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					res00[stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = ctT[stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					res00i[stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = ctTC[n+stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					res01[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = ctTC[n+stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					res01i[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				//10~11
				for idx_ll := range llen {
					cts_temp[idx_ll] = ctTC[stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					res10[stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = ctTC[stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					res10i[stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = ctT[n+stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					res11[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = ctT[n+stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					res11i[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
				}
			}
			for i := range temp1 {
				temp1[i] = nil
				temp1_[i] = nil
				temp2[i] = nil
				temp2_[i] = nil
			}
			for i := range 2 * n {
				evaluator.Mul(res00[i], 1.0/scale, res00[i])
				evaluator.Rescale(res00[i], res00[i])
				evaluator.Mul(res00i[i], 1.0/scale, res00i[i])
				evaluator.Rescale(res00i[i], res00i[i])

				evaluator.Mul(res01[i], 1.0/scale, res01[i])
				evaluator.Rescale(res01[i], res01[i])
				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
				evaluator.Rescale(res01i[i], res01i[i])

				evaluator.Mul(res10[i], 1.0/scale, res10[i])
				evaluator.Rescale(res10[i], res10[i])
				evaluator.Mul(res10i[i], 1.0/scale, res10i[i])
				evaluator.Rescale(res10i[i], res10i[i])

				evaluator.Mul(res11[i], 1.0/scale, res11[i])
				evaluator.Rescale(res11[i], res11[i])
				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
				evaluator.Rescale(res11i[i], res11i[i])

			}

		} else if l == len(SFI)-1 {
			for t := range n / llen {
				cts_temp := make([]*rlwe.Ciphertext, llen)
				stpoint := (t % inter) + inter_it*int(t/inter)

				for idx_ll := range llen {
					cts_temp[idx_ll] = res00[stpoint+inter*idx_ll]
				}
				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					temp1[stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = res00i[stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					temp1_[stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = res01[n+stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					temp2[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = res01i[n+stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					temp2_[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
				}
			}
			res00 = matmult.SubMany(temp1, temp1_, evaluator)
			res01i = matmult.AddMany(temp2, temp2_, evaluator)
			for i := range temp1 {
				temp1[i] = nil
				temp1_[i] = nil
				temp2[i] = nil
				temp2_[i] = nil
			}

			for t := range n / llen {
				cts_temp := make([]*rlwe.Ciphertext, llen)
				stpoint := (t % inter) + inter_it*int(t/inter)

				for idx_ll := range llen {
					cts_temp[idx_ll] = res10[stpoint+inter*idx_ll]
				}
				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					temp1[stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = res10i[stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					temp1_[stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = res11[n+stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					temp2[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = res11i[n+stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					temp2_[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
				}
			}
			res10 = matmult.SubMany(temp1, temp1_, evaluator)
			res11i = matmult.AddMany(temp2, temp2_, evaluator)

			for i := range 2 * n {
				if res00[i] != nil {
					evaluator.Mul(res00[i], 1.0/scale, res00[i])
					evaluator.Rescale(res00[i], res00[i])
				}
				if res01i[i] != nil {
					evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
					evaluator.Rescale(res01i[i], res01i[i])
				}
				if res10[i] != nil {
					evaluator.Mul(res10[i], -1.0/scale, res10[i])
					evaluator.Rescale(res10[i], res10[i])
				}
				if res11i[i] != nil {
					evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
					evaluator.Rescale(res11i[i], res11i[i])
				}
			}
		} else {
			m1 := make([]*rlwe.Ciphertext, 2*n)
			m2 := make([]*rlwe.Ciphertext, 2*n)
			temp := make([]*rlwe.Ciphertext, 2*n)
			m3 := make([]*rlwe.Ciphertext, 2*n)
			m4 := make([]*rlwe.Ciphertext, 2*n)
			for i := range 2 * n {
				m1[i] = ctZero.CopyNew()
				m2[i] = ctZero.CopyNew()
				m3[i] = ctZero.CopyNew()
				m4[i] = ctZero.CopyNew()
				temp[i] = ctZero.CopyNew()
			}
			for t := range n / llen {
				cts_temp := make([]*rlwe.Ciphertext, llen)
				stpoint := (t % inter) + inter_it*int(t/inter)

				for idx_ll := range llen {
					cts_temp[idx_ll] = res00[stpoint+inter*idx_ll]
				}
				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					m1[stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = res00i[stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					m2[stpoint+inter*idx_ll] = res_temp[idx_ll]
				}
			}
			temp = matmult.AddMany(res00, res00i, evaluator)
			for t := range n / llen {
				cts_temp := make([]*rlwe.Ciphertext, llen)
				stpoint := (t % inter) + inter_it*int(t/inter)

				for idx_ll := range llen {
					cts_temp[idx_ll] = temp[stpoint+inter*idx_ll]
				}
				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0si[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					m3[stpoint+inter*idx_ll] = res_temp[idx_ll]
				}
			}
			m4 = matmult.AddMany(m1, m2, evaluator)

			res00 = matmult.SubMany(m1, m2, evaluator)
			res00i = matmult.SubMany(m3, m4, evaluator)

			for i := range 2 * n {
				m1[i] = ctZero.CopyNew()
				m2[i] = ctZero.CopyNew()
				m3[i] = ctZero.CopyNew()
				m4[i] = ctZero.CopyNew()
				temp[i] = ctZero.CopyNew()
			}
			for t := range n / llen {
				cts_temp := make([]*rlwe.Ciphertext, llen)
				stpoint := (t % inter) + inter_it*int(t/inter)

				for idx_ll := range llen {
					cts_temp[idx_ll] = res01[n+stpoint+inter*idx_ll]
				}
				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					m1[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = res01i[n+stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					m2[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
				}
			}
			temp = matmult.AddMany(res01, res01i, evaluator)
			for t := range n / llen {
				cts_temp := make([]*rlwe.Ciphertext, llen)
				stpoint := (t % inter) + inter_it*int(t/inter)

				for idx_ll := range llen {
					cts_temp[idx_ll] = temp[n+stpoint+inter*idx_ll]
				}
				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0si[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					m3[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
				}
			}
			m4 = matmult.AddMany(m1, m2, evaluator)
			res01 = matmult.SubMany(m1, m2, evaluator)
			res01i = matmult.SubMany(m3, m4, evaluator)

			for i := range 2 * n {
				m1[i] = ctZero.CopyNew()
				m2[i] = ctZero.CopyNew()
				m3[i] = ctZero.CopyNew()
				m4[i] = ctZero.CopyNew()
				temp[i] = ctZero.CopyNew()
			}
			for t := range n / llen {
				cts_temp := make([]*rlwe.Ciphertext, llen)
				stpoint := (t % inter) + inter_it*int(t/inter)

				for idx_ll := range llen {
					cts_temp[idx_ll] = res10[stpoint+inter*idx_ll]
				}
				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					m1[stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = res10i[stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					m2[stpoint+inter*idx_ll] = res_temp[idx_ll]
				}
			}
			temp = matmult.AddMany(res10, res10i, evaluator)
			for t := range n / llen {
				cts_temp := make([]*rlwe.Ciphertext, llen)
				stpoint := (t % inter) + inter_it*int(t/inter)

				for idx_ll := range llen {
					cts_temp[idx_ll] = temp[stpoint+inter*idx_ll]
				}
				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0si[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					m3[stpoint+inter*idx_ll] = res_temp[idx_ll]
				}
			}
			m4 = matmult.AddMany(m1, m2, evaluator)
			res10 = matmult.SubMany(m1, m2, evaluator)
			res10i = matmult.SubMany(m3, m4, evaluator)

			for i := range 2 * n {
				m1[i] = ctZero.CopyNew()
				m2[i] = ctZero.CopyNew()
				m3[i] = ctZero.CopyNew()
				m4[i] = ctZero.CopyNew()
				temp[i] = ctZero.CopyNew()
			}
			for t := range n / llen {
				cts_temp := make([]*rlwe.Ciphertext, llen)
				stpoint := (t % inter) + inter_it*int(t/inter)

				for idx_ll := range llen {
					cts_temp[idx_ll] = res11[n+stpoint+inter*idx_ll]
				}
				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					m1[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
				}

				for idx_ll := range llen {
					cts_temp[idx_ll] = res11i[n+stpoint+inter*idx_ll]
				}
				res_temp = matmult.PPMM_Flint_CRT2(cts_temp, mat0i[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					m2[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
				}
			}
			temp = matmult.AddMany(res11, res11i, evaluator)
			for t := range n / llen {
				cts_temp := make([]*rlwe.Ciphertext, llen)
				stpoint := (t % inter) + inter_it*int(t/inter)

				for idx_ll := range llen {
					cts_temp[idx_ll] = temp[n+stpoint+inter*idx_ll]
				}
				res_temp := matmult.PPMM_Flint_CRT2(cts_temp, mat0si[l][t], llen, llen, 2*n, params)
				for idx_ll := range llen {
					m3[n+stpoint+inter*idx_ll] = res_temp[idx_ll]
				}
			}
			m4 = matmult.AddMany(m1, m2, evaluator)
			res11 = matmult.SubMany(m1, m2, evaluator)
			res11i = matmult.SubMany(m3, m4, evaluator)

			for i := range 2 * n {
				evaluator.Mul(res00[i], 1.0/scale, res00[i])
				evaluator.Rescale(res00[i], res00[i])
				evaluator.Mul(res00i[i], 1.0/scale, res00i[i])
				evaluator.Rescale(res00i[i], res00i[i])

				evaluator.Mul(res01[i], 1.0/scale, res01[i])
				evaluator.Rescale(res01[i], res01[i])
				evaluator.Mul(res01i[i], 1.0/scale, res01i[i])
				evaluator.Rescale(res01i[i], res01i[i])

				evaluator.Mul(res10[i], 1.0/scale, res10[i])
				evaluator.Rescale(res10[i], res10[i])
				evaluator.Mul(res10i[i], 1.0/scale, res10i[i])
				evaluator.Rescale(res10i[i], res10i[i])

				evaluator.Mul(res11[i], 1.0/scale, res11[i])
				evaluator.Rescale(res11[i], res11[i])
				evaluator.Mul(res11i[i], 1.0/scale, res11i[i])
				evaluator.Rescale(res11i[i], res11i[i])
			}
		}
		inter_it = inter
	}

	res0 := matmult.AddMany(res00, res01i, evaluator)
	res1 := matmult.AddMany(res10, res11i, evaluator)
	// res0 := make([]*rlwe.Ciphertext, 2*n)
	// res1 := make([]*rlwe.Ciphertext, 2*n)
	// for i := range res0 {
	// 	fmt.Println(res00[i].Scale.Value.Float64())
	// 	fmt.Println(res01i[i].Scale.Value.Float64())
	// 	res0[i], _ = evaluator.AddNew(res00[i], res01i[i])
	// 	res1[i], _ = evaluator.AddNew(res10[i], res11i[i])
	// }

	rev := matmult.BitReversePermutationMatrix(n)
	matrev := make([][]uint64, 2*n)
	for i := range matrev {
		matrev[i] = make([]uint64, 2*n)
		for j := range matrev[i] {
			if (i < n && j < n) || (i >= n && j >= n) {
				matrev[i][j] = uint64(real(rev[i%n][j%n]))
			}
		}
	}
	res0 = matmult.PPMM_Flint(res0, matrev, params, 2*n)
	res1 = matmult.PPMM_Flint(res1, matrev, params, 2*n)

	result0 := transpose.Transpose(res0, params, evaluator, encoder, 2*n)
	result1 := transpose.Transpose(res1, params, evaluator, encoder, 2*n)

	elapse = time.Since(starttime)
	fmt.Println(elapse)
	fmt.Println(result0[0].LogScale())
	fmt.Println(result1[0].LogScale())

	resvalue := make([]complex128, n)
	for i := range result0 {
		result0[i].IsBatched = true
		dept := decryptor.DecryptNew(result0[i])
		encoder.Decode(dept, resvalue)

		fmt.Println(resvalue)
	}
	fmt.Println()
	fmt.Println()
	fmt.Println()
	for i := range result1 {
		result1[i].IsBatched = true
		dept := decryptor.DecryptNew(result1[i])
		encoder.Decode(dept, resvalue)

		fmt.Println(resvalue)
	}
	fmt.Println(result0[0].LogScale())

}
