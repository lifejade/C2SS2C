package matmult

import (
	"math"
	"math/bits"
	"math/cmplx"
	"runtime"
	"sync"

	cwrappingflint "github.com/lifejade/mm/src/matmult/cwrapping_flint"
	"github.com/lifejade/mm/src/transpose"
	"github.com/lifejade/mm/src/util"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/schemes/ckks"
)

func BitReversePermutationMatrix(n int) [][]complex128 {
	m := bits.Len64(uint64(n)) - 1
	P := make([][]complex128, n)
	for i := range P {
		P[i] = make([]complex128, n)
	}

	for i := 0; i < n; i++ {
		rev := util.BitReverse(i, m)
		P[i][rev] = 1.0
	}
	return P
}

func GenSFMat(params hefloat.Parameters) (SF, SFI [][]complex128) {
	n := params.MaxSlots()
	roots := ckks.GetRootsBigComplex(n<<2, params.EncodingPrecision())
	roots_complex := make([]complex128, 4*n)

	for i := range roots_complex {
		roots_complex[i] = roots[i].Complex128()
	}

	pow5 := make([]int, (n<<1)+1)
	pow5[0] = 1
	for i := 1; i < (n<<1)+1; i++ {
		pow5[i] = pow5[i-1] * 5
		pow5[i] &= (n << 2) - 1
	}

	SF = make([][]complex128, n)

	for i := range SF {
		SF[i] = make([]complex128, n)
		for j := range SF[i] {
			idx := (pow5[i] * j) & ((n << 2) - 1)
			SF[i][j] = roots_complex[idx]
		}
	}

	SFI = make([][]complex128, n)
	for i := range SFI {
		SFI[i] = make([]complex128, n)
	}
	for i := range SFI {
		for j := range SFI[i] {
			idx := (pow5[i] * j) & ((n << 2) - 1)
			SFI[j][i] = cmplx.Conj(roots_complex[idx]) / complex((float64(n)), 0)
		}
	}
	return
}

func GenC2SMat(SFI [][]complex128, scale float64, params hefloat.Parameters) (mat0, mat1, mat2, mat3 [][][]uint64) {
	mat0 = make([][][]uint64, len(params.Q()))
	mat1 = make([][][]uint64, len(params.Q()))
	n := params.MaxSlots()

	for q := range mat0 {
		mat0[q] = make([][]uint64, 2*n)
		mat1[q] = make([][]uint64, 2*n)
		for i := range 2 * n {
			mat0[q][i] = make([]uint64, 2*n)
			mat1[q][i] = make([]uint64, 2*n)
			for j := range 2 * n {
				if i < n && j < n {
					if real(SFI[i][j]) >= 0 {
						mat0[q][i][j] = uint64(real(SFI[i][j]) * scale)
					} else {
						mat0[q][i][j] = uint64(int(params.Q()[q]) + int(real(SFI[i][j])*scale))
					}

				} else {
					mat0[q][i][j] = 0
				}
				if i >= n && j >= n {
					if imag(SFI[i%n][j%n]) >= 0 {
						mat1[q][i][j] = uint64(imag(SFI[i%n][j%n]) * scale)
					} else {
						mat1[q][i][j] = uint64(int(params.Q()[q]) + int(imag(SFI[i%n][j%n])*scale))
					}

				} else {
					mat1[q][i][j] = 0
				}
			}
		}
	}

	mat2 = make([][][]uint64, 2*n)
	mat3 = make([][][]uint64, 2*n)
	for q := range len(params.Q()) {
		mat2[q] = make([][]uint64, 2*n)
		mat3[q] = make([][]uint64, 2*n)
		for i := range 2 * n {
			mat2[q][i] = make([]uint64, 2*n)
			mat3[q][i] = make([]uint64, 2*n)
			for j := range 2 * n {
				if i < n && j < n {
					if real(SFI[i][j]) >= 0 {
						mat2[q][i][j] = uint64(int(params.Q()[q]) - int(real(SFI[i][j])*scale))
					} else {
						mat2[q][i][j] = uint64(-real(SFI[i][j]) * scale)
					}
				} else {
					mat2[q][i][j] = 0
				}
				if i >= n && j >= n {
					if imag(SFI[i%n][j%n]) >= 0 {
						mat3[q][i][j] = uint64(imag(SFI[i%n][j%n]) * scale)
					} else {
						mat3[q][i][j] = uint64(int(params.Q()[q]) + int(imag(SFI[i%n][j%n])*scale))
					}
				} else {
					mat3[q][i][j] = 0
				}

			}
		}
	}

	return
}
func C2S_OnceMul(cts []*rlwe.Ciphertext, params hefloat.Parameters, evaluator *hefloat.Evaluator, encoder *hefloat.Encoder, mat0, mat1, mat2, mat3 [][][]uint64, scale float64) (res0, res1 []*rlwe.Ciphertext) {
	n := params.MaxSlots()

	ctT := transpose.Transpose(cts, params, evaluator, encoder, 2*n)
	ctT2 := make([]*rlwe.Ciphertext, 2*n)
	for i := range ctT2 {
		if i < n {
			ctT2[i], _ = evaluator.MulNew(ctT[i+n], -1)
		} else {
			ctT2[i] = ctT[i-n].CopyNew()
		}
	}

	res00 := PPMM_Flint_CRT(ctT, mat0, params, 2*n)
	res01 := PPMM_Flint_CRT(ctT2, mat1, params, 2*n)
	res0 = make([]*rlwe.Ciphertext, 2*n)
	for i := range res0 {
		res0[i], _ = evaluator.AddNew(res00[i], res01[i])
		evaluator.Mul(res0[i], 1.0/(scale), res0[i])
		evaluator.Rescale(res0[i], res0[i])
	}

	res10 := PPMM_Flint_CRT(ctT2, mat2, params, 2*n)
	res11 := PPMM_Flint_CRT(ctT, mat3, params, 2*n)
	res1 = make([]*rlwe.Ciphertext, 2*n)
	for i := range res1 {
		res1[i], _ = evaluator.AddNew(res10[i], res11[i])
		evaluator.Mul(res1[i], 1.0/scale, res1[i])
		evaluator.Rescale(res1[i], res1[i])
	}

	res0 = transpose.Transpose(res0, params, evaluator, encoder, 2*n)
	res1 = transpose.Transpose(res1, params, evaluator, encoder, 2*n)

	return
}

func GenS2CMat(SF [][]complex128, scale float64, params hefloat.Parameters) (mat0, mat1, mat2, mat3 [][][]uint64) {
	mat0 = make([][][]uint64, len(params.Q()))
	mat1 = make([][][]uint64, len(params.Q()))
	n := params.MaxSlots()

	for q := range mat0 {
		mat0[q] = make([][]uint64, 2*n)
		mat1[q] = make([][]uint64, 2*n)
		for i := range 2 * n {
			mat0[q][i] = make([]uint64, 2*n)
			mat1[q][i] = make([]uint64, 2*n)
			for j := range 2 * n {
				if i < n && j < n {
					if real(SF[i][j]) >= 0 {
						mat0[q][i][j] = uint64(real(SF[i][j]) * scale)
					} else {
						mat0[q][i][j] = uint64(int(params.Q()[q]) + int(real(SF[i][j])*scale))
					}

				} else {
					mat0[q][i][j] = 0
				}
				if i < n && j < n {
					if imag(SF[i%n][j%n]) >= 0 {
						mat1[q][i][j] = uint64(imag(SF[i%n][j%n]) * scale)
					} else {
						mat1[q][i][j] = uint64(int(params.Q()[q]) + int(imag(SF[i%n][j%n])*scale))
					}

				} else {
					mat1[q][i][j] = 0
				}
			}
		}
	}

	mat2 = make([][][]uint64, 2*n)
	mat3 = make([][][]uint64, 2*n)
	for q := range len(params.Q()) {
		mat2[q] = make([][]uint64, 2*n)
		mat3[q] = make([][]uint64, 2*n)
		for i := range 2 * n {
			mat2[q][i] = make([]uint64, 2*n)
			mat3[q][i] = make([]uint64, 2*n)
			for j := range 2 * n {
				if i >= n && j >= n {
					if imag(SF[i%n][j%n]) >= 0 {
						mat2[q][i][j] = uint64(int(params.Q()[q]) - int(imag(SF[i%n][j%n])*scale))
					} else {
						mat2[q][i][j] = uint64(-imag(SF[i%n][j%n]) * scale)
					}
				}
				if i >= n && j >= n {
					if real(SF[i%n][j%n]) >= 0 {
						mat3[q][i][j] = uint64(real(SF[i%n][j%n]) * scale)
					} else {
						mat3[q][i][j] = uint64(int(params.Q()[q]) + int(real(SF[i%n][j%n])*scale))
					}

				}

			}
		}
	}

	return
}
func S2C_OnceMul(cts1, cts2 []*rlwe.Ciphertext, params hefloat.Parameters, evaluator *hefloat.Evaluator, encoder *hefloat.Encoder, mat0, mat1, mat2, mat3 [][][]uint64, scale float64) (res []*rlwe.Ciphertext) {
	n := params.MaxSlots()

	ctT1 := transpose.Transpose(cts1, params, evaluator, encoder, 2*n)
	ctT2 := transpose.Transpose(cts2, params, evaluator, encoder, 2*n)

	ctT1C := make([]*rlwe.Ciphertext, 2*n)
	ctT2C := make([]*rlwe.Ciphertext, 2*n)

	for i := range ctT1C {
		if i < n {
			ctT1C[i], _ = evaluator.MulNew(ctT1[i+n], -1)
			ctT2C[i], _ = evaluator.MulNew(ctT2[i+n], -1)
		} else {
			ctT1C[i] = ctT1[i-n].CopyNew()
			ctT2C[i] = ctT2[i-n].CopyNew()
		}
	}

	res00 := PPMM_Flint_CRT(ctT1, mat0, params, 2*n)
	res01 := PPMM_Flint_CRT(ctT1C, mat1, params, 2*n)
	res0 := make([]*rlwe.Ciphertext, 2*n)
	for i := range res0 {
		res0[i], _ = evaluator.AddNew(res00[i], res01[i])
	}

	res10 := PPMM_Flint_CRT(ctT2, mat2, params, 2*n)
	res11 := PPMM_Flint_CRT(ctT2C, mat3, params, 2*n)
	res1 := make([]*rlwe.Ciphertext, 2*n)
	for i := range res1 {
		res1[i], _ = evaluator.AddNew(res10[i], res11[i])
	}

	res = make([]*rlwe.Ciphertext, 2*n)
	for i := range res {
		res[i], _ = evaluator.AddNew(res0[i], res1[i])
		evaluator.Mul(res[i], 1.0/scale, res[i])
		evaluator.Rescale(res[i], res[i])
	}

	res = transpose.Transpose(res, params, evaluator, encoder, 2*n)

	return
}
func GenSFMat_CL(params hefloat.Parameters, SF_arr, SFI_arr []int) (SF_CL, SFI_CL [][][]complex128) {
	logn := params.LogMaxSlots()
	n := 1 << logn
	roots := ckks.GetRootsBigComplex(n<<2, params.EncodingPrecision())
	roots_complex := make([]complex128, 4*n)

	for i := range roots_complex {
		roots_complex[i] = roots[i].Complex128()
	}

	div := complex(math.Pow(float64(n), 1/float64(logn)), 0)
	div = complex(1, 0)
	// --- SF_CL 생성 ---
	SF_CL = make([][][]complex128, len(SF_arr))
	idx := 0
	for i := range SF_CL {
		// 원래 코드: for j := range SF_arr[i] { ... idx++ }
		// SF_arr[i]는 반복 횟수(또는 0이 아닌 요소의 개수)를 의미함
		count := SF_arr[i]
		SF_CL[i] = computeCombinedMat(n, idx, count, roots_complex, div, false)
		idx += count
	}

	// --- SFI_CL 생성 ---
	SFI_CL = make([][][]complex128, len(SFI_arr))
	idx = 0
	for i := range SFI_CL {
		count := SFI_arr[i]
		SFI_CL[i] = computeCombinedMat(n, idx, count, roots_complex, div, true)
		idx += count
	}

	return
}

func GenSFMat_CL2(params hefloat.Parameters, n int, SF_arr, SFI_arr []int) (SF_CL, SFI_CL [][][]complex128) {

	roots := ckks.GetRootsBigComplex(n<<2, params.EncodingPrecision())
	roots_complex := make([]complex128, 4*n)

	for i := range roots_complex {
		roots_complex[i] = roots[i].Complex128()
	}

	div := complex(1, 0)
	// --- SF_CL 생성 ---
	if SF_arr != nil {
		SF_CL = make([][][]complex128, len(SF_arr))
		idx := 0
		for i := range SF_CL {
			// 원래 코드: for j := range SF_arr[i] { ... idx++ }
			// SF_arr[i]는 반복 횟수(또는 0이 아닌 요소의 개수)를 의미함
			count := SF_arr[i]
			SF_CL[i] = computeCombinedMat(n, idx, count, roots_complex, div, false)
			idx += count
		}

	}

	// --- SFI_CL 생성 ---
	if SFI_arr != nil {
		SFI_CL = make([][][]complex128, len(SFI_arr))
		idx := 0
		for i := range SFI_CL {
			count := SFI_arr[i]
			SFI_CL[i] = computeCombinedMat(n, idx, count, roots_complex, div, true)
			idx += count
		}
	}

	return
}
func GenSFMat_CL3(params hefloat.Parameters, n int, SF_arr, SFI_arr []int) (SF_CL, SFI_CL [][]complex128) {

	roots := ckks.GetRootsBigComplex(n<<2, params.EncodingPrecision())
	roots_complex := make([]complex128, 4*n)

	for i := range roots_complex {
		roots_complex[i] = roots[i].Complex128()
	}

	div := complex(1, 0)
	// --- SF_CL 생성 ---
	if SF_arr != nil {
		SF_CL = make([][]complex128, len(SF_arr))
		idx := 0
		for i := range SF_CL {
			// 원래 코드: for j := range SF_arr[i] { ... idx++ }
			// SF_arr[i]는 반복 횟수(또는 0이 아닌 요소의 개수)를 의미함
			count := SF_arr[i]
			mat := cwrappingflint.ComputeCombinedMat(n, idx, count, roots_complex, div, false)
			SF_CL[i] = mat
			idx += count
		}

	}

	// --- SFI_CL 생성 ---
	if SFI_arr != nil {
		SFI_CL = make([][]complex128, len(SFI_arr))
		idx := 0
		for i := range SFI_CL {
			count := SFI_arr[i]
			mat := cwrappingflint.ComputeCombinedMat(n, idx, count, roots_complex, div, true)
			SFI_CL[i] = mat
			idx += count
		}
	}

	return
}

// 행렬 곱셈 결과를 효율적으로 계산하는 헬퍼 함수
func computeCombinedMat(n, startIdx, count int, roots []complex128, div complex128, isInverse bool) [][]complex128 {
	mat := make([][]complex128, n)
	for i := range mat {
		mat[i] = make([]complex128, n)
	}

	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	colChan := make(chan int, n)

	for w := 0; w < numCPUs; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range colChan {
				col := make([]complex128, n)
				col[j] = 1 // 단위 벡터

				currIdx := startIdx
				for l := 0; l < count; l++ {
					if isInverse {
						col = applySFIStep(currIdx, n, roots, div, col)
					} else {
						col = applySFStep(currIdx, n, roots, col)
					}
					currIdx++
				}

				for i := 0; i < n; i++ {
					mat[i][j] = col[i]
				}
			}
		}()
	}

	for j := 0; j < n; j++ {
		colChan <- j
	}
	close(colChan)
	wg.Wait()
	return mat
}

// Butterfly 연산 (Sparse Matrix 연산 최적화)
func applySFStep(idx, n int, roots []complex128, v []complex128) []complex128 {
	res := make([]complex128, n)
	m := 1 << (idx + 1)
	halfM, gap := m>>1, n/m
	for i := 0; i < n; i += m {
		pow5v := 1
		for j := 0; j < halfM; j++ {
			k := pow5v * gap
			u, w := v[i+j], v[i+j+halfM]*roots[k]
			res[i+j], res[i+j+halfM] = u+w, u-w
			pow5v = (pow5v * 5) & ((m << 2) - 1)
		}
	}
	return res
}

func applySFIStep(idx, n int, roots []complex128, div complex128, v []complex128) []complex128 {
	res := make([]complex128, n)
	m := n >> idx
	halfM, gap := m>>1, n/m

	for i := 0; i < n; i += m {
		pow5v := 1
		for j := 0; j < halfM; j++ {
			k := pow5v * gap
			wInv := cmplx.Conj(roots[k])

			a := v[i+j]
			b := v[i+j+halfM]

			// Matches:
			// [1/div, 1/div; conj(root)/div, -conj(root)/div] * [a;b]
			res[i+j] = (a + b) / div
			res[i+j+halfM] = (wInv * (a - b)) / div

			pow5v = (pow5v * 5) & ((m << 2) - 1)
		}
	}
	return res
}

func GenSFMat_CL_Slow(
	params hefloat.Parameters,
	SF_arr, SFI_arr []int,
) (SF_CL, SFI_CL [][][]complex128) {

	logn := params.LogMaxSlots()
	n := 1 << logn

	roots := ckks.GetRootsBigComplex(n<<2, params.EncodingPrecision())
	roots_complex := make([]complex128, 4*n)
	for i := range roots_complex {
		roots_complex[i] = roots[i].Complex128()
	}

	// --------------------
	// Forward SF
	// --------------------
	SF_ := make([][][]complex128, logn)
	for idx := 0; idx < logn; idx++ {
		SF_[idx] = make([][]complex128, n)
		for i := 0; i < n; i++ {
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
				pow5v &= ((m << 2) - 1)
			}
		}
	}

	// --------------------
	// Inverse SF
	// --------------------
	SFI_ := make([][][]complex128, logn)
	div := complex(math.Pow(float64(n), 1/float64(logn)), 0)
	// div := complex(1, 0)

	for idx := 0; idx < logn; idx++ {
		SFI_[idx] = make([][]complex128, n)
		for i := 0; i < n; i++ {
			SFI_[idx][i] = make([]complex128, n)
		}

		m := n >> idx
		for i := 0; i < n; i += m {
			pow5v := 1
			for j := 0; j < (m >> 1); j++ {
				k := pow5v * n / m

				SFI_[idx][i+j][i+j] = 1 / div
				SFI_[idx][i+j][i+j+(m>>1)] = 1 / div

				SFI_[idx][i+j+(m>>1)][i+j] =
					cmplx.Conj(roots_complex[k] / div)
				SFI_[idx][i+j+(m>>1)][i+j+(m>>1)] =
					-cmplx.Conj(roots_complex[k] / div)

				pow5v *= 5
				pow5v &= ((m << 2) - 1)
			}
		}
	}

	// --------------------
	// Compose forward
	// --------------------
	l := len(SF_arr)
	SF_CL = make([][][]complex128, l)
	idx := 0
	for i := range SF_CL {
		SF_CL[i] = SF_[idx]
		for j := range SF_arr[i] {
			if j == 0 {
				idx++
				continue
			}
			SF_CL[i] = mul(SF_[idx], SF_CL[i])
			idx++
		}
	}

	// --------------------
	// Compose inverse
	// --------------------
	l = len(SFI_arr)
	SFI_CL = make([][][]complex128, l)
	idx = 0
	for i := range SFI_CL {
		SFI_CL[i] = SFI_[idx]
		for j := range SFI_arr[i] {
			if j == 0 {
				idx++
				continue
			}
			SFI_CL[i] = mul(SFI_[idx], SFI_CL[i])
			idx++
		}
	}

	return
}

// 행렬 크기 확인 (직사각형/공백 방지)
func dims(m [][]complex128) (r, c int, ok bool) {
	r = len(m)
	if r == 0 {
		return 0, 0, false
	}
	c = len(m[0])
	if c == 0 {
		return 0, 0, false
	}
	for i := 1; i < r; i++ {
		if len(m[i]) != c {
			return 0, 0, false
		}
	}
	return r, c, true
}

// 행렬 곱 (r x k) * (k x c) = (r x c)
func mul(a, b [][]complex128) [][]complex128 {
	ar, ac, okA := dims(a)
	br, bc, okB := dims(b)
	if !okA || !okB || ac != br {
		return nil
	}
	out := make([][]complex128, ar)
	for i := 0; i < ar; i++ {
		out[i] = make([]complex128, bc)
		for j := 0; j < bc; j++ {
			var s complex128
			for k := 0; k < ac; k++ {
				s += a[i][k] * b[k][j]
			}
			out[i][j] = s
		}
	}
	return out
}

func PPMM_Flint(cts []*rlwe.Ciphertext, u [][]uint64, params hefloat.Parameters, n int) []*rlwe.Ciphertext {
	level := cts[0].Level() + 1

	a := make([][][]uint64, level)
	b := make([][][]uint64, level)

	result := make([]*rlwe.Ciphertext, n)
	for j := range level {
		a[j] = make([][]uint64, n)
		b[j] = make([][]uint64, n)
	}
	for i := range n {
		result[i] = cts[i].CopyNew()
		for j := range level {
			params.RingQ().AtLevel(j).INTT(result[i].Value[0], result[i].Value[0])
			params.RingQ().AtLevel(j).INTT(result[i].Value[1], result[i].Value[1])

			a[j][i] = (result[i].Value[0].Coeffs[j])
			b[j][i] = (result[i].Value[1].Coeffs[j])
		}
	}

	CA := make([][][]uint64, level)
	CB := make([][][]uint64, level)
	for i := range level {
		CA[i] = cwrappingflint.Mult_mod_mat(u, a[i], n, n, n, params.RingQ().AtLevel(i).Modulus().Uint64())
		CB[i] = cwrappingflint.Mult_mod_mat(u, b[i], n, n, n, params.RingQ().AtLevel(i).Modulus().Uint64())
	}

	for i := range result {
		for j := range level {
			result[i].Value[0].Coeffs[j] = CA[j][i]
			result[i].Value[1].Coeffs[j] = CB[j][i]
			params.RingQ().AtLevel(j).NTT(result[i].Value[0], result[i].Value[0])
			params.RingQ().AtLevel(j).NTT(result[i].Value[1], result[i].Value[1])
		}
	}
	return result
}

func PPMM_Flint_CRT(cts []*rlwe.Ciphertext, u [][][]uint64, params hefloat.Parameters, n int) []*rlwe.Ciphertext {
	level := cts[0].Level() + 1

	a := make([][][]uint64, level)
	b := make([][][]uint64, level)

	result := make([]*rlwe.Ciphertext, n)
	for j := range level {
		a[j] = make([][]uint64, n)
		b[j] = make([][]uint64, n)
	}
	for i := range n {
		result[i] = cts[i].CopyNew()
		for j := range level {
			params.RingQ().AtLevel(j).INTT(result[i].Value[0], result[i].Value[0])
			params.RingQ().AtLevel(j).INTT(result[i].Value[1], result[i].Value[1])

			a[j][i] = (result[i].Value[0].Coeffs[j])
			b[j][i] = (result[i].Value[1].Coeffs[j])
		}
	}

	CA := make([][][]uint64, level)
	CB := make([][][]uint64, level)
	for i := range level {
		CA[i] = cwrappingflint.Mult_mod_mat(u[i], a[i], n, n, n, params.Q()[i])
		CB[i] = cwrappingflint.Mult_mod_mat(u[i], b[i], n, n, n, params.Q()[i])
	}

	for i := range result {
		for j := range level {
			result[i].Value[0].Coeffs[j] = CA[j][i]
			result[i].Value[1].Coeffs[j] = CB[j][i]
			params.RingQ().AtLevel(j).NTT(result[i].Value[0], result[i].Value[0])
			params.RingQ().AtLevel(j).NTT(result[i].Value[1], result[i].Value[1])
		}
	}
	return result
}

func PPMM_Flint_CRT2(cts []*rlwe.Ciphertext, u [][][]uint64, n_a, n_b, n_c int, params hefloat.Parameters) []*rlwe.Ciphertext {
	level := cts[0].Level() + 1

	a := make([][][]uint64, level)
	b := make([][][]uint64, level)

	result := make([]*rlwe.Ciphertext, n_a)
	for j := range level {
		a[j] = make([][]uint64, n_b)
		b[j] = make([][]uint64, n_b)
	}
	for i := range n_b {
		result[i] = cts[i].CopyNew()
		for j := range level {
			params.RingQ().AtLevel(j).INTT(result[i].Value[0], result[i].Value[0])
			params.RingQ().AtLevel(j).INTT(result[i].Value[1], result[i].Value[1])

			a[j][i] = (result[i].Value[0].Coeffs[j])
			b[j][i] = (result[i].Value[1].Coeffs[j])
		}
	}

	CA := make([][][]uint64, level)
	CB := make([][][]uint64, level)
	for i := range level {
		CA[i] = cwrappingflint.Mult_mod_mat(u[i], a[i], n_a, n_b, n_c, params.Q()[i])
		CB[i] = cwrappingflint.Mult_mod_mat(u[i], b[i], n_a, n_b, n_c, params.Q()[i])
	}

	for i := range result {
		for j := range level {
			result[i].Value[0].Coeffs[j] = CA[j][i]
			result[i].Value[1].Coeffs[j] = CB[j][i]
			params.RingQ().AtLevel(j).NTT(result[i].Value[0], result[i].Value[0])
			params.RingQ().AtLevel(j).NTT(result[i].Value[1], result[i].Value[1])
		}
	}
	return result
}

func PPMM_Flint_CRT3(cts []*rlwe.Ciphertext, u [][]uint64, n_a, n_b, n_c int, params hefloat.Parameters) []*rlwe.Ciphertext {
	level := cts[0].Level() + 1

	a := make([][][]uint64, level)
	b := make([][][]uint64, level)

	result := make([]*rlwe.Ciphertext, n_a)
	for j := range level {
		a[j] = make([][]uint64, n_b)
		b[j] = make([][]uint64, n_b)
	}
	for i := range n_b {
		result[i] = cts[i].CopyNew()
		for j := range level {
			a[j][i] = (result[i].Value[0].Coeffs[j])
			b[j][i] = (result[i].Value[1].Coeffs[j])
		}
	}

	CA := make([][][]uint64, level)
	CB := make([][][]uint64, level)
	Q := params.Q()
	for i := range level {
		CA[i] = cwrappingflint.Mult_mod_mat2(u[i], a[i], n_a, n_b, n_c, Q[i])
		CB[i] = cwrappingflint.Mult_mod_mat2(u[i], b[i], n_a, n_b, n_c, Q[i])
	}

	for i := range result {
		for j := range level {
			result[i].Value[0].Coeffs[j] = CA[j][i]
			result[i].Value[1].Coeffs[j] = CB[j][i]
		}
	}
	return result
}

func PPMM_Blas_CRT(cts []ring.Poly, u [][][]uint64, params hefloat.Parameters, n_a, n_b, n_c, level int, ringP *ring.Ring, result []ring.Poly) {
	a := make([][][]uint64, level)
	for j := range level {
		a[j] = make([][]uint64, n_b)
	}

	for i := range n_b {
		for j := range level {
			a[j][i] = (cts[i].Coeffs[j])
		}
	}

	CA := make([][][]uint64, level)
	P := ringP.ModuliChain()
	for i := range level {
		CA[i] = cwrappingflint.Mult_mod_mat_Blas(u[i], a[i], n_a, n_b, n_c, P[i])
	}

	for i := range n_a {
		for j := range level {
			result[i].Coeffs[j] = CA[j][i]
		}
	}
}

func PPMM_Blas_CRT_Inplace(cts [][]ring.Poly, u []float64, n_a, n_b, n_c, level, degree int, ringP *ring.Ring, buffer1, buffer2 []float64) {
	P := ringP.ModuliChain()
	_ = P

	for d := range degree {
		index := 0
		for i := range level {
			for j := range n_b {
				coeff := cts[d][j].Coeffs[i]
				for k := range n_c {
					buffer1[index] = float64(coeff[k])
					index++
				}
			}
		}

		cwrappingflint.Mult_mod_mat_Blas_Inplace(u, buffer1, buffer2, n_a, n_b, n_c, level)

		index = 0
		for i := range level {
			p := P[i]
			for j := range n_b {
				coeff := cts[d][j].Coeffs[i]
				for k := range n_c {
					coeff[k] = uint64(buffer2[index]) % p
					index++
				}
			}
		}
	}

}

func PPMM_Blas_CRT_Stride(cts [][]ring.Poly, u []float64, n_a, n_b, n_c, stpoint, stride, level, degree int, ringP *ring.Ring, res [][]ring.Poly, buffer1, buffer2 []float64) {
	P := ringP.ModuliChain()
	_ = P
	level = level + 1
	for d := range degree {
		index := 0
		for i := range level {
			for j := range n_b {
				idx := stpoint + j*stride
				coeff := cts[d][idx].Coeffs[i]
				for k := range n_c {
					buffer1[index] = float64(coeff[k])
					index++
				}
			}
		}
		cwrappingflint.Mult_mod_mat_Blas_Inplace(u, buffer1, buffer2, n_a, n_b, n_c, level)

		index = 0
		for i := range level {
			p := int64(P[i])
			for j := range n_b {
				idx := stpoint + j*stride
				coeff := res[d][idx].Coeffs[i]
				for k := range n_c {
					val := int64(buffer2[index]) % p
					if val < 0 {
						coeff[k] = uint64(val + p)
					} else {
						coeff[k] = uint64(val)
					}
					index++

				}
			}
		}
	}
}

func PPMM_Blas_CRT_Stride2(cts []Poly, u []float64, n_a, n_b, n_c, stpoint, endpoint, stride, level int, ringP *ring.Ring, res []Poly, buffer1, buffer2 []float64) {
	P := ringP.ModuliChain()
	_ = P
	level = level + 1
	index := 0
	for i := range level {
		for j := range n_b {
			idx := stpoint + j*stride
			coeff := cts[idx].Coeffs[i]
			for k := range n_c {
				buffer1[index] = float64(coeff[k])
				index++
			}
		}
	}
	cwrappingflint.Mult_mod_mat_Blas_Inplace(u, buffer1, buffer2, n_a, n_b, n_c, level)

	index = 0
	for i := range level {
		p := int64(P[i])
		for j := range n_b {
			idx := endpoint + j*stride
			coeff := res[idx].Coeffs[i]
			for k := range n_c {
				val := int64(buffer2[index]) % p
				if val < 0 {
					coeff[k] = uint32(val + p)
				} else {
					coeff[k] = uint32(val)
				}
				index++

			}
		}
	}
}

func PPMM_Blas_CRT_Stride_LowMem(input []uint32, u []float64, n_a, n_b, n_c, stpoint, endpoint, stride int, p int64, res []uint32, buffer1, buffer2 []float64) {
	// P := ringP.ModuliChain()
	// _ = P
	index := 0
	for j := range n_b {
		idx := stpoint + j*stride
		for k := range n_c {
			buffer1[index] = float64(input[idx*n_c+k])
			index++
		}
	}
	cwrappingflint.Mult_mod_mat_Blas_Inplace2(u, buffer1, buffer2, n_a, n_b, n_c)

	index = 0
	for j := range n_b {
		idx := endpoint + j*stride
		for k := range n_c {
			val := int64(buffer2[index]) % p
			if val < 0 {
				res[idx*n_c+k] = uint32(val + p)
			} else {
				res[idx*n_c+k] = uint32(val)
			}
			index++
		}
	}

}

func PPMM_Blas_CRTBarret(cts []ring.Poly, u [][][]uint64, params hefloat.Parameters, n_a, n_b, n_c, level int, bred []uint64, ringP *ring.Ring, result []ring.Poly) {
	a := make([][][]uint64, level)
	for j := range level {
		a[j] = make([][]uint64, n_b)
	}

	for i := range n_b {
		for j := range level {
			a[j][i] = (cts[i].Coeffs[j])
		}
	}

	CA := make([][][]uint64, level)
	P := ringP.ModuliChain()
	for i := range level {
		CA[i] = cwrappingflint.Mult_mod_mat_BlasBarret(u[i], a[i], n_a, n_b, n_c, P[i], bred[i])
	}

	for i := range n_a {
		for j := range level {
			result[i].Coeffs[j] = CA[j][i]
		}
	}
}

func AddManyRing(r *ring.Ring, p1, p2, p3 [][]ring.Poly) {
	for i := range p2 {
		for j := range p2[i] {
			r.Add(p1[i][j], p2[i][j], p3[i][j])
		}
	}
}

func SubManyRing(r *ring.Ring, p1, p2, p3 [][]ring.Poly) {
	for i := range p2 {
		for j := range p2[i] {
			r.Sub(p1[i][j], p2[i][j], p3[i][j])
		}
	}
}

func AddManyRingIdx(r *ring.Ring, p1, p2, p3 []Poly, stpIdx int) {
	l := len(p2)
	for j := stpIdx; j < stpIdx+l; j++ {
		for i, s := range r.SubRings[:r.Level()+1] {
			addvec(p1[j].Coeffs[i], p2[j-stpIdx].Coeffs[i], p3[j].Coeffs[i], uint32(s.Modulus))
		}
	}
}

func SubManyRingIdx(r *ring.Ring, p1, p2, p3 []Poly, stpIdx int) {
	l := len(p2)
	for j := stpIdx; j < stpIdx+l; j++ {
		for i, s := range r.SubRings[:r.Level()+1] {
			subvec(p1[j].Coeffs[i], p2[j-stpIdx].Coeffs[i], p3[j].Coeffs[i], uint32(s.Modulus))
		}
	}
}

func AddManyRingIdx_LowMem(p1, p2, p3 []uint32, stpIdx, inter int, P uint32) {
	l := len(p2)
	for j := stpIdx; j < stpIdx+l; j++ {
		addvec(p1[j:], p2, p3[j:], P)
	}
}

func SubManyRingIdx_LowMem(p1, p2, p3 []uint32, stpIdx, inter int, P uint32) {
	l := len(p2)
	for j := stpIdx; j < stpIdx+l; j++ {
		subvec(p1[j:], p2, p3[j:], P)
	}
}

func AddMany(cts1, cts2 []*rlwe.Ciphertext, evaluator *hefloat.Evaluator) []*rlwe.Ciphertext {
	res := make([]*rlwe.Ciphertext, len(cts1))

	for i := range res {
		if cts1[i] == nil && cts2[i] == nil {
			res[i] = nil
		} else if cts1[i] == nil {
			res[i] = cts2[i]
		} else if cts2[i] == nil {
			res[i] = cts1[i]
		} else {
			res[i], _ = evaluator.AddNew(cts1[i], cts2[i])
		}
	}
	return res
}

func SubMany(cts1, cts2 []*rlwe.Ciphertext, evaluator *hefloat.Evaluator) []*rlwe.Ciphertext {
	res := make([]*rlwe.Ciphertext, len(cts1))

	for i := range cts1 {
		if cts1[i] == nil && cts2[i] == nil {
			res[i] = nil
		} else if cts1[i] == nil {
			res[i], _ = evaluator.MulNew(cts2[i], -1)
		} else if cts2[i] == nil {
			res[i] = cts1[i]
		} else {
			res[i], _ = evaluator.SubNew(cts1[i], cts2[i])
		}
	}
	return res
}
