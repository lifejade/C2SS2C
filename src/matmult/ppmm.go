package matmult

import (
	"fmt"
	"math"
	"math/cmplx"

	cwrappingflint "github.com/lifejade/mm/src/matmult/cwrapping_flint"
	"github.com/lifejade/mm/src/transpose"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/schemes/ckks"
)

func CtZero(params hefloat.Parameters, encoder *hefloat.Encoder, encryptor *rlwe.Encryptor) *rlwe.Ciphertext {
	value := make([]float64, params.MaxSlots())
	pt := rlwe.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	return ct
}

func bitReverse(i, m int) int {
	rev := 0
	for j := 0; j < m; j++ {
		rev = (rev << 1) | (i & 1)
		i >>= 1
	}
	return rev
}

func BitReversePermutationMatrix(n int) [][]complex128 {
	m := int(math.Log2(float64(n)))
	P := make([][]complex128, n)
	for i := range P {
		P[i] = make([]complex128, n)
	}

	for i := 0; i < n; i++ {
		rev := bitReverse(i, m)
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

	SFI_ := make([][][]complex128, logn)
	div := complex(math.Pow(float64(n), 1/float64(logn)), 0)
	//div := complex(1, 0)
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

				SFI_[idx][i+j][i+j] = 1 / div
				SFI_[idx][i+j][i+j+(m>>1)] = 1 / div

				SFI_[idx][i+j+(m>>1)][i+j] = cmplx.Conj(roots_complex[k] / div)
				SFI_[idx][i+j+(m>>1)][i+j+(m>>1)] = -cmplx.Conj(roots_complex[k] / div)

				pow5v *= 5
				pow5v = pow5v & ((m << 2) - 1)
			}
		}
	}

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

	fmt.Println(level)

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

	fmt.Println(level)

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
