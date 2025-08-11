package matmult

import (
	"fmt"

	"github.com/lifejade/mm/src/matmult/cwrapping"
	cwrappingflint "github.com/lifejade/mm/src/matmult/cwrapping_flint"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func PPMM(cts []*rlwe.Ciphertext, u [][]uint64, params hefloat.Parameters, n int) []*rlwe.Ciphertext {
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
		CA[i] = cwrapping.Mult_mod_mat(u, a[i], uint64(n), params.RingQ().AtLevel(i).Modulus().Uint64())
		CB[i] = cwrapping.Mult_mod_mat(u, b[i], uint64(n), params.RingQ().AtLevel(i).Modulus().Uint64())
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
		CA[i] = cwrappingflint.Mult_mod_mat(u, a[i], uint64(n), params.RingQ().AtLevel(i).Modulus().Uint64())
		CB[i] = cwrappingflint.Mult_mod_mat(u, b[i], uint64(n), params.RingQ().AtLevel(i).Modulus().Uint64())
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
