package cwrappingflint

/*
#cgo pkg-config: flint, openblas
#cgo CFLAGS:  -O2 -g -fopenmp
#cgo LDFLAGS: -fopenmp -lpthread
#include <stdlib.h>
#include "flint_wrap.h"
*/
import "C"

import (
	"unsafe"
)

// 행렬 곱셈 결과를 효율적으로 계산하는 헬퍼 함수
func ComputeCombinedMat(n, startIdx, count int, roots []complex128, div complex128, isInverse bool) []complex128 {
	res := make([]complex128, n*n)

	var inv C.int
	if isInverse {
		inv = 1
	}

	C.computeCombinedMat_blas(
		C.int(n),
		C.int(startIdx),
		C.int(count),
		(*C.dcomplex)(unsafe.Pointer(&roots[0])),
		(C.dcomplex)(div),
		inv,
		(*C.dcomplex)(unsafe.Pointer(&res[0])),
	)

	return res // column-major: res[r + c*n]
}

func Mult_mod_mat(A, B [][]uint64, size_a, size_b, size_c int, p uint64) [][]uint64 {
	a := make([]C.ulonglong, size_a*size_b)
	b := make([]C.ulonglong, size_b*size_c)
	result := make([]C.ulonglong, size_a*size_c)
	for i := range size_a {
		for j := range size_b {
			a[i*size_b+j] = C.ulonglong(A[i][j])
		}
	}
	for i := range size_b {
		for j := range size_c {
			b[i*size_c+j] = C.ulonglong(B[i][j])
		}
	}

	C.multiply_mod_matrix_flint(&a[0], &b[0], &result[0], C.ulonglong(size_a), C.ulonglong(size_b), C.ulonglong(size_c), C.ulonglong(p))
	res := make([][]uint64, size_a)
	for i := range size_a {
		res[i] = make([]uint64, size_c)
		for j := range size_c {
			res[i][j] = uint64(result[size_c*i+j])
		}
	}
	return res
}

func Mult_mod_mat2(A []uint64, B [][]uint64, size_a, size_b, size_c int, p uint64) [][]uint64 {
	a := make([]C.ulonglong, size_a*size_b)
	b := make([]C.ulonglong, size_b*size_c)
	result := make([]C.ulonglong, size_a*size_c)
	for i := range size_a {
		for j := range size_b {
			a[i*size_b+j] = C.ulonglong(A[i+size_b+j])
		}
	}
	for i := range size_b {
		for j := range size_c {
			b[i*size_c+j] = C.ulonglong(B[i][j])
		}
	}

	C.multiply_mod_matrix_flint(&a[0], &b[0], &result[0], C.ulonglong(size_a), C.ulonglong(size_b), C.ulonglong(size_c), C.ulonglong(p))
	res := make([][]uint64, size_a)
	for i := range size_a {
		res[i] = make([]uint64, size_c)
		for j := range size_c {
			res[i][j] = uint64(result[size_c*i+j])
		}
	}
	return res
}

func Mult_mod_mat_Blas(A, B [][]uint64, size_a, size_b, size_c int, p uint64) [][]uint64 {
	// --- 평탄화 (uint64 -> float64) ---
	a64 := make([]float64, size_a*size_b)
	for i := 0; i < size_a; i++ {
		row := A[i]
		base := i * size_b
		for j := 0; j < size_b; j++ {
			a64[base+j] = float64(row[j])
		}
	}

	b64 := make([]float64, size_b*size_c)
	for i := 0; i < size_b; i++ {
		row := B[i]
		base := i * size_c
		for j := 0; j < size_c; j++ {
			b64[base+j] = float64(row[j])
		}
	}

	// --- 결과 버퍼는 Go에서 잡고 포인터만 넘김 ---
	// C가 결과를 out에 써줌
	resultFlat := make([]uint64, size_a*size_c)

	// --- C 호출 (포인터 캐스팅만) ---
	C.multiply_mod_matrix_blas(
		(*C.double)(unsafe.Pointer(&a64[0])),
		(*C.double)(unsafe.Pointer(&b64[0])),
		(*C.ulonglong)(unsafe.Pointer(&resultFlat[0])),
		C.ulonglong(size_a),
		C.ulonglong(size_b),
		C.ulonglong(size_c),
		C.ulonglong(p),
	)

	// --- 2D로 되돌리기 ---
	res := make([][]uint64, size_a)
	for i := 0; i < size_a; i++ {
		row := make([]uint64, size_c)
		copy(row, resultFlat[i*size_c:(i+1)*size_c])
		res[i] = row
	}
	return res
}

func Mult_mod_mat_Blas_Inplace(A, B, res []float64, size_a, size_b, size_c, level int) {
	// --- 평탄화 (uint64 -> float64) ---
	// --- 결과 버퍼는 Go에서 잡고 포인터만 넘김 ---
	// C가 결과를 out에 써줌

	// --- C 호출 (포인터 캐스팅만) ---
	C.multiply_mod_matrix_blas_Inplace(
		(*C.double)(unsafe.Pointer(&A[0])),
		(*C.double)(unsafe.Pointer(&B[0])),
		(*C.double)(unsafe.Pointer(&res[0])),
		C.uint(size_a),
		C.uint(size_b),
		C.uint(size_c),
		C.uint(level),
	)
}

func Mult_mod_mat_Blas_Inplace2(A, B, res []float64, size_a, size_b, size_c int) {
	// --- 평탄화 (uint64 -> float64) ---
	// --- 결과 버퍼는 Go에서 잡고 포인터만 넘김 ---
	// C가 결과를 out에 써줌

	// --- C 호출 (포인터 캐스팅만) ---
	C.multiply_mod_matrix_blas_Inplace2(
		(*C.double)(unsafe.Pointer(&A[0])),
		(*C.double)(unsafe.Pointer(&B[0])),
		(*C.double)(unsafe.Pointer(&res[0])),
		C.uint(size_a),
		C.uint(size_b),
		C.uint(size_c),
	)
}

func Mult_mod_mat_Blas_Inplace_Stride(A, B, res []float64, size_a, size_b, size_c, level, lda, ldb, ldc int) {
	// --- 평탄화 (uint64 -> float64) ---
	// --- 결과 버퍼는 Go에서 잡고 포인터만 넘김 ---
	// C가 결과를 out에 써줌

	// --- C 호출 (포인터 캐스팅만) ---
	C.multiply_mod_matrix_blas_Inplace_Stride(
		(*C.double)(unsafe.Pointer(&A[0])),
		(*C.double)(unsafe.Pointer(&B[0])),
		(*C.double)(unsafe.Pointer(&res[0])),
		C.uint(size_a),
		C.uint(size_b),
		C.uint(size_c),
		C.uint(level),
		C.uint(lda),
		C.uint(ldb),
		C.uint(ldc),
	)
}

func Mult_mod_mat_Blas_ForTest(A, B []float64, res []uint64, size_a, size_b, size_c int, p uint64) {
	C.multiply_mod_matrix_blas(
		(*C.double)(unsafe.Pointer(&A[0])),
		(*C.double)(unsafe.Pointer(&B[0])),
		(*C.ulonglong)(unsafe.Pointer(&res[0])),
		C.ulonglong(size_a),
		C.ulonglong(size_b),
		C.ulonglong(size_c),
		C.ulonglong(p),
	)
}

func Mult_mod_mat_BlasBarret(A, B [][]uint64, size_a, size_b, size_c int, p uint64, bred uint64) [][]uint64 {
	// --- 평탄화 (uint64 -> float64) ---
	a64 := make([]float64, size_a*size_b)
	for i := 0; i < size_a; i++ {
		row := A[i]
		base := i * size_b
		for j := 0; j < size_b; j++ {
			a64[base+j] = float64(row[j])
		}
	}

	b64 := make([]float64, size_b*size_c)
	for i := 0; i < size_b; i++ {
		row := B[i]
		base := i * size_c
		for j := 0; j < size_c; j++ {
			b64[base+j] = float64(row[j])
		}
	}

	// --- 결과 버퍼는 Go에서 잡고 포인터만 넘김 ---
	// C가 결과를 out에 써줌
	resultFlat := make([]uint64, size_a*size_c)

	// --- C 호출 (포인터 캐스팅만) ---
	C.multiply_mod_matrix_blas2(
		(*C.double)(unsafe.Pointer(&a64[0])),
		(*C.double)(unsafe.Pointer(&b64[0])),
		(*C.ulonglong)(unsafe.Pointer(&resultFlat[0])),
		C.ulonglong(size_a),
		C.ulonglong(size_b),
		C.ulonglong(size_c),
		C.ulonglong(p),
		C.ulonglong(bred),
	)

	// --- 2D로 되돌리기 ---
	res := make([][]uint64, size_a)
	for i := 0; i < size_a; i++ {
		row := make([]uint64, size_c)
		copy(row, resultFlat[i*size_c:(i+1)*size_c])
		res[i] = row
	}
	return res
}

func Mult_mat_Blas_RoundMod(A, B [][]float64, size_a, size_b, size_c int, p uint64, bredP uint64) [][]float64 {
	// --- 평탄화 (uint64 -> float64) ---
	a64 := make([]float64, size_a*size_b)
	for i := 0; i < size_a; i++ {
		row := A[i]
		base := i * size_b
		for j := 0; j < size_b; j++ {
			a64[base+j] = (row[j])
		}
	}

	b64 := make([]float64, size_b*size_c)
	for i := 0; i < size_b; i++ {
		row := B[i]
		base := i * size_c
		for j := 0; j < size_c; j++ {
			b64[base+j] = (row[j])
		}
	}

	// --- 결과 버퍼는 Go에서 잡고 포인터만 넘김 ---
	// C가 결과를 out에 써줌
	resultFlat := make([]float64, size_a*size_c)
	fillfast(resultFlat, 0.5)

	// --- C 호출 (포인터 캐스팅만) ---
	C.multiply_mod_matrix_blas2(
		(*C.double)(unsafe.Pointer(&a64[0])),
		(*C.double)(unsafe.Pointer(&b64[0])),
		(*C.ulonglong)(unsafe.Pointer(&resultFlat[0])),
		C.ulonglong(size_a),
		C.ulonglong(size_b),
		C.ulonglong(size_c),
		C.ulonglong(p),
		C.ulonglong(bredP),
	)

	// --- 2D로 되돌리기 ---
	res := make([][]float64, size_a)
	for i := 0; i < size_a; i++ {
		row := make([]float64, size_c)
		copy(row, resultFlat[i*size_c:(i+1)*size_c])
		res[i] = row
	}
	return res
}
func fillfast(s []float64, v float64) {
	if len(s) == 0 {
		return
	}
	s[0] = v
	for bp := 1; bp < len(s); bp *= 2 {
		copy(s[bp:], s[:bp])
	}
}
