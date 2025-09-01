package cwrappingflint

/*
#cgo pkg-config: flint
#cgo CFLAGS:  -O2 -g -fopenmp
#cgo LDFLAGS: -fopenmp -lpthread
#include <stdlib.h>
#include "flint_wrap.h"
*/
import "C"

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
