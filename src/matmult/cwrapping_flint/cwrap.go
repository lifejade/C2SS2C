package cwrappingflint

/*
#cgo pkg-config: flint
#cgo CFLAGS:  -O2 -g -fopenmp
#cgo LDFLAGS: -fopenmp -lpthread
#include <stdlib.h>
#include "flint_wrap.h"
*/
import "C"
import (
	"strconv"
)

func Mult_mod_mat(A, B [][]uint64, size uint64, p uint64) [][]uint64 {
	a := make([]C.ulonglong, size*size)
	b := make([]C.ulonglong, size*size)
	result := make([]C.ulonglong, size*size)
	for i := range size {
		for j := range size {
			a[i*size+j] = C.ulonglong(A[i][j])
			b[i*size+j] = C.ulonglong(B[i][j])
		}

	}
	C.multiply_mod_matrix_flint(&a[0], &b[0], &result[0], C.ulonglong(size), C.CString(strconv.FormatUint(p, 10)))
	res := make([][]uint64, size)
	for i := range size {
		res[i] = make([]uint64, size)
		for j := range size {
			res[i][j] = uint64(result[size*i+j])
		}
	}
	return res
}
