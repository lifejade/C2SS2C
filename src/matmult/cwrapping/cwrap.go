package cwrapping

/*
#cgo CXXFLAGS: -std=c++11
#cgo LDFLAGS: -L. -lwrapper -lntl -lgmp -lstdc++ -lpthread -lm
#include "wrapper.hpp"
*/
import "C"

import (
	"fmt"
	"strconv"
)

func Test() {

	n := 1 << 13
	var p uint64
	p = 7 // Z_7

	a := make([]C.ulonglong, n*n)
	b := make([]C.ulonglong, n*n)

	for i := range n * n {
		a[i] = 1
		b[i] = 3
	}

	result := make([]C.ulonglong, n*n)

	C.multiply_mod_matrix(&a[0], &b[0], &result[0], C.ulonglong(n), C.CString(strconv.FormatUint(p, 10)))

	fmt.Println("결과 행렬:")
	fmt.Println(result[0], result[4], result[168])
}

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
	C.multiply_mod_matrix(&a[0], &b[0], &result[0], C.ulonglong(size), C.CString(strconv.FormatUint(p, 10)))
	res := make([][]uint64, size)
	for i := range size {
		res[i] = make([]uint64, size)
		for j := range size {
			res[i][j] = uint64(result[size*i+j])
		}
	}

	return res
}
