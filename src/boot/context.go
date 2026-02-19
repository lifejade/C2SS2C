package boot

import (
	"fmt"
	"math/big"
	"time"

	"github.com/lifejade/mm/src/matmult"
	"github.com/lifejade/mm/src/util"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils/bignum"
)

type Context struct {
	params    hefloat.Parameters
	Encoder   *hefloat.Encoder
	Encryptor *rlwe.Encryptor
	Evaluator *hefloat.Evaluator

	N       int
	SparseN int
	P       []uint32

	C2SParams     *MatmultParams
	S2CParams     *MatmultParams
	EvalModParams *EvalModParams

	alloced *preAlloced
}

type ContextLiteral struct {
	params    hefloat.Parameters
	Encoder   *hefloat.Encoder
	Encryptor *rlwe.Encryptor
	Evaluator *hefloat.Evaluator

	N       int
	SparseN int
	P       []uint32
}

type EvalModParams struct {
	params EvalModParamsLiteral
}
type EvalModParamsLiteral struct {
	H int
}
type MatmultParams struct {
	params MatmultParamsLiteral
	mat0   [][][]float64
	mat0i  [][][]float64
	mat0s  [][][]float64
}

type MatmultParamsLiteral struct {
	StartLevel int
	PLevel     int
	EndLevel   int

	Scaling float64
	CL_arr  []int
}

type preAlloced struct {
	ringQ *ring.Ring
	ringP *matmult.Ring
	be    *matmult.BasisExtender

	cttemp *rlwe.Ciphertext

	work []*rlwe.Ciphertext
	aux  []*rlwe.Ciphertext

	inputPolys [][]matmult.Poly

	realPolys []matmult.Poly
	imagPolys []matmult.Poly
	tempPolys []matmult.Poly

	realbuffer []uint32
	imagbuffer []uint32
	tempbuffer []uint32

	ppmmbuffer1 []float64
	ppmmbuffer2 []float64

	resPolys [][]matmult.Poly
}

func InitContext(contextparams ContextLiteral, CTS, STC MatmultParamsLiteral, EvalMod EvalModParamsLiteral) (context *Context) {
	params := contextparams.params
	Encoder := contextparams.Encoder
	Encryptor := contextparams.Encryptor
	N := contextparams.N
	sparseN := contextparams.SparseN
	evaluator := contextparams.Evaluator
	P := contextparams.P

	context = &Context{
		params:    params,
		Encoder:   Encoder,
		Encryptor: Encryptor,
		N:         N,
		SparseN:   sparseN,
		Evaluator: evaluator,
		P:         P,
	}
	if util.Debug.IsDebug {
		fmt.Println("pre allocate start")
		util.Debug.StartTime = time.Now()
	}
	context.ContextPreAlloc()
	util.PrintMemUsage()
	SF, SFI := matmult.GenSFMat_CL2(params, sparseN>>1, STC.CL_arr, CTS.CL_arr)
	context.S2CParams = context.GenMatParams(STC, SF, true)
	context.C2SParams = context.GenMatParams(CTS, SFI, false)
	context.EvalModParams = context.GenEvalModParams(EvalMod)
	if util.Debug.IsDebug {
		elapse := time.Since(util.Debug.StartTime)
		fmt.Println("pre allocate end : ", elapse)
		util.PrintMemUsage()
	}
	return context
}

func (context *Context) GenEvalModParams(EvalMod EvalModParamsLiteral) *EvalModParams {

	return nil
}

func (context *Context) ContextPreAlloc() {
	params := context.params
	encoder := context.Encoder
	encryptor := context.Encryptor
	N := context.N
	sparseN := context.SparseN
	P := context.P
	ringQ := params.RingQ()
	ringP, _ := matmult.NewRing(N, P)

	inputPolys := make([][]matmult.Poly, 2)
	for i := range inputPolys {
		inputPolys[i] = make([]matmult.Poly, sparseN)
		for j := range inputPolys[i] {
			inputPolys[i][j] = ringP.NewPoly()
		}
	}
	resPolys := make([][]matmult.Poly, 2)
	for i := range 2 {
		resPolys[i] = make([]matmult.Poly, sparseN)
		for j := range sparseN {
			resPolys[i][j] = ringP.NewPoly()
		}
	}

	ctzero := util.CtZero(params, encoder, encryptor)
	var realPolys, imagPolys, tempPolys []matmult.Poly
	// var realbuffer, imagbuffer, tempbuffer []uint32
	realPolys = make([]matmult.Poly, sparseN/2)
	imagPolys = make([]matmult.Poly, sparseN/2)
	tempPolys = make([]matmult.Poly, sparseN/2)
	for j := range sparseN / 2 {
		realPolys[j] = ringP.NewPoly()
		imagPolys[j] = ringP.NewPoly()
		tempPolys[j] = ringP.NewPoly()
	}

	work := make([]*rlwe.Ciphertext, sparseN)
	for i := range work {
		work[i] = ctzero.CopyNew()
	}
	aux := make([]*rlwe.Ciphertext, sparseN)
	for i := range aux {
		aux[i] = ctzero.CopyNew()
	}

	alloc := new(preAlloced)
	alloc.ringQ = ringQ
	alloc.ringP = ringP
	alloc.be = matmult.NewBasisExtender(ringQ, ringP, nil, nil)
	alloc.inputPolys = inputPolys
	alloc.realPolys = realPolys
	alloc.imagPolys = imagPolys
	alloc.tempPolys = tempPolys
	alloc.resPolys = resPolys

	alloc.work = work
	alloc.aux = aux
	alloc.cttemp = ctzero

	context.alloced = alloc

}

func (context *Context) GenMatParams(param MatmultParamsLiteral, mat [][][]complex128, isSF bool) (matparams *MatmultParams) {
	if mat == nil {
		return
	}
	matparams = new(MatmultParams)

	C2Scaling := param.Scaling
	PLevel := param.PLevel
	CL_arr := param.CL_arr
	P := context.P[:PLevel+1]

	params := context.params
	// N := context.N
	sparseN := context.SparseN
	n := sparseN >> 1

	scaling_ := big.NewFloat(C2Scaling)
	// scaling_.Mul(scaling_, new(big.Float).SetFloat64(1/float64(sparseN)))
	if !isSF {
		scaling_.Mul(scaling_, new(big.Float).SetFloat64(1/float64(sparseN)))
	}

	scaling_ = bignum.Pow(scaling_, new(big.Float).Quo(new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(1), new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(float64(len(CL_arr)))))
	for i := range mat {
		for j := range mat[i] {
			for k := range mat[i][j] {
				val := new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(real(mat[i][j][k]))
				val = val.Mul(val, scaling_)
				val2 := new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(imag(mat[i][j][k]))
				val2 = val2.Mul(val2, scaling_)
				v1, _ := val.Float64()
				v2, _ := val2.Float64()
				scale := float64(params.Q()[param.EndLevel-i])
				mat[i][j][k] = complex(v1*scale, v2*scale)
			}
		}
	}

	mat0 := make([][][]float64, len(mat))
	mat0i := make([][][]float64, len(mat))
	var mat0s [][][]float64
	maxllen := 0

	if !isSF {
		inter_it := n
		for l := range mat {
			inter := inter_it >> CL_arr[l]
			llen := (1 << CL_arr[l])
			mat0[l] = make([][]float64, n/llen)
			mat0i[l] = make([][]float64, n/llen)
			if maxllen < llen {
				maxllen = llen
			}
			for t := range n / llen {
				mat0[l][t] = make([]float64, len(P)*llen*llen)
				mat0i[l][t] = make([]float64, len(P)*llen*llen)
				stpoint := inter_it*int(t/inter) + (t % inter)
				for q := range len(P) {
					for i := range llen {
						for j := range llen {
							idx := q*llen*llen + i*llen + j

							if real(mat[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
								mat0[l][t][idx] = float64(int64(real(mat[l][stpoint+inter*i][stpoint+inter*j])+0.5) % int64(P[q]))
							} else {
								mat0[l][t][idx] = float64(int64(P[q]) - (int64(-real(mat[l][stpoint+inter*i][stpoint+inter*j])+0.5) % int64(P[q])))
							}
							if imag(mat[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
								mat0i[l][t][idx] = float64(int64(imag(mat[l][stpoint+inter*i][stpoint+inter*j])+0.5) % int64(P[q]))
							} else {
								mat0i[l][t][idx] = float64(int64(P[q]) - (int64(-imag(mat[l][stpoint+inter*i][stpoint+inter*j])+0.5) % int64(P[q])))
							}
						}
					}
				}
			}

			inter_it = inter
		}

		// var tempPoly [][]ring.Poly
		if len(CL_arr) > 2 {
			mat0s = make([][][]float64, len(mat))
			inter_it := n
			for l := range mat {
				inter := inter_it >> CL_arr[l]

				llen := (1 << CL_arr[l])
				mat0s[l] = make([][]float64, n/llen)
				if maxllen < llen {
					maxllen = llen
				}
				for t := range n / llen {
					mat0s[l][t] = make([]float64, len(P)*llen*llen)
					for q := range len(P) {
						for i := range llen {
							for j := range llen {
								idx := q*llen*llen + i*llen + j
								mat0s[l][t][idx] = float64(uint32(mat0i[l][t][idx]+mat0[l][t][idx]+0.5) % P[q])
							}
						}
					}
				}

				inter_it = inter
			}
		}
	} else {
		inter_it := 1
		for l := range mat {
			inter := inter_it
			inter_it = inter << CL_arr[l]
			llen := (1 << CL_arr[l])
			mat0[l] = make([][]float64, n/llen)
			mat0i[l] = make([][]float64, n/llen)
			if maxllen < llen {
				maxllen = llen
			}
			for t := range n / llen {
				mat0[l][t] = make([]float64, len(P)*llen*llen)
				mat0i[l][t] = make([]float64, len(P)*llen*llen)
				stpoint := inter_it*int(t/inter) + (t % inter)
				for q := range len(P) {
					for i := range llen {
						for j := range llen {
							idx := q*llen*llen + i*llen + j

							if real(mat[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
								mat0[l][t][idx] = float64(int64(real(mat[l][stpoint+inter*i][stpoint+inter*j])+0.5) % int64(P[q]))
							} else {
								mat0[l][t][idx] = float64(int64(P[q]) - (int64(-real(mat[l][stpoint+inter*i][stpoint+inter*j])+0.5) % int64(P[q])))
							}
							if imag(mat[l][stpoint+inter*i][stpoint+inter*j]) >= 0 {
								mat0i[l][t][idx] = float64(int64(imag(mat[l][stpoint+inter*i][stpoint+inter*j])+0.5) % int64(P[q]))
							} else {
								mat0i[l][t][idx] = float64(int64(P[q]) - (int64(-imag(mat[l][stpoint+inter*i][stpoint+inter*j])+0.5) % int64(P[q])))
							}
						}
					}
				}
			}

		}

		if len(CL_arr) > 2 {
			mat0s = make([][][]float64, len(mat))
			inter_it := 1
			for l := range mat {
				inter := inter_it
				inter_it = inter << CL_arr[l]
				llen := (1 << CL_arr[l])
				mat0s[l] = make([][]float64, n/llen)
				if maxllen < llen {
					maxllen = llen
				}
				for t := range n / llen {
					mat0s[l][t] = make([]float64, len(P)*llen*llen)
					for q := range len(P) {
						for i := range llen {
							for j := range llen {
								idx := q*llen*llen + i*llen + j
								mat0s[l][t][idx] = float64(uint32(mat0i[l][t][idx]+mat0[l][t][idx]+0.5) % P[q])
							}
						}
					}
				}
			}
		}
	}

	context.alloced.be.AddSwitchDic([]matmult.Key{{From: param.StartLevel, To: PLevel}}, []matmult.Key{{From: PLevel, To: param.EndLevel}})
	matparams.mat0 = mat0
	matparams.mat0i = mat0i
	matparams.mat0s = mat0s
	matparams.params = param

	N := context.N

	if len(context.alloced.ppmmbuffer1) < len(P)*maxllen*N {
		context.alloced.ppmmbuffer1 = make([]float64, maxllen*N)
		context.alloced.ppmmbuffer2 = make([]float64, maxllen*N)
		// fmt.Println("ppmmbuffer len : ", len(context.alloced.ppmmbuffer1))
	}

	return
}

func (context *Context) GenMatParams2(param MatmultParamsLiteral, mat [][]complex128) (matparams *MatmultParams) {
	if mat == nil {
		return
	}
	matparams = new(MatmultParams)

	C2Scaling := param.Scaling
	PLevel := param.PLevel
	CL_arr := param.CL_arr
	P := context.P[:PLevel+1]

	params := context.params
	// N := context.N
	sparseN := context.SparseN
	n := sparseN >> 1

	scaling_ := big.NewFloat(C2Scaling)
	scaling_.Mul(scaling_, new(big.Float).SetFloat64(1/float64(sparseN)))
	scaling_.Mul(scaling_, new(big.Float).SetFloat64(1/float64(sparseN)))
	scaling_ = bignum.Pow(scaling_, new(big.Float).Quo(new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(1), new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(float64(len(CL_arr)))))

	for i := range mat {
		for j := range n {
			for k := range n {
				val := new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(real(mat[i][j+k*n]))
				val = val.Mul(val, scaling_)
				val2 := new(big.Float).SetPrec(params.EncodingPrecision()).SetFloat64(imag(mat[i][j+k*n]))
				val2 = val2.Mul(val2, scaling_)
				v1, _ := val.Float64()
				v2, _ := val2.Float64()
				scale := float64(params.Q()[param.EndLevel-i])
				mat[i][j+k*n] = complex(v1*scale, v2*scale)
			}
		}
	}

	mat0 := make([][][]float64, len(mat))
	mat0i := make([][][]float64, len(mat))
	maxllen := 0
	inter_it := n
	for l := range mat {
		inter := inter_it >> CL_arr[l]

		llen := (1 << CL_arr[l])
		mat0[l] = make([][]float64, n/llen)
		mat0i[l] = make([][]float64, n/llen)
		if maxllen < llen {
			maxllen = llen
		}
		for t := range n / llen {
			mat0[l][t] = make([]float64, len(P)*llen*llen)
			mat0i[l][t] = make([]float64, len(P)*llen*llen)
			stpoint := inter_it*int(t/inter) + (t % inter)
			for q := range len(P) {
				for i := range llen {
					for j := range llen {
						idx := q*llen*llen + i*llen + j
						x := stpoint + inter*i
						y := stpoint + inter*j
						if real(mat[l][x+y*n]) >= 0 {
							mat0[l][t][idx] = float64(int64(real(mat[l][x+y*n])+0.5) % int64(P[q]))
						} else {
							mat0[l][t][idx] = float64(int64(P[q]) - (int64(-real(mat[l][x+y*n])+0.5) % int64(P[q])))
						}
						if imag(mat[l][x+y*n]) >= 0 {
							mat0i[l][t][idx] = float64(int64(imag(mat[l][x+y*n])+0.5) % int64(P[q]))
						} else {
							mat0i[l][t][idx] = float64(int64(P[q]) - (int64(-imag(mat[l][x+y*n])+0.5) % int64(P[q])))
						}
					}
				}
			}
		}

		inter_it = inter
	}

	var mat0s [][][]float64
	// var tempPoly [][]ring.Poly
	if len(CL_arr) > 2 {
		mat0s = make([][][]float64, len(mat))
		inter_it := n
		for l := range mat {
			inter := inter_it >> CL_arr[l]

			llen := (1 << CL_arr[l])
			mat0s[l] = make([][]float64, n/llen)
			if maxllen < llen {
				maxllen = llen
			}
			for t := range n / llen {
				mat0s[l][t] = make([]float64, len(P)*llen*llen)
				for q := range len(P) {
					for i := range llen {
						for j := range llen {
							idx := q*llen*llen + i*llen + j
							mat0s[l][t][idx] = float64(uint32(mat0i[l][t][idx]+mat0[l][t][idx]+0.5) % P[q])
						}
					}
				}
			}

			inter_it = inter
		}

		// if context.alloced.tempPoly == nil {
		// 	tempPoly = make([][]ring.Poly, 2)
		// 	for i := range tempPoly {
		// 		tempPoly[i] = make([]ring.Poly, sparseN)
		// 		for j := range tempPoly[i] {
		// 			tempPoly[i][j] = context.alloced.ringP.NewPoly()
		// 		}
		// 	}
		// 	context.alloced.tempPoly = tempPoly
		// }
	}
	context.alloced.be.AddSwitchDic([]matmult.Key{{From: param.StartLevel, To: PLevel}}, []matmult.Key{{From: PLevel, To: param.EndLevel}})
	matparams.mat0 = mat0
	matparams.mat0i = mat0i
	matparams.mat0s = mat0s
	matparams.params = param

	N := context.N

	if len(context.alloced.ppmmbuffer1) < len(P)*maxllen*N {
		context.alloced.ppmmbuffer1 = make([]float64, len(P)*maxllen*N)
		context.alloced.ppmmbuffer2 = make([]float64, len(P)*maxllen*N)
		// fmt.Println("ppmmbuffer len : ", len(context.alloced.ppmmbuffer1))
	}

	return
}
