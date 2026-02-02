package boot

import (
	"fmt"
	"math/bits"
	"time"

	"github.com/lifejade/mm/src/matmult"
	"github.com/lifejade/mm/src/transpose"
	"github.com/lifejade/mm/src/util"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

func ModUpQtoQ(ringQ *ring.Ring, levelQ int, polQIn, polQOut ring.Poly) {
	BRCQ := ringQ.BRedConstants()
	Q := ringQ.ModuliChain()
	q := Q[0]
	// levelQ := len(Q) - 1
	// levelQ := 0
	QHalf := q >> 1
	N := ringQ.N()
	// polQOut.Resize(levelQ)
	var pos, neg, tmp uint64
	for j := 0; j < N; j++ {
		coeff := polQIn.Coeffs[0][j]
		pos, neg = 1, 0
		if coeff >= QHalf {
			coeff = q - coeff
			pos, neg = 0, 1
		}

		for i := 0; i < levelQ+1; i++ {
			tmp = ring.BRedAdd(coeff, Q[i], BRCQ[i])
			polQOut.Coeffs[i][j] = tmp*pos + (Q[i]-tmp)*neg
		}

	}
}

// input INTT, output INTT
func ModUp(cts []*rlwe.Ciphertext, params hefloat.Parameters, encoder *hefloat.Encoder, encryptor *rlwe.Encryptor, eval *hefloat.Evaluator, N, sparseN, H int, rescts []*rlwe.Ciphertext) {
	ringQ := params.RingQ().AtLevel(params.MaxLevel())
	ratio := N / sparseN
	defer func() {
		if util.Debug.IsDebug {
			util.Debug.StartTime = time.Time{}
			util.Debug.AccTime = 0
		}
	}()

	for i := range cts {
		if cts[i].Level() != 0 {
			eval.DropLevel(cts[i], cts[i].Level())
		}
	}
	ctsj := util.CtZero(params, encoder, encryptor)

	logratio := bits.Len(uint(N/sparseN)) - 1

	logN := params.LogN()
	var gks []*rlwe.GaloisKey
	var err error

	if util.Debug.IsDebug {
		util.Debug.StartTime = time.Now()
		fmt.Println("gk init start")
	}
	gks = make([]*rlwe.GaloisKey, logratio)
	for j := range logratio {
		var gk *rlwe.GaloisKey
		galEl := uint64(1<<(logN-j) + 1)
		if gk, err = eval.CheckAndGetGaloisKey(galEl); err != nil {
			if util.SContext.Sk == nil && !util.Debug.IsDebug {
				panic(err)
			}
			kgen_ := rlwe.NewKeyGenerator(params)
			gk = kgen_.GenGaloisKeyNew(galEl, util.SContext.Sk)
		}
		gks[j] = gk
	}
	var dtsk, stdk *rlwe.EvaluationKey
	if H > 0 && util.SContext.Sk != nil {
		paramsSparse, _ := rlwe.NewParametersFromLiteral(rlwe.ParametersLiteral{
			LogN: params.LogN(),
			Q:    params.Q()[:1],
			P:    params.P()[:1],
		})

		kgenSparse := rlwe.NewKeyGenerator(paramsSparse)
		kgenDense := rlwe.NewKeyGenerator(params)
		skSparse := kgenSparse.GenSecretKeyWithHammingWeightNew(H)

		dtsk = kgenDense.GenEvaluationKeyNew(util.SContext.Sk, skSparse)
		stdk = kgenDense.GenEvaluationKeyNew(skSparse, util.SContext.Sk)
	}

	if util.Debug.IsDebug {
		elapse := time.Since(util.Debug.StartTime)
		fmt.Println("gk init end", elapse)
		util.Debug.StartTime = time.Now()
	}

	if H > 0 {
		for i := range cts {
			cts[i].IsNTT = false
			eval.ApplyEvaluationKey(cts[i], dtsk, rescts[i])
			cts[i].IsNTT = true
			rescts[i].IsNTT = true
		}
	} else {
		copy(rescts, cts)
	}

	if util.Debug.IsDebug && H > 0 {
		elapse := time.Since(util.Debug.StartTime)
		fmt.Println("Dense To Sparse End : ", elapse)
		util.Debug.AccTime += elapse
		util.Debug.StartTime = time.Now()
	}

	for i := range cts {
		rescts[i].Resize(1, params.MaxLevel())
		ModUpQtoQ(ringQ, params.MaxLevel(), rescts[i].Value[0], rescts[i].Value[0])
		ModUpQtoQ(ringQ, params.MaxLevel(), rescts[i].Value[1], rescts[i].Value[1])
	}
	if H > 0 {
		for i := range cts {
			rescts[i].IsNTT = false
			eval.ApplyEvaluationKey(rescts[i], stdk, rescts[i])
			rescts[i].IsNTT = true
		}
	}
	if util.Debug.IsDebug && H > 0 {
		elapse := time.Since(util.Debug.StartTime)
		util.Debug.AccTime += elapse
		fmt.Println("Sparse To Dense End : ", util.Debug.AccTime)
		util.Debug.StartTime = time.Now()
	}
	for i := range rescts {
		ninv := ringQ.NewRNSScalarFromUInt64(uint64(ratio))
		ringQ.MFormRNSScalar(ninv, ninv)
		ringQ.Inverse(ninv)

		ringQ.MForm(rescts[i].Value[0], rescts[i].Value[0])
		ringQ.MForm(rescts[i].Value[1], rescts[i].Value[1])

		ringQ.MulRNSScalarMontgomery(rescts[i].Value[0], ninv, rescts[i].Value[0])
		ringQ.MulRNSScalarMontgomery(rescts[i].Value[1], ninv, rescts[i].Value[1])

		ringQ.IMForm(rescts[i].Value[0], rescts[i].Value[0])
		ringQ.IMForm(rescts[i].Value[1], rescts[i].Value[1])

		for j := range logratio {
			galEl := uint64(1<<(logN-j) + 1)
			rescts[i].IsNTT = false
			transpose.Automorphism(eval, ringQ, rescts[i], (galEl), gks[j], ctsj)
			rescts[i].IsNTT = true
			ctsj.IsNTT = true
			eval.Add(rescts[i], ctsj, rescts[i])
		}
	}

	if util.Debug.IsDebug {
		elapse := time.Since(util.Debug.StartTime)
		util.Debug.AccTime += elapse
		fmt.Println("PartialSum End : ", elapse)
		fmt.Println("ModUp End : ", util.Debug.AccTime)
	}

}

// // input must be INTT
// func (context *Context) CoeffToSlot(cts []*rlwe.Ciphertext) (result, result2 []*rlwe.Ciphertext) {
// 	N := context.N
// 	sparseN := context.SparseN
// 	n := sparseN >> 1

// 	params := context.params
// 	evaluator := context.Evaluator

// 	ringQ := context.alloced.ringQ
// 	ringP := context.alloced.ringP
// 	be := context.alloced.be
// 	inputPolys := context.alloced.inputPolys
// 	inputPolysC := context.alloced.inputPolysC
// 	ppmmbuffer1 := context.alloced.ppmmbuffer1
// 	ppmmbuffer2 := context.alloced.ppmmbuffer2
// 	resPolys00 := context.alloced.resPolys00
// 	resPolys00i := context.alloced.resPolys00i
// 	resPolys01 := context.alloced.resPolys01
// 	resPolys01i := context.alloced.resPolys01i
// 	resPolys10 := context.alloced.resPolys10
// 	resPolys10i := context.alloced.resPolys10i
// 	resPolys11 := context.alloced.resPolys11
// 	resPolys11i := context.alloced.resPolys11i
// 	tempPoly := context.alloced.tempPoly
// 	work := context.alloced.work
// 	aux := context.alloced.aux

// 	C2SParams := context.C2SParams
// 	CL_arr := C2SParams.params.CL_arr
// 	PLevel := C2SParams.params.PLevel
// 	mat0 := C2SParams.mat0
// 	mat0i := C2SParams.mat0i
// 	mat0s := C2SParams.mat0s

// 	// fmt.Println(tempPoly[0][0].Level())
// 	// fmt.Println(resPolys00[0][0].Level())

// 	ctzero := context.alloced.ctzero
// 	result = make([]*rlwe.Ciphertext, sparseN)
// 	result2 = make([]*rlwe.Ciphertext, sparseN)
// 	for i := range sparseN {
// 		result[i] = ctzero.CopyNew()
// 		result2[i] = ctzero.CopyNew()
// 	}

// 	if util.Debug.IsDebug {
// 		fmt.Println("start c2s")
// 		util.Debug.StartTime = time.Now()
// 	}

// 	for i := range sparseN {
// 		for d := range 2 {
// 			be.ModSwitchQtoP(cts[i].Level(), PLevel, cts[i].Value[d], inputPolys[d][i])
// 		}
// 		if i < n {
// 			ringQ.AtLevel(cts[i].Level()).Neg(cts[i+n].Value[0], work[0].Value[0])
// 			ringQ.AtLevel(cts[i].Level()).Neg(cts[i+n].Value[1], work[0].Value[1])
// 		} else {
// 			work[0] = cts[i-n]
// 		}
// 		for d := range 2 {
// 			be.ModSwitchQtoP(cts[i].Level(), PLevel, work[0].Value[d], inputPolysC[d][i])
// 		}
// 	}
// 	work[0] = ctzero.CopyNew()

// 	inter_it := n
// 	for l := range len(CL_arr) {
// 		inter := inter_it >> CL_arr[l]
// 		llen := (1 << CL_arr[l])
// 		if l == 0 {
// 			for t := range n / llen {
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

// 		} else if l == len(CL_arr)-1 {
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

// 		} else {
// 			matmult.AddManyRing(ringP, resPolys00, resPolys00i, tempPoly)
// 			for t := range n / llen {
// 				stpoint := (t % inter) + inter_it*int(t/inter)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys00, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys00i, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00i, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(tempPoly, mat0s[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, tempPoly, ppmmbuffer1, ppmmbuffer2)
// 			}
// 			matmult.SubManyRing(ringP, tempPoly, resPolys00, tempPoly)
// 			matmult.SubManyRing(ringP, resPolys00, resPolys00i, resPolys00)
// 			matmult.SubManyRing(ringP, tempPoly, resPolys00i, resPolys00i)

// 			matmult.AddManyRing(ringP, resPolys01, resPolys01i, tempPoly)
// 			for t := range n / llen {
// 				stpoint := (t % inter) + inter_it*int(t/inter)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys01, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys01i, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys01i, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(tempPoly, mat0s[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, tempPoly, ppmmbuffer1, ppmmbuffer2)
// 			}
// 			matmult.SubManyRing(ringP, tempPoly, resPolys01, tempPoly)
// 			matmult.SubManyRing(ringP, resPolys01, resPolys01i, resPolys01)
// 			matmult.SubManyRing(ringP, tempPoly, resPolys01i, resPolys01i)

// 			matmult.AddManyRing(ringP, resPolys10, resPolys10i, tempPoly)
// 			for t := range n / llen {
// 				stpoint := (t % inter) + inter_it*int(t/inter)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys10, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys10i, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys10i, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(tempPoly, mat0s[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, tempPoly, ppmmbuffer1, ppmmbuffer2)
// 			}
// 			matmult.SubManyRing(ringP, tempPoly, resPolys10, tempPoly)
// 			matmult.SubManyRing(ringP, resPolys10, resPolys10i, resPolys10)
// 			matmult.SubManyRing(ringP, tempPoly, resPolys10i, resPolys10i)

// 			matmult.AddManyRing(ringP, resPolys11, resPolys11i, tempPoly)
// 			for t := range n / llen {
// 				stpoint := (t % inter) + inter_it*int(t/inter)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys11, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys11i, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11i, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(tempPoly, mat0s[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, tempPoly, ppmmbuffer1, ppmmbuffer2)
// 			}
// 			matmult.SubManyRing(ringP, tempPoly, resPolys11, tempPoly)
// 			matmult.SubManyRing(ringP, resPolys11, resPolys11i, resPolys11)
// 			matmult.SubManyRing(ringP, tempPoly, resPolys11i, resPolys11i)

// 		}
// 		inter_it = inter
// 	}
// 	matmult.AddManyRing(ringP, resPolys00, resPolys01i, resPolys00)
// 	matmult.AddManyRing(ringP, resPolys10, resPolys11i, resPolys10)

// 	bitlen := bits.Len64(uint64(n)) - 1
// 	for d := range 2 {
// 		for i := range sparseN {
// 			if i < n {
// 				br := util.BitReverse(i, bitlen)
// 				if i >= br {
// 					continue
// 				}
// 				resPolys00[d][i], resPolys00[d][util.BitReverse(i, bitlen)] = resPolys00[d][util.BitReverse(i, bitlen)], resPolys00[d][i]
// 				resPolys10[d][i], resPolys10[d][util.BitReverse(i, bitlen)] = resPolys10[d][util.BitReverse(i, bitlen)], resPolys10[d][i]
// 			} else {
// 				i := i % n
// 				br := util.BitReverse(i, bitlen)
// 				if i >= br {
// 					continue
// 				}
// 				resPolys00[d][n+i], resPolys00[d][n+util.BitReverse(i, bitlen)] = resPolys00[d][n+util.BitReverse(i, bitlen)], resPolys00[d][n+i]
// 				resPolys10[d][n+i], resPolys10[d][n+util.BitReverse(i, bitlen)] = resPolys10[d][n+util.BitReverse(i, bitlen)], resPolys10[d][n+i]
// 			}
// 		}
// 	}

// 	q := rlwe.NewScale(params.DefaultScale())
// 	for i := range len(CL_arr) {
// 		q = q.Mul(rlwe.NewScale(params.Q()[C2SParams.params.EndLevel-i]))
// 	}

// 	for i := range sparseN {
// 		result[i].Resize(1, C2SParams.params.EndLevel)
// 		result2[i].Resize(1, C2SParams.params.EndLevel)
// 		for idx := range 2 {
// 			be.ModSwitchPtoQ(PLevel, C2SParams.params.EndLevel, resPolys00[idx][i], result[i].Value[idx])
// 			be.ModSwitchPtoQ(PLevel, C2SParams.params.EndLevel, resPolys10[idx][i], result2[i].Value[idx])
// 		}

// 		result[i].Scale = q
// 		result2[i].Scale = q
// 		util.Rescale_NonNTT(evaluator, result[i], result[i])
// 		util.Rescale_NonNTT(evaluator, result2[i], result2[i])
// 	}
// 	// fmt.Println(result[0].LogScale())
// 	if util.Debug.IsDebug {
// 		elapse := time.Since(util.Debug.StartTime)
// 		fmt.Println("CTS-PPMM End : ", elapse)
// 		util.Debug.AccTime += elapse
// 		fmt.Println("CTS-CMT Start")
// 		util.Debug.StartTime = time.Now()
// 	}

// 	transpose.Transpose_Sparse(result, params, evaluator, ringQ.AtLevel(result[0].Level()), N, sparseN, work, aux, result)
// 	transpose.Transpose_Sparse(result2, params, evaluator, ringQ.AtLevel(result2[0].Level()), N, sparseN, work, aux, result2)

// 	if util.Debug.IsDebug {
// 		elapse := time.Since(util.Debug.StartTime)
// 		fmt.Println("CTS-CMT End: ", elapse)
// 		util.Debug.AccTime += elapse
// 		util.Debug.StartTime = time.Now()
// 	}

// 	for i := range result {

// 		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[0], result[i].Value[0])
// 		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[1], result[i].Value[1])

// 		conj, _ := evaluator.ConjugateNew(result[i])
// 		evaluator.Add(result[i], conj, result[i])

// 		ringQ.AtLevel(result2[i].Level()).NTT(result2[i].Value[0], result2[i].Value[0])
// 		ringQ.AtLevel(result2[i].Level()).NTT(result2[i].Value[1], result2[i].Value[1])

// 		conj, _ = evaluator.ConjugateNew(result2[i])
// 		evaluator.Add(result2[i], conj, result2[i])
// 	}

// 	if util.Debug.IsDebug {
// 		elapse := time.Since(util.Debug.StartTime)
// 		fmt.Println("end c2s : ", elapse)
// 		util.Debug.AccTime += elapse
// 	}

// 	return
// }

// input must be INTT
func (context *Context) CoeffToSlot2(cts []*rlwe.Ciphertext) (result, result2 []*rlwe.Ciphertext) {
	N := context.N
	sparseN := context.SparseN
	n := sparseN >> 1
	params := context.params
	evaluator := context.Evaluator

	ringQ := context.alloced.ringQ
	be := context.alloced.be
	inputPolys := context.alloced.inputPolys
	resPolys := context.alloced.resPolys
	work := context.alloced.work
	aux := context.alloced.aux

	C2SParams := context.C2SParams
	CL_arr := C2SParams.params.CL_arr
	PLevel := C2SParams.params.PLevel

	// fmt.Println(tempPoly[0][0].Level())
	// fmt.Println(resPolys00[0][0].Level())

	ctzero := context.alloced.ctzero
	result = make([]*rlwe.Ciphertext, sparseN)
	result2 = make([]*rlwe.Ciphertext, sparseN)
	for i := range sparseN {
		result[i] = ctzero.CopyNew()
		result2[i] = ctzero.CopyNew()
	}
	for i := range sparseN {
		for d := range 2 {
			be.ModSwitchQtoP(cts[i].Level(), PLevel, cts[i].Value[d], inputPolys[d][i])
		}
	}

	if util.Debug.IsDebug {
		fmt.Println("start c2s")
		util.Debug.StartTime = time.Now()
	}

	context.CTS_PCMM(inputPolys, resPolys)

	q := rlwe.NewScale(params.DefaultScale())
	for i := range len(CL_arr) {
		q = q.Mul(rlwe.NewScale(params.Q()[C2SParams.params.EndLevel-i]))
	}

	for i := range sparseN {
		result[i].Resize(1, C2SParams.params.EndLevel)
		for idx := range 2 {
			be.ModSwitchPtoQ(PLevel, C2SParams.params.EndLevel, resPolys[idx][i], result[i].Value[idx])
		}

		result[i].Scale = q
		util.Rescale_NonNTT(evaluator, result[i], result[i])
	}

	for d := range inputPolys {
		for j := range n {
			inputPolys[d][j], inputPolys[d][j+n] = inputPolys[d][j+n], inputPolys[d][j]
		}
	}

	context.CTS_PCMM(inputPolys, resPolys)
	for i := range sparseN {
		result2[i].Resize(1, C2SParams.params.EndLevel)
		for idx := range 2 {
			be.ModSwitchPtoQ(PLevel, C2SParams.params.EndLevel, resPolys[idx][i], result2[i].Value[idx])
		}

		result2[i].Scale = q
		util.Rescale_NonNTT(evaluator, result2[i], result2[i])
	}

	// fmt.Println(result[0].LogScale())
	if util.Debug.IsDebug {
		elapse := time.Since(util.Debug.StartTime)
		fmt.Println("CTS-PPMM & Rescale End : ", elapse)
		util.Debug.AccTime += elapse
		fmt.Println("CTS-CMT Start")
		util.Debug.StartTime = time.Now()
	}

	transpose.Transpose_Sparse(result, params, evaluator, ringQ.AtLevel(result[0].Level()), N, sparseN, work, aux, result)
	transpose.Transpose_Sparse(result2, params, evaluator, ringQ.AtLevel(result2[0].Level()), N, sparseN, work, aux, result2)

	if util.Debug.IsDebug {
		elapse := time.Since(util.Debug.StartTime)
		fmt.Println("CTS-CMT End: ", elapse)
		util.Debug.AccTime += elapse
		util.Debug.StartTime = time.Now()
	}
	conj := aux[0].CopyNew()
	for i := range sparseN {
		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[0], result[i].Value[0])
		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[1], result[i].Value[1])

		evaluator.Conjugate(result[i], conj)
		evaluator.Add(result[i], conj, result[i])

		ringQ.AtLevel(result2[i].Level()).NTT(result2[i].Value[0], result2[i].Value[0])
		ringQ.AtLevel(result2[i].Level()).NTT(result2[i].Value[1], result2[i].Value[1])
		evaluator.Conjugate(result2[i], conj)
		evaluator.Add(result2[i], conj, result2[i])
	}

	if util.Debug.IsDebug {
		elapse := time.Since(util.Debug.StartTime)
		fmt.Println("end c2s : ", elapse)
		util.Debug.AccTime += elapse
	}

	return
}

func (context *Context) bufferclear() {
	realPolys := context.alloced.realPolys
	imagPolys := context.alloced.imagPolys
	tempPolys := context.alloced.tempPolys
	resPolys := context.alloced.resPolys
	for d := range realPolys {
		for i := range realPolys[d] {
			for l := range realPolys[d][i].Coeffs {
				clear(realPolys[d][i].Coeffs[l])
			}
		}
	}

	for d := range imagPolys {
		for i := range imagPolys[d] {
			for l := range imagPolys[d][i].Coeffs {
				clear(imagPolys[d][i].Coeffs[l])
			}
		}
	}

	for d := range tempPolys {
		for i := range tempPolys[d] {
			for l := range tempPolys[d][i].Coeffs {
				clear(tempPolys[d][i].Coeffs[l])
			}
		}
	}

	for d := range resPolys {
		for i := range resPolys[d] {
			for l := range resPolys[d][i].Coeffs {
				clear(resPolys[d][i].Coeffs[l])
			}
		}
	}
}

func (context *Context) CTS_PCMM(inputPolys [][]ring.Poly, resPolys [][]ring.Poly) {
	ppmmbuffer1 := context.alloced.ppmmbuffer1
	ppmmbuffer2 := context.alloced.ppmmbuffer2

	//len(P) * n * N
	realPolys := context.alloced.realPolys
	imagPolys := context.alloced.imagPolys
	tempPolys := context.alloced.tempPolys
	ringP := context.alloced.ringP

	C2SParams := context.C2SParams
	PLevel := C2SParams.params.PLevel
	CL_arr := C2SParams.params.CL_arr
	mat0 := C2SParams.mat0
	mat0i := C2SParams.mat0i
	mat0s := C2SParams.mat0s

	N := context.N
	sparseN := context.SparseN
	n := sparseN >> 1
	context.bufferclear()
	//00
	inter_it := n
	for l := range len(CL_arr) {
		inter := inter_it >> CL_arr[l]
		llen := (1 << CL_arr[l])
		if l == 0 {
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
			}
		} else if l == len(CL_arr)-1 {
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(realPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(imagPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.SubManyRing(ringP, realPolys, imagPolys, resPolys)
		} else {
			matmult.AddManyRing(ringP, realPolys, imagPolys, tempPolys)
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(realPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(imagPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(tempPolys, mat0s[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, tempPolys, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.SubManyRing(ringP, tempPolys, realPolys, tempPolys)
			matmult.SubManyRing(ringP, realPolys, imagPolys, realPolys)
			matmult.SubManyRing(ringP, tempPolys, imagPolys, imagPolys)
		}

		inter_it = inter
	}

	//01
	inter_it = n
	for l := range len(CL_arr) {
		inter := inter_it >> CL_arr[l]
		llen := (1 << CL_arr[l])
		if l == 0 {
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride2(inputPolys, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride2(inputPolys, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
			}

		} else if l == len(CL_arr)-1 {
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(realPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(imagPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.AddManyRingIdx(ringP, resPolys, realPolys, resPolys, n)
			matmult.AddManyRingIdx(ringP, resPolys, imagPolys, resPolys, n)
		} else {
			matmult.AddManyRing(ringP, realPolys, imagPolys, tempPolys)
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(realPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(imagPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(tempPolys, mat0s[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, tempPolys, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.SubManyRing(ringP, tempPolys, realPolys, tempPolys)
			matmult.SubManyRing(ringP, realPolys, imagPolys, realPolys)
			matmult.SubManyRing(ringP, tempPolys, imagPolys, imagPolys)
		}

		inter_it = inter
	}

	bitlen := bits.Len64(uint64(n)) - 1
	for d := range 2 {
		for i := range sparseN {
			if i < n {
				br := util.BitReverse(i, bitlen)
				if i >= br {
					continue
				}
				resPolys[d][i], resPolys[d][util.BitReverse(i, bitlen)] = resPolys[d][util.BitReverse(i, bitlen)], resPolys[d][i]
			} else {
				i := i % n
				br := util.BitReverse(i, bitlen)
				if i >= br {
					continue
				}
				resPolys[d][n+i], resPolys[d][n+util.BitReverse(i, bitlen)] = resPolys[d][n+util.BitReverse(i, bitlen)], resPolys[d][n+i]
			}
		}
	}
}

// // input must be NTT
// func (context *Context) SlotToCoeff(ctreal, ctimage []*rlwe.Ciphertext) (result []*rlwe.Ciphertext) {
// 	N := context.N
// 	sparseN := context.SparseN
// 	n := sparseN >> 1

// 	params := context.params
// 	evaluator := context.Evaluator

// 	S2CParams := context.S2CParams
// 	CL_arr := S2CParams.params.CL_arr
// 	PLevel := S2CParams.params.PLevel
// 	P := context.P[:PLevel+1]

// 	mat0 := S2CParams.mat0
// 	mat0i := S2CParams.mat0i
// 	// mat0s := S2CParams.mat0s

// 	ctzero := context.alloced.ctzero

// 	result = make([]*rlwe.Ciphertext, sparseN)
// 	for i := range result {
// 		result[i] = ctzero.CopyNew()
// 	}

// 	work := context.alloced.work
// 	aux := context.alloced.aux

// 	if ctimage != nil {
// 		for i := range sparseN {
// 			imag, _ := evaluator.MulNew(ctimage[i], 1i)

// 			cts[i], _ = evaluator.AddNew(ctreal[i], imag)
// 			ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[0], cts[i].Value[0])
// 			ringQ.AtLevel(cts[i].Level()).INTT(cts[i].Value[1], cts[i].Value[1])
// 		}
// 	} else {
// 		copy(cts, ctreal)
// 	}

// 	transpose.Transpose3(cts, params, evaluator, ringQ.AtLevel(cts[0].Level()), N, sparseN, work, aux, cts)

// 	for i := range sparseN {
// 		for d := range 2 {
// 			be.ModSwitchQtoP(cts[i].Level(), PLevel, cts[i].Value[d], inputPolys[d][i])
// 		}
// 		if i < n {
// 			ringQ.AtLevel(cts[0].Level()).Neg(cts[i+n].Value[0], work[0].Value[0])
// 			ringQ.AtLevel(cts[0].Level()).Neg(cts[i+n].Value[1], work[0].Value[1])
// 		} else {
// 			work[0] = cts[i-n]
// 		}
// 		for d := range 2 {
// 			be.ModSwitchQtoP(cts[i].Level(), PLevel, work[0].Value[d], inputPolysC[d][i])
// 		}
// 	}

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

// 	matmult.PPMM_Blas_CRT_Inplace(inputPolys, matrev, N, N, N, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)
// 	matmult.PPMM_Blas_CRT_Inplace(inputPolysC, matrev, N, N, N, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)

// 	inter_it = 1
// 	for l := range len(SF) {
// 		inter := inter_it
// 		inter_it = inter << CL_arr[l]
// 		llen := (1 << CL_arr[l])
// 		if l == 0 {
// 			for t := range n / llen {
// 				stpoint := (t % inter) + inter_it*int(t/inter)
// 				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00i, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys01, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys01i, ppmmbuffer1, ppmmbuffer2)

// 				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys10, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys10i, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(inputPolysC, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11i, ppmmbuffer1, ppmmbuffer2)
// 			}

// 		} else if l == len(SF)-1 {
// 			for t := range n / llen {
// 				stpoint := (t % inter) + inter_it*int(t/inter)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys00, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys00i, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys00i, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys01, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys01, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys01i, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, resPolys01i, ppmmbuffer1, ppmmbuffer2)
// 			}
// 			matmult.SubManyRing(ringP, resPolys00, resPolys00i, resPolys00)
// 			matmult.AddManyRing(ringP, resPolys01, resPolys01i, resPolys01i)

// 			for t := range n / llen {
// 				stpoint := (t % inter) + inter_it*int(t/inter)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys10, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys10, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys10i, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys10i, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys11, mat0i[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11, ppmmbuffer1, ppmmbuffer2)
// 				matmult.PPMM_Blas_CRT_Stride(resPolys11i, mat0[l][t], llen, llen, N, n+stpoint, inter, PLevel, 2, ringP, resPolys11i, ppmmbuffer1, ppmmbuffer2)
// 			}
// 			matmult.SubManyRing(ringP, resPolys10, resPolys10i, resPolys10)
// 			matmult.AddManyRing(ringP, resPolys11, resPolys11i, resPolys11i)

// 		}

// 	}
// 	matmult.AddManyRing(ringP, resPolys00, resPolys01i, resPolys00)
// 	matmult.AddManyRing(ringP, resPolys00, resPolys10, resPolys00)
// 	matmult.AddManyRing(ringP, resPolys00, resPolys11i, resPolys00)

// 	// rev := matmult.BitReversePermutationMatrix(n)
// 	// matrev := make([]float64, len(P)*N*N)
// 	// for p := range len(P) {
// 	// 	for i := range N {
// 	// 		for j := range N {
// 	// 			if (i < n && j < n) || (i >= n && j >= n) {
// 	// 				matrev[p*N*N+i*N+j] = (real(rev[i%n][j%n]))
// 	// 			}
// 	// 		}
// 	// 	}
// 	// }

// 	// matmult.PPMM_Blas_CRT_Inplace(resPolys00, matrev, N, N, N, PLevel+1, 2, ringP, ppmmbuffer1, ppmmbuffer2)

// 	for i := range N {
// 		for idx := range 2 {
// 			be.ModSwitchPtoQ(PLevel, S2CParams.EndLevel, resPolys00[idx][i], result[i].Value[idx])
// 		}

// 		q := rlwe.NewScale(params.Q()[result[i].Level()])
// 		util.Mul_ScaleExact(evaluator, result[i], 1.0/(scale), result[i], q)
// 		util.Rescale_NonNTT(evaluator, result[i], result[i])

// 		q = rlwe.NewScale(params.Q()[result[i].Level()])
// 		util.Mul_ScaleExact(evaluator, result[i], 1.0/(scale), result[i], q)
// 		util.Rescale_NonNTT(evaluator, result[i], result[i])
// 	}

// 	transpose.Transpose3(result, params, evaluator, ringQ.AtLevel(result[0].Level()), N, sparseN, work, aux, result)

// 	for i := range result {
// 		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[0], result[i].Value[0])
// 		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[1], result[i].Value[1])
// 	}

// 	return
// }

// input must be NTT
func (context *Context) SlotToCoeff2(ctreal, ctimage []*rlwe.Ciphertext) (result []*rlwe.Ciphertext) {
	N := context.N
	sparseN := context.SparseN
	// n := sparseN >> 1

	params := context.params
	evaluator := context.Evaluator

	S2CParams := context.S2CParams
	CL_arr := S2CParams.params.CL_arr
	PLevel := S2CParams.params.PLevel
	// P := context.P[:PLevel+1]

	ctzero := context.alloced.ctzero

	result = make([]*rlwe.Ciphertext, sparseN)
	for i := range result {
		result[i] = ctzero.CopyNew()
	}

	work := context.alloced.work
	aux := context.alloced.aux
	ringQ := context.alloced.ringQ
	be := context.alloced.be

	inputPolys := context.alloced.inputPolys
	resPolys := context.alloced.resPolys

	if ctimage != nil {
		for i := range sparseN {
			imag, _ := evaluator.MulNew(ctimage[i], 1i)
			result[i], _ = evaluator.AddNew(ctreal[i], imag)
			ringQ.AtLevel(result[i].Level()).INTT(result[i].Value[0], result[i].Value[0])
			ringQ.AtLevel(result[i].Level()).INTT(result[i].Value[1], result[i].Value[1])
		}
	} else {
		for i := range sparseN {
			ringQ.AtLevel(ctreal[i].Level()).INTT(ctreal[i].Value[0], result[i].Value[0])
			ringQ.AtLevel(ctreal[i].Level()).INTT(ctreal[i].Value[1], result[i].Value[1])
		}
	}

	transpose.Transpose_Sparse(result, params, evaluator, ringQ.AtLevel(result[0].Level()), N, sparseN, work, aux, result)

	for i := range sparseN {
		for d := range 2 {
			be.ModSwitchQtoP(result[i].Level(), PLevel, result[i].Value[d], inputPolys[d][i])
		}
	}

	context.STC_PPMM(inputPolys, resPolys)
	q := rlwe.NewScale(params.DefaultScale())
	for i := range len(CL_arr) {
		q = q.Mul(rlwe.NewScale(params.Q()[S2CParams.params.EndLevel-i]))
	}

	for i := range sparseN {
		for idx := range 2 {
			be.ModSwitchPtoQ(PLevel, S2CParams.params.EndLevel, resPolys[idx][i], result[i].Value[idx])
		}

		result[i].Scale = q
		util.Rescale_NonNTT(evaluator, result[i], result[i])
	}

	for i := range result {
		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[0], result[i].Value[0])
		ringQ.AtLevel(result[i].Level()).NTT(result[i].Value[1], result[i].Value[1])
	}

	return
}

func (context *Context) STC_PPMM(inputPolys [][]ring.Poly, resPolys [][]ring.Poly) {
	ppmmbuffer1 := context.alloced.ppmmbuffer1
	ppmmbuffer2 := context.alloced.ppmmbuffer2

	realPolys := context.alloced.realPolys
	imagPolys := context.alloced.imagPolys
	tempPolys := context.alloced.tempPolys
	ringP := context.alloced.ringP
	S2CParams := context.S2CParams

	PLevel := S2CParams.params.PLevel
	CL_arr := S2CParams.params.CL_arr
	mat0 := S2CParams.mat0
	mat0i := S2CParams.mat0i
	mat0s := S2CParams.mat0s

	N := context.N
	sparseN := context.SparseN
	n := sparseN >> 1

	bitlen := bits.Len64(uint64(n)) - 1
	for d := range 2 {
		for i := range sparseN {
			if i < n {
				br := util.BitReverse(i, bitlen)
				if i >= br {
					continue
				}
				inputPolys[d][i], inputPolys[d][util.BitReverse(i, bitlen)] = inputPolys[d][util.BitReverse(i, bitlen)], inputPolys[d][i]
			} else {
				i := i % n
				br := util.BitReverse(i, bitlen)
				if i >= br {
					continue
				}
				inputPolys[d][n+i], inputPolys[d][n+util.BitReverse(i, bitlen)] = inputPolys[d][n+util.BitReverse(i, bitlen)], inputPolys[d][n+i]
			}
		}
	}

	//00
	inter_it := 1
	for l := range len(CL_arr) {
		inter := inter_it
		inter_it = inter << CL_arr[l]
		llen := (1 << CL_arr[l])
		if l == 0 {
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
			}

		} else if l == len(CL_arr)-1 {
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(realPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(imagPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.SubManyRing(ringP, realPolys, imagPolys, resPolys)
		} else {
			matmult.AddManyRing(ringP, realPolys, imagPolys, tempPolys)
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(realPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(imagPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(tempPolys, mat0s[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, tempPolys, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.SubManyRing(ringP, tempPolys, realPolys, tempPolys)
			matmult.SubManyRing(ringP, realPolys, imagPolys, realPolys)
			matmult.SubManyRing(ringP, tempPolys, imagPolys, imagPolys)
		}
	}

	//01
	inter_it = 1
	for l := range len(CL_arr) {
		inter := inter_it
		inter_it = inter << CL_arr[l]
		llen := (1 << CL_arr[l])
		if l == 0 {
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride2(inputPolys, mat0[l][t], llen, llen, N, n+stpoint, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride2(inputPolys, mat0i[l][t], llen, llen, N, n+stpoint, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
			}

		} else if l == len(CL_arr)-1 {
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(realPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(imagPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.SubManyRing(ringP, resPolys, realPolys, resPolys)
			matmult.SubManyRing(ringP, resPolys, imagPolys, resPolys)
		} else {
			matmult.AddManyRing(ringP, realPolys, imagPolys, tempPolys)
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(realPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(imagPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(tempPolys, mat0s[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, tempPolys, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.SubManyRing(ringP, tempPolys, realPolys, tempPolys)
			matmult.SubManyRing(ringP, realPolys, imagPolys, realPolys)
			matmult.SubManyRing(ringP, tempPolys, imagPolys, imagPolys)
		}
	}

	//10
	inter_it = 1
	for l := range len(CL_arr) {
		inter := inter_it
		inter_it = inter << CL_arr[l]
		llen := (1 << CL_arr[l])
		if l == 0 {
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride2(inputPolys, mat0[l][t], llen, llen, N, n+stpoint, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride2(inputPolys, mat0i[l][t], llen, llen, N, n+stpoint, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
			}

		} else if l == len(CL_arr)-1 {
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(realPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(imagPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.AddManyRingIdx(ringP, resPolys, realPolys, resPolys, n)
			matmult.SubManyRingIdx(ringP, resPolys, imagPolys, resPolys, n)
		} else {
			matmult.AddManyRing(ringP, realPolys, imagPolys, tempPolys)
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(realPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(imagPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(tempPolys, mat0s[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, tempPolys, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.SubManyRing(ringP, tempPolys, realPolys, tempPolys)
			matmult.SubManyRing(ringP, realPolys, imagPolys, realPolys)
			matmult.SubManyRing(ringP, tempPolys, imagPolys, imagPolys)
		}
	}

	//11
	inter_it = 1
	for l := range len(CL_arr) {
		inter := inter_it
		inter_it = inter << CL_arr[l]
		llen := (1 << CL_arr[l])
		if l == 0 {
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(inputPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
			}

		} else if l == len(CL_arr)-1 {
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(realPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(imagPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.AddManyRingIdx(ringP, resPolys, realPolys, resPolys, n)
			matmult.AddManyRingIdx(ringP, resPolys, imagPolys, resPolys, n)
		} else {
			matmult.AddManyRing(ringP, realPolys, imagPolys, tempPolys)
			for t := range n / llen {
				stpoint := (t % inter) + inter_it*int(t/inter)
				matmult.PPMM_Blas_CRT_Stride(realPolys, mat0[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(imagPolys, mat0i[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				matmult.PPMM_Blas_CRT_Stride(tempPolys, mat0s[l][t], llen, llen, N, stpoint, inter, PLevel, 2, ringP, tempPolys, ppmmbuffer1, ppmmbuffer2)
			}
			matmult.SubManyRing(ringP, tempPolys, realPolys, tempPolys)
			matmult.SubManyRing(ringP, realPolys, imagPolys, realPolys)
			matmult.SubManyRing(ringP, tempPolys, imagPolys, imagPolys)
		}
	}
}
