package boot

import (
	"fmt"
	"math/bits"
	"time"

	"github.com/lifejade/mm/src/matmult"
	"github.com/lifejade/mm/src/transpose"
	"github.com/lifejade/mm/src/util"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/ring"
)

// inNTT : is input NTT? outNTT : is output NTT?
func (context *Context) ModUpAdjNTT(cts []*rlwe.Ciphertext, rescts []*rlwe.Ciphertext, inNTT, outNTT bool) {
	if inNTT {
		context.INTT(cts, rescts)
		context.ModUp(rescts, rescts)
	} else {
		context.ModUp(cts, rescts)
	}

	if outNTT {
		context.NTT(rescts, rescts)
	}
}

// inNTT : is input NTT? outNTT : is output NTT?
func (context *Context) CoeffsToSlotsAdjNTT(cts []*rlwe.Ciphertext, rescts1, rescts2 []*rlwe.Ciphertext, inNTT, outNTT bool) {
	if inNTT {
		context.INTT(cts, rescts1)
		context.CoeffToSlot2(rescts1, rescts1, rescts2)
	} else {
		context.CoeffToSlot2(cts, rescts1, rescts2)
	}

	if outNTT {
		context.NTT(rescts1, rescts1)
		context.NTT(rescts2, rescts2)
	}
}

// inNTT : is input NTT? outNTT : is output NTT?
func (context *Context) SlotsToCoeffsAdjNTT(ctsreal, ctsimag []*rlwe.Ciphertext, rescts []*rlwe.Ciphertext, inNTT, outNTT bool) {
	if inNTT {
		context.INTT(ctsreal, rescts)
		temp := context.alloced.work
		context.INTT(ctsimag, temp)
		context.SlotToCoeff2(rescts, temp, rescts)
	} else {
		context.SlotToCoeff2(ctsreal, ctsimag, rescts)
	}

	if outNTT {
		context.NTT(rescts, rescts)
	}
}

// input INTT, output INTT
func (context *Context) ModUp(cts []*rlwe.Ciphertext, rescts []*rlwe.Ciphertext) {
	var elapseThis time.Duration

	ringQ := context.alloced.ringQ
	N := context.N
	sparseN := context.SparseN
	ratio := N / sparseN

	eval := context.Evaluator
	ctsj := context.alloced.cttemp
	logN := context.params.LogN()
	params := context.params
	H := context.EvalModParams.params.H

	for i := range cts {
		if cts[i].Level() != 0 {
			eval.DropLevel(cts[i], cts[i].Level())
		}
	}

	logratio := bits.Len(uint(N/sparseN)) - 1

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
		elapseThis += elapse
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
		// util.Debug.AccTime += elapse
		elapseThis += elapse
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
		// util.Debug.AccTime += elapse
		elapseThis += elapse
		fmt.Println("Sparse To Dense End : ", elapse)
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
		elapseThis += elapse
		util.Debug.AccTime += elapseThis
		fmt.Println("PartialSum End : ", elapse)
		fmt.Println("ModUp End : ", elapseThis)
	}
}
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

// input&output must be INTT
func (context *Context) RemoveSparse(cts []*rlwe.Ciphertext, result []*rlwe.Ciphertext) {
	sparseN := context.SparseN
	N := context.N
	ratio := N / sparseN
	ringQ := context.alloced.ringQ.AtLevel(cts[0].Level())
	evaluator := context.Evaluator
	logN := context.params.LogN()
	params := context.params
	ninv := ringQ.NewRNSScalarFromUInt64(uint64(ratio))
	ringQ.MFormRNSScalar(ninv, ninv)
	ringQ.Inverse(ninv)
	temp := context.alloced.cttemp

	var err error
	logratio := bits.Len(uint(ratio)) - 1
	gks := make([]*rlwe.GaloisKey, logratio)
	for j := range logratio {
		var gk *rlwe.GaloisKey
		galEl := uint64(1<<(logN-j) + 1)

		if gk, err = evaluator.CheckAndGetGaloisKey(galEl); err != nil {
			if util.SContext.Sk == nil && !util.Debug.IsDebug {
				panic(err)
			}
			elapse := time.Since(util.Debug.StartTime)
			util.Debug.AccTime += elapse
			kgen_ := rlwe.NewKeyGenerator(params)
			gk = kgen_.GenGaloisKeyNew(galEl, util.SContext.Sk)
			util.Debug.StartTime = time.Now()
		}
		gks[j] = gk
	}

	for i := range result {

		ringQ.MForm(result[i].Value[0], result[i].Value[0])
		ringQ.MForm(result[i].Value[1], result[i].Value[1])

		ringQ.MulRNSScalarMontgomery(result[i].Value[0], ninv, result[i].Value[0])
		ringQ.MulRNSScalarMontgomery(result[i].Value[1], ninv, result[i].Value[1])

		ringQ.IMForm(result[i].Value[0], result[i].Value[0])
		ringQ.IMForm(result[i].Value[1], result[i].Value[1])

		for j := range logratio {
			galEl := uint64(1<<(logN-j) + 1)
			result[i].IsNTT = false
			transpose.Automorphism(evaluator, ringQ, result[i], (galEl), gks[j], temp)
			result[i].IsNTT = true
			temp.IsNTT = true
			evaluator.Add(result[i], temp, result[i])
		}
	}
}

// input&output must be INTT
func (context *Context) CoeffToSlot2(cts, result, result2 []*rlwe.Ciphertext) {
	var elapseThis time.Duration
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

	temp := context.alloced.cttemp

	if util.Debug.IsDebug {
		fmt.Println("start c2s")
		util.PrintMemUsage()
		util.Debug.StartTime = time.Now()
	}

	for i := range sparseN {
		for d := range 2 {
			be.ModSwitchQtoP(cts[i].Level(), PLevel, cts[i].Value[d], inputPolys[d][i])
		}
	}

	if len(CL_arr) != 1 {
		context.CTS_PCMM(inputPolys, resPolys)
	} else {
		context.CTS_PCMM_NoCollpase(inputPolys, resPolys)
	}

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
	if len(CL_arr) != 1 {
		context.CTS_PCMM(inputPolys, resPolys)
	} else {
		context.CTS_PCMM_NoCollpase(inputPolys, resPolys)
	}

	for i := range sparseN {
		result2[i].Resize(1, C2SParams.params.EndLevel)
		for idx := range 2 {
			be.ModSwitchPtoQ(PLevel, C2SParams.params.EndLevel, resPolys[idx][i], result2[i].Value[idx])
		}

		result2[i].Scale = q
		util.Rescale_NonNTT(evaluator, result2[i], result2[i])
	}

	if util.Debug.IsDebug {
		elapse := time.Since(util.Debug.StartTime)
		fmt.Println("CTS-PPMM & Rescale End : ", elapse)
		elapseThis += elapse
		fmt.Println("CTS-CMT Start")
		util.Debug.StartTime = time.Now()
	}

	transpose.Transpose3(result, params, evaluator, ringQ.AtLevel(result[0].Level()), N, sparseN, work, aux, result)
	transpose.Transpose3(result2, params, evaluator, ringQ.AtLevel(result2[0].Level()), N, sparseN, work, aux, result2)
	context.RemoveSparse(result, result)
	context.RemoveSparse(result2, result2)

	if util.Debug.IsDebug {
		elapse := time.Since(util.Debug.StartTime)
		elapseThis += elapse
		util.Debug.StartTime = time.Now()
	}
	{
		ringQ := ringQ.AtLevel(result[0].Level())
		var gk *rlwe.GaloisKey
		var err error
		galEl := params.GaloisElementOrderTwoOrthogonalSubgroup()
		if gk, err = evaluator.CheckAndGetGaloisKey(galEl); err != nil {
			if util.SContext.Sk == nil && !util.Debug.IsDebug {
				panic(err)
			}
			util.Debug.AccTime += time.Since(util.Debug.StartTime)
			kgen_ := rlwe.NewKeyGenerator(params)
			gk = kgen_.GenGaloisKeyNew(galEl, util.SContext.Sk)
			util.Debug.StartTime = time.Now()
		}

		for i := range sparseN {
			result[i].IsNTT = false
			temp.IsNTT = false
			transpose.Automorphism(evaluator, ringQ, result[i], galEl, gk, temp)
			result[i].IsNTT = true
			temp.IsNTT = true
			evaluator.Add(result[i], temp, result[i])

			result2[i].IsNTT = false
			temp.IsNTT = false
			transpose.Automorphism(evaluator, ringQ, result2[i], galEl, gk, temp)
			result2[i].IsNTT = true
			temp.IsNTT = true
			evaluator.Add(result2[i], temp, result2[i])

		}

	}

	if util.Debug.IsDebug {
		elapse := time.Since(util.Debug.StartTime)
		elapseThis += elapse
		fmt.Println("end c2s : ", elapseThis)
		util.Debug.AccTime += elapseThis
	}

	return
}

func (context *Context) CTS_PCMM(inputPolys [][]matmult.Poly, resPolys [][]matmult.Poly) {
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
	context.resbufferClear()

	for d := range 2 {
		//00
		inter_it := n
		for l := range len(CL_arr) {
			inter := inter_it >> CL_arr[l]
			llen := (1 << CL_arr[l])
			if l == 0 {
				for t := range n / llen {
					stpoint := (t % inter) + inter_it*int(t/inter)
					matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				}
			} else if l == len(CL_arr)-1 {
				for t := range n / llen {
					stpoint := (t % inter) + inter_it*int(t/inter)
					matmult.PPMM_Blas_CRT_Stride2(realPolys, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(imagPolys, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				}
				matmult.SubManyRingIdx(ringP, realPolys, imagPolys, resPolys[d], 0)
			} else {
				matmult.AddManyRingIdx(ringP, realPolys, imagPolys, tempPolys, 0)
				for t := range n / llen {
					stpoint := (t % inter) + inter_it*int(t/inter)
					matmult.PPMM_Blas_CRT_Stride2(realPolys, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(imagPolys, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(tempPolys, mat0s[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, tempPolys, ppmmbuffer1, ppmmbuffer2)
				}
				matmult.SubManyRingIdx(ringP, tempPolys, realPolys, tempPolys, 0)
				matmult.SubManyRingIdx(ringP, realPolys, imagPolys, realPolys, 0)
				matmult.SubManyRingIdx(ringP, tempPolys, imagPolys, imagPolys, 0)
			}

			inter_it = inter
		}
		fmt.Println(d, "PCMM")
		util.PrintMemUsage()
		//01
		inter_it = n
		for l := range len(CL_arr) {
			inter := inter_it >> CL_arr[l]
			llen := (1 << CL_arr[l])
			if l == 0 {
				for t := range n / llen {
					stpoint := (t % inter) + inter_it*int(t/inter)
					matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				}

			} else if l == len(CL_arr)-1 {
				for t := range n / llen {
					stpoint := (t % inter) + inter_it*int(t/inter)
					matmult.PPMM_Blas_CRT_Stride2(realPolys, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(imagPolys, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				}
				matmult.AddManyRingIdx(ringP, resPolys[d], realPolys, resPolys[d], n)
				matmult.AddManyRingIdx(ringP, resPolys[d], imagPolys, resPolys[d], n)
			} else {
				matmult.AddManyRingIdx(ringP, realPolys, imagPolys, tempPolys, 0)
				for t := range n / llen {
					stpoint := (t % inter) + inter_it*int(t/inter)
					matmult.PPMM_Blas_CRT_Stride2(realPolys, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(imagPolys, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(tempPolys, mat0s[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, tempPolys, ppmmbuffer1, ppmmbuffer2)
				}
				matmult.SubManyRingIdx(ringP, tempPolys, realPolys, tempPolys, 0)
				matmult.SubManyRingIdx(ringP, realPolys, imagPolys, realPolys, 0)
				matmult.SubManyRingIdx(ringP, tempPolys, imagPolys, imagPolys, 0)
			}

			inter_it = inter
		}
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

func (context *Context) CTS_PCMM_NoCollpase(inputPolys [][]matmult.Poly, resPolys [][]matmult.Poly) {
	ppmmbuffer1 := context.alloced.ppmmbuffer1
	ppmmbuffer2 := context.alloced.ppmmbuffer2

	//len(P) * n * N
	realPolys := context.alloced.realPolys
	imagPolys := context.alloced.imagPolys
	ringP := context.alloced.ringP

	C2SParams := context.C2SParams
	PLevel := C2SParams.params.PLevel
	mat0 := C2SParams.mat0
	mat0i := C2SParams.mat0i

	N := context.N
	sparseN := context.SparseN
	n := sparseN >> 1
	context.resbufferClear()

	inter := 1
	llen := n
	for d := range 2 {
		matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0[0][0], llen, llen, N, 0, 0, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
		matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0i[0][0], llen, llen, N, 0, 0, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
		matmult.AddManyRingIdx(ringP, resPolys[d], realPolys, resPolys[d], 0)
		matmult.AddManyRingIdx(ringP, resPolys[d], imagPolys, resPolys[d], n)
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

// input must be INTT
func (context *Context) SlotToCoeff2(ctreal, ctimage, result []*rlwe.Ciphertext) {
	var elapseThis time.Duration
	N := context.N
	sparseN := context.SparseN
	// n := sparseN >> 1

	params := context.params
	evaluator := context.Evaluator

	S2CParams := context.S2CParams
	CL_arr := S2CParams.params.CL_arr
	PLevel := S2CParams.params.PLevel
	// P := context.P[:PLevel+1]

	work := context.alloced.work
	aux := context.alloced.aux
	ringQ := context.alloced.ringQ
	be := context.alloced.be

	cttemp := context.alloced.cttemp

	inputPolys := context.alloced.inputPolys
	resPolys := context.alloced.resPolys
	if util.Debug.IsDebug {
		fmt.Println("STC Start")
		util.Debug.StartTime = time.Now()
	}

	if ctimage != nil {
		for i := range sparseN {
			// imag, _ := evaluator.MulNew(ctimage[i], 1i)
			util.MulImag(evaluator, ctimage[i], cttemp)
			result[i], _ = evaluator.AddNew(ctreal[i], cttemp)
		}
	}

	transpose.Transpose3(result, params, evaluator, ringQ.AtLevel(result[0].Level()), N, sparseN, work, aux, result)
	for i := range sparseN {
		for d := range 2 {
			be.ModSwitchQtoP(result[i].Level(), PLevel, result[i].Value[d], inputPolys[d][i])
		}
	}
	if len(CL_arr) != 1 {
		context.STC_PPMM(inputPolys, resPolys)
	} else {
		context.STC_PPMM_NoCollapse(inputPolys, resPolys)
	}

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
	if util.Debug.IsDebug {
		elapse := time.Since(util.Debug.StartTime)
		elapseThis += elapse
		fmt.Println("STC End")
		util.Debug.AccTime += elapseThis
	}
	return
}

func (context *Context) STC_PPMM(inputPolys [][]matmult.Poly, resPolys [][]matmult.Poly) {
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
	context.resbufferClear()
	//00
	for d := range 2 {
		inter_it := 1
		for l := range len(CL_arr) {
			inter := inter_it
			inter_it = inter << CL_arr[l]
			llen := (1 << CL_arr[l])
			if l == 0 {
				for t := range n / llen {
					stpoint := (t % inter) + inter_it*int(t/inter)
					matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				}

			} else if l == len(CL_arr)-1 {
				for t := range n / llen {
					stpoint := (t % inter) + inter_it*int(t/inter)
					matmult.PPMM_Blas_CRT_Stride2(realPolys, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(imagPolys, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				}
				matmult.SubManyRingIdx(ringP, realPolys, imagPolys, resPolys[d], 0)
			} else {
				matmult.AddManyRingIdx(ringP, realPolys, imagPolys, tempPolys, 0)
				for t := range n / llen {
					stpoint := (t % inter) + inter_it*int(t/inter)
					matmult.PPMM_Blas_CRT_Stride2(realPolys, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(imagPolys, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(tempPolys, mat0s[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, tempPolys, ppmmbuffer1, ppmmbuffer2)
				}
				matmult.SubManyRingIdx(ringP, tempPolys, realPolys, tempPolys, 0)
				matmult.SubManyRingIdx(ringP, realPolys, imagPolys, realPolys, 0)
				matmult.SubManyRingIdx(ringP, tempPolys, imagPolys, imagPolys, 0)
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
					matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0[l][t], llen, llen, N, n+stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0i[l][t], llen, llen, N, n+stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				}

			} else if l == len(CL_arr)-1 {
				for t := range n / llen {
					stpoint := (t % inter) + inter_it*int(t/inter)
					matmult.PPMM_Blas_CRT_Stride2(realPolys, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(imagPolys, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				}
				matmult.SubManyRingIdx(ringP, resPolys[d], realPolys, resPolys[d], 0)
				matmult.SubManyRingIdx(ringP, resPolys[d], imagPolys, resPolys[d], 0)
			} else {
				matmult.AddManyRingIdx(ringP, realPolys, imagPolys, tempPolys, 0)
				for t := range n / llen {
					stpoint := (t % inter) + inter_it*int(t/inter)
					matmult.PPMM_Blas_CRT_Stride2(realPolys, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(imagPolys, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(tempPolys, mat0s[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, tempPolys, ppmmbuffer1, ppmmbuffer2)
				}
				matmult.SubManyRingIdx(ringP, tempPolys, realPolys, tempPolys, 0)
				matmult.SubManyRingIdx(ringP, realPolys, imagPolys, realPolys, 0)
				matmult.SubManyRingIdx(ringP, tempPolys, imagPolys, imagPolys, 0)
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
					matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0[l][t], llen, llen, N, n+stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0i[l][t], llen, llen, N, n+stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				}

			} else if l == len(CL_arr)-1 {
				for t := range n / llen {
					stpoint := (t % inter) + inter_it*int(t/inter)
					matmult.PPMM_Blas_CRT_Stride2(realPolys, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(imagPolys, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				}
				matmult.AddManyRingIdx(ringP, resPolys[d], realPolys, resPolys[d], n)
				matmult.SubManyRingIdx(ringP, resPolys[d], imagPolys, resPolys[d], n)
			} else {
				matmult.AddManyRingIdx(ringP, realPolys, imagPolys, tempPolys, 0)
				for t := range n / llen {
					stpoint := (t % inter) + inter_it*int(t/inter)
					matmult.PPMM_Blas_CRT_Stride2(realPolys, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(imagPolys, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(tempPolys, mat0s[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, tempPolys, ppmmbuffer1, ppmmbuffer2)
				}
				matmult.SubManyRingIdx(ringP, tempPolys, realPolys, tempPolys, 0)
				matmult.SubManyRingIdx(ringP, realPolys, imagPolys, realPolys, 0)
				matmult.SubManyRingIdx(ringP, tempPolys, imagPolys, imagPolys, 0)
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
					matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				}

			} else if l == len(CL_arr)-1 {
				for t := range n / llen {
					stpoint := (t % inter) + inter_it*int(t/inter)
					matmult.PPMM_Blas_CRT_Stride2(realPolys, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(imagPolys, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
				}
				matmult.AddManyRingIdx(ringP, resPolys[d], realPolys, resPolys[d], n)
				matmult.AddManyRingIdx(ringP, resPolys[d], imagPolys, resPolys[d], n)
			} else {
				matmult.AddManyRingIdx(ringP, realPolys, imagPolys, tempPolys, 0)
				for t := range n / llen {
					stpoint := (t % inter) + inter_it*int(t/inter)
					matmult.PPMM_Blas_CRT_Stride2(realPolys, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(imagPolys, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
					matmult.PPMM_Blas_CRT_Stride2(tempPolys, mat0s[l][t], llen, llen, N, stpoint, stpoint, inter, PLevel, ringP, tempPolys, ppmmbuffer1, ppmmbuffer2)
				}
				matmult.SubManyRingIdx(ringP, tempPolys, realPolys, tempPolys, 0)
				matmult.SubManyRingIdx(ringP, realPolys, imagPolys, realPolys, 0)
				matmult.SubManyRingIdx(ringP, tempPolys, imagPolys, imagPolys, 0)
			}
		}
	}

}

func (context *Context) STC_PPMM_NoCollapse(inputPolys [][]matmult.Poly, resPolys [][]matmult.Poly) {
	ppmmbuffer1 := context.alloced.ppmmbuffer1
	ppmmbuffer2 := context.alloced.ppmmbuffer2

	realPolys := context.alloced.realPolys
	imagPolys := context.alloced.imagPolys
	ringP := context.alloced.ringP
	S2CParams := context.S2CParams

	PLevel := S2CParams.params.PLevel
	mat0 := S2CParams.mat0
	mat0i := S2CParams.mat0i

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
	context.resbufferClear()
	for d := range 2 {
		matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0[0][0], n, n, N, 0, 0, 1, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
		matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0i[0][0], n, n, N, n, 0, 1, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
		matmult.SubManyRingIdx(ringP, realPolys, imagPolys, resPolys[d], 0)
		matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0[0][0], n, n, N, n, 0, 1, PLevel, ringP, realPolys, ppmmbuffer1, ppmmbuffer2)
		matmult.AddManyRingIdx(ringP, resPolys[d], realPolys, resPolys[d], n)
		matmult.PPMM_Blas_CRT_Stride2(inputPolys[d], mat0i[0][0], n, n, N, 0, 0, 1, PLevel, ringP, imagPolys, ppmmbuffer1, ppmmbuffer2)
		matmult.AddManyRingIdx(ringP, resPolys[d], imagPolys, resPolys[d], n)
	}

}

func (context *Context) NTT(cts []*rlwe.Ciphertext, rescts []*rlwe.Ciphertext) {
	ringQ := context.alloced.ringQ.AtLevel(cts[0].Level())
	for i := range cts {
		ringQ.NTT(cts[i].Value[0], rescts[i].Value[0])
		ringQ.NTT(cts[i].Value[1], rescts[i].Value[1])
	}
}

func (context *Context) INTT(cts []*rlwe.Ciphertext, rescts []*rlwe.Ciphertext) {
	ringQ := context.alloced.ringQ.AtLevel(cts[0].Level())
	for i := range cts {
		ringQ.INTT(cts[i].Value[0], rescts[i].Value[0])
		ringQ.INTT(cts[i].Value[1], rescts[i].Value[1])
	}
}

func (context *Context) resbufferClear() {
	resPolys := context.alloced.resPolys

	for d := range resPolys {
		for i := range resPolys[d] {
			for l := range resPolys[d][i].Coeffs {
				clear(resPolys[d][i].Coeffs[l])
			}
		}
	}
}

// func (context *Context) CTS_PCMM_LowMem(inputPolys [][]matmult.Poly, resPolys [][]matmult.Poly) {
// 	ppmmbuffer1 := context.alloced.ppmmbuffer1
// 	ppmmbuffer2 := context.alloced.ppmmbuffer2

// 	ringP := context.alloced.ringP

// 	C2SParams := context.C2SParams
// 	PLevel := C2SParams.params.PLevel
// 	CL_arr := C2SParams.params.CL_arr
// 	mat0 := C2SParams.mat0
// 	mat0i := C2SParams.mat0i
// 	mat0s := C2SParams.mat0s

// 	N := context.N
// 	sparseN := context.SparseN
// 	n := sparseN >> 1
// 	P := ringP.ModuliChain()
// 	context.resbufferClear()
// 	inputbuffer := make([]uint32, sparseN*N)
// 	realbuffer := make([]uint32, n*N)
// 	imagbuffer := make([]uint32, n*N)
// 	tempbuffer := make([]uint32, n*N)
// 	util.PrintMemUsage()
// 	// copy(buffer[i*N:], inputPolys[d][i].Coeffs[level])

// 	//00

// 	for d := range 2 {
// 		for level := range PLevel {
// 			for i := range inputPolys[d] {
// 				copy(inputbuffer[i*N:(i+1)*N], inputPolys[d][i].Coeffs[level])
// 			}
// 			inter_it := n
// 			for l := range len(CL_arr) {
// 				inter := inter_it >> CL_arr[l]
// 				llen := (1 << CL_arr[l])
// 				if l == 0 {
// 					for t := range n / llen {
// 						stpoint := (t % inter) + inter_it*int(t/inter)
// 						matmult.PPMM_Blas_CRT_Stride_LowMem(inputbuffer, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, int64(P[level]), realbuffer, ppmmbuffer1, ppmmbuffer2)
// 						matmult.PPMM_Blas_CRT_Stride_LowMem(inputbuffer, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, int64(P[level]), imagbuffer, ppmmbuffer1, ppmmbuffer2)
// 					}
// 				} else if l == len(CL_arr)-1 {
// 					for t := range n / llen {
// 						stpoint := (t % inter) + inter_it*int(t/inter)
// 						matmult.PPMM_Blas_CRT_Stride_LowMem(realbuffer, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, int64(P[level]), realbuffer, ppmmbuffer1, ppmmbuffer2)
// 						matmult.PPMM_Blas_CRT_Stride_LowMem(imagbuffer, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, int64(P[level]), imagbuffer, ppmmbuffer1, ppmmbuffer2)
// 					}
// 					matmult.SubManyRingIdx_LowMem(realbuffer, imagbuffer, tempbuffer, 0, N, uint32(P[level]))
// 					for j := range n {
// 						copy(resPolys[d][j].Coeffs[level], tempbuffer[j*N:(j+1)*N])
// 					}

// 				} else {
// 					matmult.AddManyRingIdx_LowMem(realbuffer, imagbuffer, tempbuffer, 0, N, uint32(P[level]))
// 					for t := range n / llen {
// 						stpoint := (t % inter) + inter_it*int(t/inter)
// 						matmult.PPMM_Blas_CRT_Stride_LowMem(realbuffer, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, int64(P[level]), realbuffer, ppmmbuffer1, ppmmbuffer2)
// 						matmult.PPMM_Blas_CRT_Stride_LowMem(imagbuffer, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, int64(P[level]), imagbuffer, ppmmbuffer1, ppmmbuffer2)
// 						matmult.PPMM_Blas_CRT_Stride_LowMem(tempbuffer, mat0s[l][t], llen, llen, N, stpoint, stpoint, inter, int64(P[level]), tempbuffer, ppmmbuffer1, ppmmbuffer2)
// 					}
// 					matmult.SubManyRingIdx_LowMem(tempbuffer, realbuffer, tempbuffer, 0, N, uint32(P[level]))
// 					matmult.SubManyRingIdx_LowMem(realbuffer, imagbuffer, realbuffer, 0, N, uint32(P[level]))
// 					matmult.SubManyRingIdx_LowMem(tempbuffer, imagbuffer, imagbuffer, 0, N, uint32(P[level]))
// 				}

// 				inter_it = inter
// 			}

// 			//01
// 			inter_it = n
// 			for l := range len(CL_arr) {
// 				inter := inter_it >> CL_arr[l]
// 				llen := (1 << CL_arr[l])
// 				if l == 0 {
// 					for t := range n / llen {
// 						stpoint := (t % inter) + inter_it*int(t/inter)
// 						matmult.PPMM_Blas_CRT_Stride_LowMem(inputbuffer, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, int64(P[level]), realbuffer, ppmmbuffer1, ppmmbuffer2)
// 						matmult.PPMM_Blas_CRT_Stride_LowMem(inputbuffer, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, int64(P[level]), imagbuffer, ppmmbuffer1, ppmmbuffer2)
// 					}

// 				} else if l == len(CL_arr)-1 {
// 					for t := range n / llen {
// 						stpoint := (t % inter) + inter_it*int(t/inter)
// 						matmult.PPMM_Blas_CRT_Stride_LowMem(realbuffer, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, int64(P[level]), realbuffer, ppmmbuffer1, ppmmbuffer2)
// 						matmult.PPMM_Blas_CRT_Stride_LowMem(imagbuffer, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, int64(P[level]), imagbuffer, ppmmbuffer1, ppmmbuffer2)
// 					}
// 					// matmult.AddManyRingIdx_LowMem(resPolys, realbuffer, resPolys, n, N, uint32(P[level]))
// 					matmult.AddManyRingIdx_LowMem(realbuffer, imagbuffer, tempbuffer, 0, N, uint32(P[level]))

// 					for j := range n {
// 						copy(resPolys[d][n+j].Coeffs[level], tempbuffer[j*N:(j+1)*N])
// 					}
// 				} else {
// 					matmult.AddManyRingIdx_LowMem(realbuffer, imagbuffer, tempbuffer, 0, N, uint32(P[level]))
// 					for t := range n / llen {
// 						stpoint := (t % inter) + inter_it*int(t/inter)
// 						matmult.PPMM_Blas_CRT_Stride_LowMem(realbuffer, mat0[l][t], llen, llen, N, stpoint, stpoint, inter, int64(P[level]), realbuffer, ppmmbuffer1, ppmmbuffer2)
// 						matmult.PPMM_Blas_CRT_Stride_LowMem(imagbuffer, mat0i[l][t], llen, llen, N, stpoint, stpoint, inter, int64(P[level]), imagbuffer, ppmmbuffer1, ppmmbuffer2)
// 						matmult.PPMM_Blas_CRT_Stride_LowMem(tempbuffer, mat0s[l][t], llen, llen, N, stpoint, stpoint, inter, int64(P[level]), tempbuffer, ppmmbuffer1, ppmmbuffer2)
// 					}
// 					matmult.SubManyRingIdx_LowMem(tempbuffer, realbuffer, tempbuffer, 0, N, uint32(P[level]))
// 					matmult.SubManyRingIdx_LowMem(realbuffer, imagbuffer, realbuffer, 0, N, uint32(P[level]))
// 					matmult.SubManyRingIdx_LowMem(tempbuffer, imagbuffer, imagbuffer, 0, N, uint32(P[level]))
// 				}

// 				inter_it = inter
// 			}
// 		}
// 	}
// 	bitlen := bits.Len64(uint64(n)) - 1
// 	for d := range 2 {
// 		for i := range sparseN {
// 			if i < n {
// 				br := util.BitReverse(i, bitlen)
// 				if i >= br {
// 					continue
// 				}
// 				resPolys[d][i], resPolys[d][util.BitReverse(i, bitlen)] = resPolys[d][util.BitReverse(i, bitlen)], resPolys[d][i]
// 			} else {
// 				i := i % n
// 				br := util.BitReverse(i, bitlen)
// 				if i >= br {
// 					continue
// 				}
// 				resPolys[d][n+i], resPolys[d][n+util.BitReverse(i, bitlen)] = resPolys[d][n+util.BitReverse(i, bitlen)], resPolys[d][n+i]
// 			}
// 		}
// 	}
// }
