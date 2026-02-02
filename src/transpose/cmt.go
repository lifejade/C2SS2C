package transpose

import (
	"fmt"
	"math"
	"math/bits"
	"runtime"
	"time"

	"github.com/lifejade/mm/src/util"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

func Tweak(cts []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, encoder *hefloat.Encoder, n int) []*rlwe.Ciphertext {
	if n == 1 {
		return cts
	}

	cts_ := make([]*rlwe.Ciphertext, n)
	cts_[0] = cts[0].CopyNew()
	logn := int(math.Round(math.Log2(float64(n))))

	ringQ := params.RingQ().AtLevel(cts[0].Level())
	for l := range logn {
		powl := 1 << l
		temp := make([]*rlwe.Ciphertext, powl)
		for j := range powl {
			temp[j] = cts[((2*j+1)*n)/(powl*2)].CopyNew()
		}
		aux := Tweak(temp, params, eval, encoder, powl)
		for j := range powl {

			//tmp, _ := eval.MulNew(aux[j], pt)
			tmp := aux[j].CopyNew()

			ringQ.INTT(tmp.Value[0], tmp.Value[0])
			ringQ.INTT(tmp.Value[1], tmp.Value[1])
			ringQ.MultByMonomial(tmp.Value[0], params.MaxSlots()*2/powl*j, tmp.Value[0])
			ringQ.MultByMonomial(tmp.Value[1], params.MaxSlots()*2/powl*j, tmp.Value[1])

			ringQ.NTT(tmp.Value[0], tmp.Value[0])
			ringQ.NTT(tmp.Value[1], tmp.Value[1])
			//eval.Rescale(tmp, tmp)

			var err error
			cts_[j+powl], err = eval.SubNew(cts_[j], tmp)
			if err != nil {
				fmt.Println(err)
			}
			cts_[j], err = eval.AddNew(cts_[j], tmp)
			if err != nil {
				fmt.Println(err)
			}

		}
	}

	return cts_
}

func ModInv(a, m uint64) (uint64, bool) {
	t, newT := int64(0), int64(1)
	r, newR := int64(m), int64(a)

	for newR != 0 {
		quotient := r / newR
		t, newT = newT, t-quotient*newT
		r, newR = newR, r-quotient*newR
	}

	if r > 1 {
		return 0, false // inverse does not exist
	}

	if t < 0 {
		t += int64(m)
	}

	return uint64(t), true
}
func Transpose(inputs []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, encoder *hefloat.Encoder, n int) []*rlwe.Ciphertext {
	cts := make([]*rlwe.Ciphertext, n)
	ringQ := params.RingQ().AtLevel(inputs[0].Level())
	ninv := ringQ.NewRNSScalarFromUInt64(uint64(n))
	ringQ.MFormRNSScalar(ninv, ninv)
	ringQ.Inverse(ninv)

	starttime := time.Now()
	for i := range cts {
		cts[i] = inputs[i].CopyNew()
		ringQ.INTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.INTT(cts[i].Value[1], cts[i].Value[1])

		ringQ.MultByMonomial(cts[i].Value[0], i, cts[i].Value[0])
		ringQ.MultByMonomial(cts[i].Value[1], i, cts[i].Value[1])

		ringQ.NTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.NTT(cts[i].Value[1], cts[i].Value[1])
	}
	elapse := time.Since(starttime)
	fmt.Println("mult by monomial", elapse)

	starttime = time.Now()
	aux := Tweak(cts, params, eval, encoder, n)
	elapse = time.Since(starttime)
	fmt.Println("TWEAK", elapse)

	starttime = time.Now()
	res := make([]*rlwe.Ciphertext, n)
	for i := range res {
		idx, ch := ModInv(uint64(2*i+1), uint64(2*n))
		if !ch {
			fmt.Println("err ", i, " ", idx)
		}
		res[i] = aux[(idx-1)/2].CopyNew()

		ringQ.INTT(res[i].Value[0], res[i].Value[0])
		ringQ.INTT(res[i].Value[1], res[i].Value[1])
		ringQ.MForm(res[i].Value[0], res[i].Value[0])
		ringQ.MForm(res[i].Value[1], res[i].Value[1])

		ringQ.MulRNSScalarMontgomery(res[i].Value[0], ninv, res[i].Value[0])
		ringQ.MulRNSScalarMontgomery(res[i].Value[1], ninv, res[i].Value[1])

		ringQ.IMForm(res[i].Value[0], res[i].Value[0])
		ringQ.IMForm(res[i].Value[1], res[i].Value[1])
		ringQ.NTT(res[i].Value[0], res[i].Value[0])
		ringQ.NTT(res[i].Value[1], res[i].Value[1])

		//uint64(2*i+1)
		if err := eval.Automorphism(res[i], uint64(2*i+1), res[i]); err != nil {
			fmt.Println(err)
		}
	}
	elapse = time.Since(starttime)
	fmt.Println("Automorphism", elapse)

	starttime = time.Now()
	res2 := Tweak(res, params, eval, encoder, n)
	result := make([]*rlwe.Ciphertext, n)
	for i := range n {
		ringQ.INTT(res2[i].Value[0], res2[i].Value[0])
		ringQ.INTT(res2[i].Value[1], res2[i].Value[1])

		ringQ.MultByMonomial(res2[i].Value[0], i, res2[i].Value[0])
		ringQ.MultByMonomial(res2[i].Value[1], i, res2[i].Value[1])
		if i != 0 {
			ringQ.Neg(res2[i].Value[0], res2[i].Value[0])
			ringQ.Neg(res2[i].Value[1], res2[i].Value[1])
		}
		ringQ.NTT(res2[i].Value[0], res2[i].Value[0])
		ringQ.NTT(res2[i].Value[1], res2[i].Value[1])

		result[(n-i)%(n)] = res2[i]
	}
	elapse = time.Since(starttime)
	fmt.Println("TWEAK2 & replace", elapse)

	return result
}

// for test
func Transpose2(inputs []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, n int) []*rlwe.Ciphertext {
	cts := make([]*rlwe.Ciphertext, n)
	ringQ := params.RingQ().AtLevel(inputs[0].Level())
	ninv := ringQ.NewRNSScalarFromUInt64(uint64(n))
	ringQ.MFormRNSScalar(ninv, ninv)
	ringQ.Inverse(ninv)

	starttime := time.Now()
	for i := range cts {
		cts[i] = inputs[i]
		// ringQ.INTT(cts[i].Value[0], cts[i].Value[0])
		// ringQ.INTT(cts[i].Value[1], cts[i].Value[1])

		ringQ.MultByMonomial(cts[i].Value[0], i, cts[i].Value[0])
		ringQ.MultByMonomial(cts[i].Value[1], i, cts[i].Value[1])

		// ringQ.NTT(cts[i].Value[0], cts[i].Value[0])
		// ringQ.NTT(cts[i].Value[1], cts[i].Value[1])
	}
	elapse := time.Since(starttime)
	fmt.Println("mult by monomial", elapse)

	starttime = time.Now()
	aux := cts
	// aux := Tweak3(cts, params, eval, ringQ, n)
	elapse = time.Since(starttime)
	fmt.Println("TWEAK", elapse)

	starttime = time.Now()
	res := make([]*rlwe.Ciphertext, n)
	for i := range res {
		st := time.Now()
		idx, ch := ModInv(uint64(2*i+1), uint64(2*n))
		if !ch {
			fmt.Println("err ", i, " ", idx)
		}
		res[i] = aux[(idx-1)/2]

		// ringQ.INTT(res[i].Value[0], res[i].Value[0])
		// ringQ.INTT(res[i].Value[1], res[i].Value[1])
		ringQ.MForm(res[i].Value[0], res[i].Value[0])
		ringQ.MForm(res[i].Value[1], res[i].Value[1])

		ringQ.MulRNSScalarMontgomery(res[i].Value[0], ninv, res[i].Value[0])
		ringQ.MulRNSScalarMontgomery(res[i].Value[1], ninv, res[i].Value[1])

		ringQ.IMForm(res[i].Value[0], res[i].Value[0])
		ringQ.IMForm(res[i].Value[1], res[i].Value[1])
		// ringQ.NTT(res[i].Value[0], res[i].Value[0])
		// ringQ.NTT(res[i].Value[1], res[i].Value[1])
		res[i].IsNTT = false

		//uint64(2*i+1)
		if err := eval.Automorphism(res[i], uint64(21), res[i]); err != nil {
			fmt.Println(err)
		}
		res[i].IsNTT = true
		if i == 0 {
			el := time.Since(st)
			fmt.Println(el)
		}
	}
	elapse = time.Since(starttime)
	fmt.Println("Automorphism", elapse)

	starttime = time.Now()
	res2 := Tweak3(res, params, eval, ringQ, n)
	result := make([]*rlwe.Ciphertext, n)
	for i := range n {
		// ringQ.INTT(res2[i].Value[0], res2[i].Value[0])
		// ringQ.INTT(res2[i].Value[1], res2[i].Value[1])

		ringQ.MultByMonomial(res2[i].Value[0], i, res2[i].Value[0])
		ringQ.MultByMonomial(res2[i].Value[1], i, res2[i].Value[1])
		if i != 0 {
			ringQ.Neg(res2[i].Value[0], res2[i].Value[0])
			ringQ.Neg(res2[i].Value[1], res2[i].Value[1])
		}
		// ringQ.NTT(res2[i].Value[0], res2[i].Value[0])
		// ringQ.NTT(res2[i].Value[1], res2[i].Value[1])

		result[(n-i)%(n)] = res2[i]
	}
	elapse = time.Since(starttime)
	fmt.Println("TWEAK2 & replace", elapse)

	return result
}

func Transpose3(inputs []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, ringQ *ring.Ring, N, sparseN int, work, aux, res []*rlwe.Ciphertext) {

	ninv := ringQ.NewRNSScalarFromUInt64(uint64(sparseN))
	ringQ.MFormRNSScalar(ninv, ninv)
	ringQ.Inverse(ninv)
	ratio := N / sparseN
	idxarr := make([]uint64, sparseN)

	for i := range idxarr {
		idx, ch := ModInv(uint64(2*i+1), uint64(2*sparseN))
		_ = ch
		idxarr[i] = idx
	}

	for i := range inputs {
		ringQ.MultByMonomial(inputs[i].Value[0], i*ratio, inputs[i].Value[0])
		ringQ.MultByMonomial(inputs[i].Value[1], i*ratio, inputs[i].Value[1])
	}

	Tweak4(inputs, params, eval, ringQ, sparseN, work, 0, aux, 0)
	// var err error
	for i := range res {
		res[i] = aux[(idxarr[i]-1)/2].CopyNew()

		ringQ.MForm(res[i].Value[0], res[i].Value[0])
		ringQ.MForm(res[i].Value[1], res[i].Value[1])

		ringQ.MulRNSScalarMontgomery(res[i].Value[0], ninv, res[i].Value[0])
		ringQ.MulRNSScalarMontgomery(res[i].Value[1], ninv, res[i].Value[1])

		ringQ.IMForm(res[i].Value[0], res[i].Value[0])
		ringQ.IMForm(res[i].Value[1], res[i].Value[1])

		res[i].IsNTT = false
		// ringQ.NTT(res[i].Value[0], res[i].Value[0])
		// ringQ.NTT(res[i].Value[1], res[i].Value[1])

		galEl := uint64((2*i + 1))
		// var gk *rlwe.GaloisKey
		// if gk, err = eval.CheckAndGetGaloisKey(galEl); err != nil {
		// 	panic(err)
		// }
		eval.Automorphism(res[i], galEl, res[i])

		// ringQ.INTT(res[i].Value[0], res[i].Value[0])
		// ringQ.INTT(res[i].Value[1], res[i].Value[1])
		res[i].IsNTT = true

	}
	Tweak4(res, params, eval, ringQ, sparseN, work, 0, aux, 0)

	for idx := range sparseN {
		// idx := i / ratio
		i := idx * ratio

		ringQ.MultByMonomial(aux[idx].Value[0], i, aux[idx].Value[0])
		ringQ.MultByMonomial(aux[idx].Value[1], i, aux[idx].Value[1])
		if i != 0 {
			ringQ.Neg(aux[idx].Value[0], aux[idx].Value[0])
			ringQ.Neg(aux[idx].Value[1], aux[idx].Value[1])
		}
		res[(sparseN-idx)%(sparseN)] = aux[idx].CopyNew()
	}

}

func Transpose_Sparse(inputs []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, ringQ *ring.Ring, N, sparseN int, work, aux, res []*rlwe.Ciphertext) {
	// if inputs[0].Level() != work[0].Level() {
	// 	for i := range work {
	// 		work[i].Resize(1, inputs[0].Level())
	// 	}
	// }
	// if inputs[0].Level() != aux[0].Level() {
	// 	for i := range aux {
	// 		aux[i].Resize(1, inputs[0].Level())
	// 	}
	// }

	ratio := N / sparseN
	idxarr := make([]uint64, sparseN)

	for i := range idxarr {
		idx, ch := ModInv(uint64(2*i+1), uint64(2*sparseN))
		_ = ch
		idxarr[i] = idx
	}

	for i := range inputs {
		ringQ.MultByMonomial(inputs[i].Value[0], i*ratio, inputs[i].Value[0])
		ringQ.MultByMonomial(inputs[i].Value[1], i*ratio, inputs[i].Value[1])
	}

	Tweak4(inputs, params, eval, ringQ, sparseN, work, 0, aux, 0)
	var err error
	for i := range res {
		res[i] = aux[(idxarr[i]-1)/2].CopyNew()
		galEl := uint64((2*i + 1))
		var gk *rlwe.GaloisKey
		if gk, err = eval.CheckAndGetGaloisKey(galEl); err != nil {
			if util.SContext.Sk == nil && !util.Debug.IsDebug {
				panic(err)
			}
			elapse := time.Since(util.Debug.StartTime)
			util.Debug.AccTime += elapse
			galEl := uint64((2*i + 1))
			kgen_ := rlwe.NewKeyGenerator(params)
			// gk := kgen_.GenGaloisKeyNew(galEl, sk)
			gk = kgen_.GenGaloisKeyNew(galEl, util.SContext.Sk)
			util.Debug.StartTime = time.Now()
		}
		res[i].IsNTT = false
		Automorphism(eval, ringQ.AtLevel(inputs[i].Level()), res[i], galEl, gk, res[i])
		res[i].IsNTT = true
	}
	Tweak4(res, params, eval, ringQ, sparseN, work, 0, aux, 0)

	for idx := range sparseN {
		// idx := i / ratio
		i := idx * ratio

		ringQ.MultByMonomial(aux[idx].Value[0], i, aux[idx].Value[0])
		ringQ.MultByMonomial(aux[idx].Value[1], i, aux[idx].Value[1])
		if i != 0 {
			ringQ.Neg(aux[idx].Value[0], aux[idx].Value[0])
			ringQ.Neg(aux[idx].Value[1], aux[idx].Value[1])
		}
		res[(sparseN-idx)%(sparseN)] = aux[idx].CopyNew()
	}

}

func Automorphism(eval *hefloat.Evaluator, ringQ *ring.Ring, ctIn *rlwe.Ciphertext, galEl uint64, evk *rlwe.GaloisKey, opOut *rlwe.Ciphertext) {

	if galEl == 1 {
		if opOut != ctIn {
			opOut.Copy(ctIn)
		}
		return
	}

	ctTmp := &rlwe.Ciphertext{Element: rlwe.Element[ring.Poly]{Value: []ring.Poly{eval.BuffQP[0].Q, eval.BuffQP[1].Q}}}
	ctTmp.MetaData = ctIn.MetaData

	// eval.GadgetProduct(level, ctIn.Value[1], &evk.GadgetCiphertext, ctTmp)

	// ringQ.Add(ctTmp.Value[0], ctIn.Value[0], ctTmp.Value[0])

	KeySwitching(eval, ringQ, ctIn, &evk.EvaluationKey, ctTmp)

	ringQ.Automorphism(ctTmp.Value[0], galEl, opOut.Value[0])
	ringQ.Automorphism(ctTmp.Value[1], galEl, opOut.Value[1])

	*opOut.MetaData = *ctIn.MetaData
}

func KeySwitching(eval *hefloat.Evaluator, ringQ *ring.Ring, ctIn *rlwe.Ciphertext, evk *rlwe.EvaluationKey, opOut *rlwe.Ciphertext) {
	level := ctIn.Level()

	eval.GadgetProduct(level, ctIn.Value[1], &evk.GadgetCiphertext, opOut)

	ringQ.Add(opOut.Value[0], ctIn.Value[0], opOut.Value[0])
}

// for test
func Tweak3(cts []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, ringQ *ring.Ring, n int) []*rlwe.Ciphertext {
	if n == 1 {
		return []*rlwe.Ciphertext{cts[0]}
	}

	out := make([]*rlwe.Ciphertext, n)
	out[0] = cts[0]

	logn := bits.Len(uint(n)) - 1

	for l := 0; l < logn; l++ {
		powl := 1 << l

		temp := make([]*rlwe.Ciphertext, powl)
		den := powl * 2
		for j := 0; j < powl; j++ {
			idx := ((2*j + 1) * n) / den
			temp[j] = cts[idx]
		}

		aux := Tweak3(temp, params, eval, ringQ, powl)

		step := (params.MaxSlots() * 2) / powl

		for j := 0; j < powl; j++ {
			tmp := aux[j]

			shift := step * j
			ringQ.MultByMonomial(tmp.Value[0], shift, tmp.Value[0])
			ringQ.MultByMonomial(tmp.Value[1], shift, tmp.Value[1])

			sum, err := eval.AddNew(out[j], tmp)
			if err != nil {
				panic(fmt.Errorf("AddNew failed: %w", err))
			}
			diff, err := eval.SubNew(out[j], tmp)
			if err != nil {
				panic(fmt.Errorf("SubNew failed: %w", err))
			}
			out[j] = sum
			out[j+powl] = diff
		}
	}

	return out
}

// Original recursive Tweak4 (restored). Writes output into out[outOff:outOff+n]
func Tweak4(cts []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, ringQ *ring.Ring, n int, work []*rlwe.Ciphertext, workOff int, out []*rlwe.Ciphertext, outOff int) {
	out[outOff] = cts[0]
	if n == 1 {
		return
	}

	logn := bits.Len(uint(n)) - 1
	maxPowl := n / 2
	tempBuf := work[workOff : workOff+maxPowl]

	for l := 0; l < logn; l++ {
		powl := 1 << l
		temp := tempBuf[:powl]
		den := powl * 2
		for j := 0; j < powl; j++ {
			idx := ((2*j + 1) * n) / den
			temp[j] = cts[idx]
		}
		Tweak4(temp, params, eval, ringQ, powl, work, workOff+powl, out, outOff+powl)
		res := out[outOff : outOff+n]

		step := (params.MaxSlots() * 2) / powl

		for j := 0; j < powl; j++ {
			work[len(work)-1].Copy(res[powl+j])
			tmp := work[len(work)-1]

			shift := step * j
			ringQ.MultByMonomial(tmp.Value[0], shift, tmp.Value[0])
			ringQ.MultByMonomial(tmp.Value[1], shift, tmp.Value[1])

			if err := eval.Sub(res[j], tmp, res[j+powl]); err != nil {
				panic(fmt.Errorf("Sub failed: %w", err))
			}

			if err := eval.Add(res[j], tmp, res[j]); err != nil {
				panic(fmt.Errorf("Add failed: %w", err))
			}
		}
	}
}

// Tweak4_nonrecur: iterative, non-recursive version using explicit stack.
func Tweak4_nonrecur(cts []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, ringQ *ring.Ring, n int, work []*rlwe.Ciphertext, workOff int, out []*rlwe.Ciphertext, outOff int) {
	type frame struct {
		cts     []*rlwe.Ciphertext
		n       int
		workOff int
		outOff  int
		l       int
		logn    int
		maxPowl int
		started bool
		waiting bool
		childP  int
	}

	stack := make([]frame, 0, 32)
	stack = append(stack, frame{cts: cts, n: n, workOff: workOff, outOff: outOff})
	temp := ringQ.NewPoly()

	for len(stack) > 0 {
		fi := &stack[len(stack)-1]

		if !fi.started {
			out[fi.outOff] = fi.cts[0]
			fi.started = true
			if fi.n == 1 {
				stack = stack[:len(stack)-1]
				continue
			}
			fi.logn = bits.Len(uint(fi.n)) - 1
			fi.maxPowl = fi.n / 2
			fi.l = 0
			fi.waiting = false
		}

		if fi.waiting {
			powl := fi.childP
			res := out[fi.outOff : fi.outOff+fi.n]
			step := (params.MaxSlots() * 2) / powl

			for j := 0; j < powl; j++ {
				work[len(work)-1].Copy(res[powl+j])
				tmp := work[len(work)-1]

				shift := step * j
				MultByMonomial(ringQ, tmp.Value[0], shift, tmp.Value[0], temp)
				MultByMonomial(ringQ, tmp.Value[1], shift, tmp.Value[1], temp)

				if err := eval.Sub(res[j], tmp, res[j+powl]); err != nil {
					panic(fmt.Errorf("Sub failed: %w", err))
				}

				if err := eval.Add(res[j], tmp, res[j]); err != nil {
					panic(fmt.Errorf("Add failed: %w", err))
				}
			}

			fi.waiting = false
			fi.l++
			continue
		}

		if fi.l >= fi.logn {
			stack = stack[:len(stack)-1]
			continue
		}

		powl := 1 << fi.l
		den := powl * 2
		temp := work[fi.workOff : fi.workOff+fi.maxPowl][:powl]
		for j := 0; j < powl; j++ {
			idx := ((2*j + 1) * fi.n) / den
			temp[j] = fi.cts[idx]
		}

		fi.waiting = true
		fi.childP = powl

		child := frame{cts: temp, n: powl, workOff: fi.workOff + powl, outOff: fi.outOff + powl}
		stack = append(stack, child)
	}
}

// MultByMonomial evaluates p2 = p1 * X^k coefficient-wise in the ring.
func MultByMonomial(r *ring.Ring, p1 ring.Poly, k int, p2 ring.Poly, tmpx ring.Poly) {

	N := r.N()

	shift := (k + (N << 1)) % (N << 1)
	level := r.Level()

	if shift == 0 {

		for i := range r.SubRings[:level+1] {
			p1tmp, p2tmp := p1.Coeffs[i], p2.Coeffs[i]
			for j := 0; j < N; j++ {
				p2tmp[j] = p1tmp[j]
			}
		}

	} else {

		if shift < N {

			for i := range r.SubRings[:level+1] {
				p1tmp, tmpxT := p1.Coeffs[i], tmpx.Coeffs[i]
				for j := 0; j < N; j++ {
					tmpxT[j] = p1tmp[j]
				}
			}

		} else {

			for i, s := range r.SubRings[:level+1] {
				qi := s.Modulus
				p1tmp, tmpxT := p1.Coeffs[i], tmpx.Coeffs[i]
				for j := 0; j < N; j++ {
					tmpxT[j] = qi - p1tmp[j]
				}
			}
		}

		shift %= N

		for i, s := range r.SubRings[:level+1] {
			qi := s.Modulus
			p2tmp, tmpxT := p2.Coeffs[i], tmpx.Coeffs[i]
			for j := 0; j < shift; j++ {
				p2tmp[j] = qi - tmpxT[N-shift+j]
			}
		}

		for i := range r.SubRings[:level+1] {
			p2tmp, tmpxT := p2.Coeffs[i], tmpx.Coeffs[i]
			for j := shift; j < N; j++ {
				p2tmp[j] = tmpxT[j-shift]

			}
		}
	}
}

// TODO
func TransposeInplace(inputs []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, encoder *hefloat.Encoder, n int) {
	ringQ := params.RingQ().AtLevel(inputs[0].Level())
	ninv := ringQ.NewRNSScalarFromUInt64(uint64(n))
	ringQ.MFormRNSScalar(ninv, ninv)
	ringQ.Inverse(ninv)

	starttime := time.Now()
	for i := range inputs {
		ringQ.MultByMonomial(inputs[i].Value[0], i, inputs[i].Value[0])
		ringQ.MultByMonomial(inputs[i].Value[1], i, inputs[i].Value[1])
	}
	elapse := time.Since(starttime)
	fmt.Println("mult by monomial", elapse)

	starttime = time.Now()
	TweakInplace(inputs, params, eval, n)
	elapse = time.Since(starttime)
	fmt.Println("TWEAK", elapse)

	starttime = time.Now()
	res := make([]*rlwe.Ciphertext, n)
	for i := range res {
		idx, ch := ModInv(uint64(2*i+1), uint64(2*n))
		if !ch {
			fmt.Println("err ", i, " ", idx)
		}
		res[i] = inputs[(idx-1)/2].CopyNew()

		ringQ.MForm(res[i].Value[0], res[i].Value[0])
		ringQ.MForm(res[i].Value[1], res[i].Value[1])

		ringQ.MulRNSScalarMontgomery(res[i].Value[0], ninv, res[i].Value[0])
		ringQ.MulRNSScalarMontgomery(res[i].Value[1], ninv, res[i].Value[1])

		ringQ.IMForm(res[i].Value[0], res[i].Value[0])
		ringQ.IMForm(res[i].Value[1], res[i].Value[1])
		res[i].IsNTT = false

		//uint64(2*i+1)
		if err := eval.Automorphism(res[i], uint64(2*i+1), res[i]); err != nil {
			fmt.Println(err)
		}
		res[i].IsNTT = true
	}
	elapse = time.Since(starttime)
	fmt.Println("Automorphism", elapse)

	starttime = time.Now()
	TweakInplace(res, params, eval, n)
	for i := range n {
		// ringQ.INTT(res2[i].Value[0], res2[i].Value[0])
		// ringQ.INTT(res2[i].Value[1], res2[i].Value[1])

		ringQ.MultByMonomial(res[i].Value[0], i, res[i].Value[0])
		ringQ.MultByMonomial(res[i].Value[1], i, res[i].Value[1])
		if i != 0 {
			ringQ.Neg(res[i].Value[0], res[i].Value[0])
			ringQ.Neg(res[i].Value[1], res[i].Value[1])
		}
		// ringQ.NTT(res2[i].Value[0], res2[i].Value[0])
		// ringQ.NTT(res2[i].Value[1], res2[i].Value[1])

		inputs[(n-i)%(n)] = res[i]
	}
	elapse = time.Since(starttime)
	fmt.Println("TWEAK2 & replace", elapse)
}

// cts[0:n]을 제자리 갱신. n=2^L.
func TweakInplace(cts []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, n int) {
	if n <= 0 || n > len(cts) {
		panic("Tweak3: invalid n")
	}
	if n&(n-1) != 0 {
		panic("Tweak3: n must be power of two")
	}
	if n == 1 {
		cts[0] = cts[0].CopyNew()
		return
	}

	// 1) 원본 스냅샷
	snap := make([]*rlwe.Ciphertext, n)
	for i := 0; i < n; i++ {
		snap[i] = cts[i].CopyNew()
	}

	L := bits.Len(uint(n)) - 1
	ringQ := params.RingQ().AtLevel(snap[0].Level())

	// 누적 출력은 cts에 기록
	cts[0] = snap[0].CopyNew()

	for l := 0; l < L; l++ {

		powl := 1 << l
		den := powl * 2
		step := (params.MaxSlots() * 2) / powl

		// temp = 원본 스냅샷에서만 읽음
		temp := make([]*rlwe.Ciphertext, powl)
		for j := 0; j < powl; j++ {
			idx := ((2*j + 1) * n) / den
			temp[j] = snap[idx].CopyNew()
		}
		aux := Tweak(temp, params, eval, nil, powl)

		for j := 0; j < powl; j++ {
			tmp := aux[j].CopyNew()
			shift := step * j
			ringQ.MultByMonomial(tmp.Value[0], shift, tmp.Value[0])
			ringQ.MultByMonomial(tmp.Value[1], shift, tmp.Value[1])

			sum, err := eval.AddNew(cts[j], tmp)
			if err != nil {
				panic(fmt.Errorf("AddNew failed: %w", err))
			}
			diff, err := eval.SubNew(cts[j], tmp)
			if err != nil {
				panic(fmt.Errorf("SubNew failed: %w", err))
			}
			cts[j] = sum
			cts[j+powl] = diff
		}
	}
}

func printMemUsage() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	fmt.Println("////////////////////////////////////////////////////")
	fmt.Printf("Alloc = %v MiB\n", bToMb(m.Alloc))
	fmt.Printf("Sys = %v MiB\n", bToMb(m.Sys))
	fmt.Printf("NumGC = %v\n", m.NumGC)
	fmt.Println("////////////////////////////////////////////////////")
}

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}
