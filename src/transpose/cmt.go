package transpose

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func Tweak(cts []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, encoder *hefloat.Encoder, n int) []*rlwe.Ciphertext {
	if n == 1 {
		return cts
	}

	cts_ := make([]*rlwe.Ciphertext, n)
	cts_[0] = cts[0].CopyNew()
	logn := int(math.Floor(math.Log2(float64(n))))

	for l := range logn {
		powl := int(math.Pow(2, float64(l)))
		temp := make([]*rlwe.Ciphertext, powl)
		for j := range powl {
			temp[j] = cts[(2*j+1)*n/(powl*2)].CopyNew()
		}
		aux := Tweak(temp, params, eval, encoder, powl)
		for j := range powl {
			arr := make([]float64, params.MaxSlots()*2)
			arr[params.MaxSlots()*2/powl*j] = 1
			pt := hefloat.NewPlaintext(params, params.MaxLevel())
			pt.IsBatched = false
			encoder.Encode(arr, pt)

			tmp, _ := eval.MulNew(aux[j], pt)
			if tmp.Level() == 0 {
				fmt.Println("no")
			}
			eval.Rescale(tmp, tmp)

			cts_[j+powl], _ = eval.SubNew(cts_[j], tmp)
			cts_[j], _ = eval.AddNew(cts_[j], tmp)
		}
	}

	return cts_
}

func Tweak2(cts []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, encoder *hefloat.Encoder, n int) []*rlwe.Ciphertext {
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
		aux := Tweak2(temp, params, eval, encoder, powl)
		for j := range powl {

			//tmp, _ := eval.MulNew(aux[j], pt)
			tmp := aux[j].CopyNew()

			ringQ.INTT(tmp.Value[0], tmp.Value[0])
			ringQ.INTT(tmp.Value[1], tmp.Value[1])
			ringQ.MultByMonomial(tmp.Value[0], params.MaxSlots()*2/powl*j, tmp.Value[0])
			ringQ.MultByMonomial(tmp.Value[1], params.MaxSlots()*2/powl*j, tmp.Value[1])
			if tmp.Level() == 0 {
				fmt.Println("no")
			}

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

	for i := range cts {
		cts[i] = inputs[i].CopyNew()
		ringQ.INTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.INTT(cts[i].Value[1], cts[i].Value[1])

		ringQ.MultByMonomial(cts[i].Value[0], i, cts[i].Value[0])
		ringQ.MultByMonomial(cts[i].Value[1], i, cts[i].Value[1])

		ringQ.NTT(cts[i].Value[0], cts[i].Value[0])
		ringQ.NTT(cts[i].Value[1], cts[i].Value[1])

	}
	aux := Tweak2(cts, params, eval, encoder, n)
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

		if err := eval.Automorphism(res[i], uint64(2*i+1), res[i]); err != nil {
			fmt.Println(err)
		}

	}

	res2 := Tweak2(res, params, eval, encoder, n)
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

	return result
}
