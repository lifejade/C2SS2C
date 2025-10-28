package transpose

import (
	"fmt"
	"math"
	"math/bits"
	"runtime"
	"time"

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
	aux := Tweak2(cts, params, eval, encoder, n)
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
	elapse = time.Since(starttime)
	fmt.Println("TWEAK2 & replace", elapse)

	return result
}

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
	aux := Tweak3(cts, params, eval, ringQ, n)
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
		if err := eval.Automorphism(res[i], uint64(2*i+1), res[i]); err != nil {
			fmt.Println(err)
		}
		res[i].IsNTT = true
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

func Tweak3(cts []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, ringQ *ring.Ring, n int) []*rlwe.Ciphertext {
	if n == 1 {
		return []*rlwe.Ciphertext{cts[0].CopyNew()}
	}

	out := make([]*rlwe.Ciphertext, n)
	out[0] = cts[0]

	logn := bits.Len(uint(n)) - 1

	for l := 0; l < logn; l++ {
		powl := 1 << l

		// temp[j] = cts[idx(j,l)] 사전 구성
		temp := make([]*rlwe.Ciphertext, powl)
		den := powl * 2
		for j := 0; j < powl; j++ {
			// ((2*j+1)*n)/(powl*2)  인덱스는 [0,n) 보장
			idx := ((2*j + 1) * n) / den
			temp[j] = cts[idx]
		}

		aux := Tweak3(temp, params, eval, ringQ, powl)

		// monomial shift 간결화: ((MaxSlots()*2)/powl) * j
		step := (params.MaxSlots() * 2) / powl

		for j := 0; j < powl; j++ {
			// aux[j]를 보존하기 위해 복사 후 in-place shift
			tmp := aux[j]

			shift := step * j
			ringQ.MultByMonomial(tmp.Value[0], shift, tmp.Value[0])
			ringQ.MultByMonomial(tmp.Value[1], shift, tmp.Value[1])

			// out[j]는 반드시 존재. out[j+powl]은 새로 생성.
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

func Tweak3_(cts []*rlwe.Ciphertext, params hefloat.Parameters, eval *hefloat.Evaluator, ringQ *ring.Ring, n int, result []*rlwe.Ciphertext) {
	if n == 1 {
		result[0] = cts[0].CopyNew()
		return
	}

	out := make([]*rlwe.Ciphertext, n)
	out[0] = cts[0]

	logn := bits.Len(uint(n)) - 1

	for l := 0; l < logn; l++ {
		powl := 1 << l

		temp := make([]*rlwe.Ciphertext, powl)
		den := powl * 2
		for j := 0; j < powl; j++ {
			// ((2*j+1)*n)/(powl*2)  인덱스는 [0,n) 보장
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
		aux := Tweak2(temp, params, eval, nil, powl)

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
		fmt.Println(l, " : ")
		printMemUsage()
		fmt.Println()
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
