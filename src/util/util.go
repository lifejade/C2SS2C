package util

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/bits"
	"os"
	"runtime"
	"time"

	"github.com/shirou/gopsutil/v3/process"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils/bignum"
)

type DebugContext struct {
	IsDebug   bool
	AccTime   time.Duration
	StartTime time.Time
}

var Debug DebugContext

type Val interface {
	~float64 | ~complex128
}

type SecretContext struct {
	Sk        *rlwe.SecretKey
	Decryptor *rlwe.Decryptor
	Values    [][]float64
}

var SContext SecretContext

func DebugPrec(cts []*rlwe.Ciphertext, params hefloat.Parameters, encoder *hefloat.Encoder, decryptor *rlwe.Decryptor, ptvalues [][]float64, ratio int, IsBatched bool) {
	if decryptor == nil {
		return
	}
	fmt.Println()
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
	fmt.Println("Debug Precision")
	var n int
	if IsBatched {
		n = params.MaxSlots()
	} else {
		n = params.N()
	}

	maxerr := 0.0
	accerr := 0.0
	value := make([]float64, n)
	for i := range cts {
		cttmp := cts[i].CopyNew()
		cttmp.IsBatched = IsBatched
		dept := decryptor.DecryptNew(cttmp)
		encoder.Decode(dept, value)

		// fmt.Println(value)
		// fmt.Println(ptvalues[i])
		// fmt.Println()

		for j := range ptvalues[i] {
			val := math.Abs(value[j*ratio] - ptvalues[i][j])
			if val > maxerr {
				maxerr = val
			}
			accerr += (val * val)
		}
	}
	accerr = math.Sqrt(accerr)

	fmt.Println("LInf-Norm Err : ", maxerr)
	fmt.Println("LInf-Norm Err Bit : ", -math.Log2(maxerr))
	fmt.Println("L2-Norm Err : ", accerr)
	fmt.Println("L2-Norm Err Bit : ", -math.Log2(accerr))
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
	fmt.Println()
}

func DebugCTS(cts []*rlwe.Ciphertext, params hefloat.Parameters, encoder *hefloat.Encoder, decryptor *rlwe.Decryptor) {
	if decryptor == nil {
		return
	}
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////")
	fmt.Println("debug cts")
	var n int
	if cts[0].IsBatched {
		n = params.MaxSlots()
	} else {
		n = params.N()
	}
	value := make([]float64, n)
	for i := range cts {
		dept := decryptor.DecryptNew(cts[i])
		encoder.Decode(dept, value)
		for j := range value {
			fmt.Print(value[j], " ")
		}
		fmt.Println()
	}
	fmt.Println("debug cts end")
	fmt.Println("////////////////////////////////////////////////////////////////////////////////////")
}

func MulImag(eval *hefloat.Evaluator, op0 *rlwe.Ciphertext, opOut *rlwe.Ciphertext) (err error) {

	_, level, err := eval.InitOutputUnaryOp(op0.El(), opOut.El())
	if err != nil {
		return fmt.Errorf("cannot Mul: %w", err)
	}

	opOut.Resize(op0.Degree(), level)

	// Gets the ring at the target level
	ringQ := eval.GetParameters().RingQ().AtLevel(level)

	N := eval.GetParameters().N()
	for i := range op0.Value {
		// ringQ.MulDoubleRNSScalar(op0.Value[i], RNSReal, RNSImag, opOut.Value[i])
		// ringQ.MulRNSScalarMontgomery(op0.Value[i], RNSReal, opOut.Value[i])
		ringQ.MultByMonomial(op0.Value[i], N/2, op0.Value[i])
	}

	return nil
}

func bigComplexToRNSScalar(r *ring.Ring, cmplx *bignum.Complex) (RNSReal, RNSImag ring.RNSScalar) {

	real := new(big.Int)
	if cmplx[0] != nil {
		r := cmplx[0]

		if cmp := cmplx[0].Cmp(new(big.Float)); cmp > 0 {
			r.Add(r, new(big.Float).SetFloat64(0.5))
		} else if cmp < 0 {
			r.Sub(r, new(big.Float).SetFloat64(0.5))
		}

		r.Int(real)
	}

	imag := new(big.Int)
	if cmplx[1] != nil {
		i := cmplx[1]

		if cmp := cmplx[1].Cmp(new(big.Float)); cmp > 0 {
			i.Add(i, new(big.Float).SetFloat64(0.5))
		} else if cmp < 0 {
			i.Sub(i, new(big.Float).SetFloat64(0.5))
		}

		i.Int(imag)
	}

	return r.NewRNSScalarFromBigint(real), r.NewRNSScalarFromBigint(imag)
}

func Mul_ScaleExact(eval *hefloat.Evaluator, op0 *rlwe.Ciphertext, op1 float64, opOut *rlwe.Ciphertext, scale rlwe.Scale) (err error) {

	_, level, err := eval.InitOutputUnaryOp(op0.El(), opOut.El())
	if err != nil {
		return fmt.Errorf("cannot Mul: %w", err)
	}

	opOut.Resize(op0.Degree(), level)

	// Gets the ring at the target level
	ringQ := eval.GetParameters().RingQ().AtLevel(level)

	// Convertes the *bignum.Complex to a complex RNS scalar
	RNSReal := bigFloatToRNSScalar(ringQ, &scale.Value, op1)
	// fmt.Println(RNSReal)
	// for i, s := range eval.GetParameters().RingQ().SubRings[:level+1] {
	// 	RNSImag[i] = ring.MRed(RNSImag[i], s.RootsForward[1], s.Modulus, s.MRedConstant)
	// 	RNSReal[i], RNSImag[i] = ring.CRed(RNSReal[i]+RNSImag[i], s.Modulus), ring.CRed(RNSReal[i]+s.Modulus-RNSImag[i], s.Modulus)
	// }
	// fmt.Println(RNSReal)
	// fmt.Println(RNSImag)
	for i, s := range ringQ.SubRings[:level+1] {
		RNSReal[i] = ring.MForm(RNSReal[i], s.Modulus, s.BRedConstant)
	}
	for i := range op0.Value {

		//ringQ.MulDoubleRNSScalar(op0.Value[i], RNSReal, RNSImag, opOut.Value[i])
		ringQ.MulRNSScalarMontgomery(op0.Value[i], RNSReal, opOut.Value[i])
	}

	// Copies the metadata on the output
	opOut.Scale = op0.Scale.Mul(scale) // updates the scaling factor

	return nil
}

func Mul_(eval *hefloat.Evaluator, op0 *rlwe.Ciphertext, op1 float64, opOut *rlwe.Ciphertext) (err error) {

	_, level, err := eval.InitOutputUnaryOp(op0.El(), opOut.El())
	if err != nil {
		return fmt.Errorf("cannot Mul: %w", err)
	}

	opOut.Resize(op0.Degree(), level)

	// Gets the ring at the target level
	ringQ := eval.GetParameters().RingQ().AtLevel(level)

	var scale rlwe.Scale
	scale = rlwe.NewScale(ringQ.SubRings[level].Modulus) // Current modulus scaling factor

	// If DefaultScalingFactor > 2^60, then multiple moduli are used per single rescale
	// thus continues multiplying the scale with the appropriate number of moduli
	for i := 1; i < eval.GetParameters().LevelsConsumedPerRescaling(); i++ {
		scale = scale.Mul(rlwe.NewScale(ringQ.SubRings[level-i].Modulus))
	}

	// Convertes the *bignum.Complex to a complex RNS scalar
	RNSReal := bigFloatToRNSScalar(ringQ, &scale.Value, op1)
	// fmt.Println(RNSReal)
	// for i, s := range eval.GetParameters().RingQ().SubRings[:level+1] {
	// 	RNSImag[i] = ring.MRed(RNSImag[i], s.RootsForward[1], s.Modulus, s.MRedConstant)
	// 	RNSReal[i], RNSImag[i] = ring.CRed(RNSReal[i]+RNSImag[i], s.Modulus), ring.CRed(RNSReal[i]+s.Modulus-RNSImag[i], s.Modulus)
	// }
	// fmt.Println(RNSReal)
	// fmt.Println(RNSImag)
	for i, s := range ringQ.SubRings[:level+1] {
		RNSReal[i] = ring.MForm(RNSReal[i], s.Modulus, s.BRedConstant)
	}
	for i := range op0.Value {

		//ringQ.MulDoubleRNSScalar(op0.Value[i], RNSReal, RNSImag, opOut.Value[i])
		ringQ.MulRNSScalarMontgomery(op0.Value[i], RNSReal, opOut.Value[i])
	}

	// Copies the metadata on the output
	opOut.Scale = op0.Scale.Mul(scale) // updates the scaling factor

	return nil
}
func bigFloatToRNSScalar(r *ring.Ring, scale *big.Float, value float64) (RNSReal ring.RNSScalar) {

	if scale == nil {
		scale = new(big.Float).SetFloat64(1)
	}

	real := new(big.Int)
	v := big.NewFloat(value)
	res := new(big.Float).Mul(v, scale)

	if cmp := v.Cmp(new(big.Float)); cmp > 0 {
		res.Add(res, new(big.Float).SetFloat64(0.5))
	} else if cmp < 0 {
		res.Sub(res, new(big.Float).SetFloat64(0.5))
	}

	res.Int(real)

	return r.NewRNSScalarFromBigint(real)
}

func Rescale_NonNTT(eval *hefloat.Evaluator, op0, opOut *rlwe.Ciphertext) (err error) {

	if op0.MetaData == nil || opOut.MetaData == nil {
		return fmt.Errorf("cannot RescaleTo: op0.MetaData or opOut.MetaData is nil")
	}
	minScale := eval.GetParameters().DefaultScale()

	if minScale.Cmp(rlwe.NewScale(0)) != 1 {
		return fmt.Errorf("cannot RescaleTo: minScale is <0")
	}

	minScale = minScale.Div(rlwe.NewScale(2))

	if op0.Scale.Cmp(rlwe.NewScale(0)) != 1 {
		return fmt.Errorf("cannot RescaleTo: ciphertext scale is <0")
	}

	if op0.Level() == 0 {
		return fmt.Errorf("cannot RescaleTo: input Ciphertext already at level 0")
	}

	*opOut.MetaData = *op0.MetaData

	newLevel := op0.Level()

	ringQ := eval.GetParameters().RingQ().AtLevel(op0.Level())

	// Divides the scale by each moduli of the modulus chain as long as the scale isn't smaller than minScale/2
	// or until the output Level() would be zero
	var nbRescales int
	for newLevel >= 0 {

		scale := opOut.Scale.Div(rlwe.NewScale(ringQ.SubRings[newLevel].Modulus))

		if scale.Cmp(minScale) == -1 {
			break
		}

		opOut.Scale = scale

		nbRescales++
		newLevel--
	}

	if op0 != opOut {
		opOut.Resize(op0.Degree(), op0.Level()-nbRescales)
	}

	if nbRescales > 0 {
		for i := range opOut.Value {
			ringQ.DivRoundByLastModulusMany(nbRescales, op0.Value[i], eval.BuffQ()[0], opOut.Value[i])
		}
		opOut.Resize(opOut.Degree(), newLevel)
	} else {
		if op0 != opOut {
			opOut.Copy(op0)
		}
	}

	return nil
}

func PrintMemUsage() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	fmt.Println("////////////////////////////////////////////////////")
	fmt.Printf("Alloc = %v MiB\n", bToMb(m.Alloc))
	fmt.Printf("Sys = %v MiB\n", bToMb(m.Sys))
	fmt.Printf("NumGC = %v\n", m.NumGC)
	fmt.Println("////////////////////////////////////////////////////")

	// 현재 실행 중인 프로세스 정보 가져오기
	p, _ := process.NewProcess(int32(os.Getpid()))

	// 실제 물리 메모리 점유량 (RSS) 가져오기
	memInfo, _ := p.MemoryInfo()

	// RSS: 실제 물리 램 점유량
	// VMS: 가상 메모리 전체 (현재 보시는 24TB 수치는 여기에 해당)
	fmt.Printf("실제 물리 메모리 (RSS): %v MiB\n", memInfo.RSS/1024/1024)
	fmt.Printf("가상 메모리 (VMS): %v MiB\n", memInfo.VMS/1024/1024)
}

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}

func BitReverse(i, m int) int {
	rev := 0
	for j := 0; j < m; j++ {
		rev = (rev << 1) | (i & 1)
		i >>= 1
	}
	return rev
}

func CtZero(params hefloat.Parameters, encoder *hefloat.Encoder, encryptor *rlwe.Encryptor) *rlwe.Ciphertext {
	value := make([]float64, params.MaxSlots())
	pt := hefloat.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(value, pt)
	ct, _ := encryptor.EncryptNew(pt)
	return ct
}

func FindPrimes(startbit, bitlen, maxlogsize int) ([]uint32, bool) {
	res := make([]uint32, 1)
	isOverMax := false
	var err error
	if res[0], err = smallestPrimeInBitRange(startbit); err != nil {
		res[0] = 0
		return res, isOverMax
	}
	size := math.Log2(float64(res[0]))
	value := res[0]

	for true {
		value += 2
		iscoprime := true
		for i := range res {
			if gcd(res[i], value) != 1 {
				iscoprime = false
				break
			}
		}
		if iscoprime {
			size += math.Log2(float64(value))
			res = append(res, value)
		}
		if size > float64(maxlogsize) {
			isOverMax = true
			break
		}
		if bits.Len32(value) > bitlen {
			break
		}
	}
	return res, isOverMax
}

func smallestPrimeInBitRange(n int) (uint32, error) {
	if n < 2 || n >= 32 {
		return 0, errors.New("n must satisfy 2 <= n < 32")
	}

	start := uint32(1) << (n - 1)
	end := uint32(1) << n

	// ensure odd (except for 2)
	if start > 2 && start%2 == 0 {
		start++
	}

	for x := start; x < end; x += 2 {
		if isPrime(x) {
			return x, nil
		}
	}

	return 0, errors.New("no prime found in range")
}
func isPrime(x uint32) bool {
	if x < 2 {
		return false
	}
	if x == 2 {
		return true
	}
	if x%2 == 0 {
		return false
	}
	for i := uint32(3); i*i <= x; i += 2 {
		if x%i == 0 {
			return false
		}
	}
	return true
}

func gcd(a, b uint32) uint32 {
	if a < 0 {
		a = -a
	}
	if b < 0 {
		b = -b
	}
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
