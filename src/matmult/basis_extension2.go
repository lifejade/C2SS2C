package matmult

import (
	"fmt"
	"math"
	"math/big"
	"math/bits"
	"unsafe"

	"github.com/tuneinsight/lattigo/v5/ring"
)

// BasisExtender stores the necessary parameters for RNS basis extension.
// The used algorithm is from https://eprint.iacr.org/2018/117.pdf.
type BasisExtender struct {
	ringQ            *ring.Ring
	ringP            *ring.Ring
	dicConstantsQtoP map[Key]ModUpConstants
	dicConstantsPtoQ map[Key]ModUpConstants
	buffQ            ring.Poly
	buffP            ring.Poly
}

type Key struct {
	From int
	To   int
}

// NewBasisExtender creates a new BasisExtender, enabling RNS basis extension from Q to P and P to Q.
func NewBasisExtender(ringQ, ringP *ring.Ring, QtoP, PtoQ []Key) (be *BasisExtender) {

	be = new(BasisExtender)

	be.ringQ = ringQ
	be.ringP = ringP

	Q := ringQ.ModuliChain()
	P := ringP.ModuliChain()

	be.dicConstantsQtoP = make(map[Key]ModUpConstants)
	for _, key := range QtoP {
		be.dicConstantsQtoP[key] = GenModUpConstants(Q[:key.From+1], P[:key.To+1])
	}
	be.dicConstantsPtoQ = make(map[Key]ModUpConstants)
	for _, key := range PtoQ {
		be.dicConstantsPtoQ[key] = GenModUpConstants(P[:key.From+1], Q[:key.To+1])
	}

	be.buffQ = ringQ.NewPoly()
	be.buffP = ringP.NewPoly()

	return
}

// ModUpConstants stores the necessary parameters for RNS basis extension.
type ModUpConstants struct {
	// Parameters for basis extension from Q to P
	// (Q/Qi)^-1) (mod each Qi) (in Montgomery form)
	qoverqiinvqi []uint64
	// alpha_i (mod each Pj) (in Montgomery form)
	alphaimodp [][]uint64
	// beta_i / qi (-1/2 < x <1/2)
	betaioverqi []float64
}

// GenModUpConstants generates the ModUpConstants for basis extension from Q to P and P to Q.
func GenModUpConstants(Q, P []uint64) ModUpConstants {

	bredQ := make([][]uint64, len(Q))
	mredQ := make([]uint64, len(Q))
	bredP := make([][]uint64, len(P))
	mredP := make([]uint64, len(P))

	for i := range Q {
		bredQ[i] = ring.BRedConstant(Q[i])
		mredQ[i] = ring.MRedConstant(Q[i])
	}

	for i := range P {
		bredP[i] = ring.BRedConstant(P[i])
		mredP[i] = ring.MRedConstant(P[i])
	}
	qoverqiinvqi := make([]uint64, len(Q))
	alphaimodp := make([][]uint64, len(P))
	for i := range alphaimodp {
		alphaimodp[i] = make([]uint64, len(Q))
	}
	betaioverqi := make([]float64, len(Q))

	var qiStar uint64
	for i, qi := range Q {

		qiStar = ring.MForm(1, qi, bredQ[i])

		for j := 0; j < len(Q); j++ {
			if j != i {
				qiStar = ring.MRed(qiStar, ring.MForm(Q[j], qi, bredQ[i]), qi, mredQ[i])
			}
		}

		// (Q/Qi)^-1) * r (mod Qi) (in Montgomery form)
		qoverqiinvqi[i] = ring.ModexpMontgomery(qiStar, int(qi-2), qi, mredQ[i], bredQ[i])
	}
	QQ, RQ := ProductDivModBig(P, Q)
	fmt.Println(QQ, RQ)
	for j := range Q {
		temp := new(big.Float)
		temp = temp.Quo(new(big.Float).SetInt(RQ[j]), new(big.Float).SetUint64((Q[j])))

		// biggerthanhalf := uint64(0)
		// // fmt.Println(betaioverqi)
		// if temp.Cmp(new(big.Float).SetFloat64(0.5)) > 0 {
		// 	biggerthanhalf++
		// 	temp.Sub(temp, new(big.Float).SetInt64(1))
		// }
		betaioverqi[j], _ = temp.Float64()
		// fmt.Println(betaioverqi)

		for i := range P {
			value := new(big.Int)
			// value.Add(QQ[j], new(big.Int).SetUint64(biggerthanhalf))
			value.Mod(QQ[j], new(big.Int).SetUint64(P[i]))
			alphaimodp[i][j] = ring.MForm(value.Uint64(), P[i], bredP[i])
			// fmt.Println("alpha non Mform", value.Uint64())
			// fmt.Println("alpha Mform", alphaimodp[i][j], " of ", P[i])
		}
	}

	return ModUpConstants{qoverqiinvqi: qoverqiinvqi, alphaimodp: alphaimodp, betaioverqi: betaioverqi}
}

func ProductDivModBig(ps_ []uint64, qi_ []uint64) (Q, R []*big.Int) {

	// 몫(Q)과 나머지(R)를 저장할 슬라이스를 초기화합니다.
	Q = make([]*big.Int, len(qi_))
	R = make([]*big.Int, len(qi_))

	// qi의 각 요소(q)에 대해 반복합니다.
	for i, q := range qi_ {
		// 현재 몫과 나머지를 0으로 초기화합니다.
		Q[i] = new(big.Int).SetInt64(0)
		R[i] = new(big.Int).SetInt64(0)

		// ps의 각 요소(p)에 대해 반복합니다.
		for j, p := range ps_ {
			// 임시 *big.Int 변수들을 초기화합니다.
			tempMul := new(big.Int)
			tempQ := new(big.Int)
			tempR := new(big.Int)
			if j == 0 {
				// ps의 첫 번째 요소일 경우, p * 1을 q로 나눕니다.
				tempMul.Mul(new(big.Int).SetUint64(p), big.NewInt(1))
				tempQ.Div(tempMul, new(big.Int).SetUint64(q))
				tempR.Mod(tempMul, new(big.Int).SetUint64(q))

				// 첫 번째 몫과 나머지를 Q[i]와 R[i]에 할당합니다.
				Q[i].Set(tempQ)
				R[i].Set(tempR)
			} else {
				// ps의 두 번째 요소부터는 (p * 이전 나머지)를 q로 나눕니다.
				tempMul.Mul(new(big.Int).SetUint64(p), R[i])
				tempQ.Div(tempMul, new(big.Int).SetUint64(q))
				tempR.Mod(tempMul, new(big.Int).SetUint64(q))

				// R[i]를 업데이트합니다.
				R[i].Set(tempR)

				// Q[i]를 업데이트합니다: Q[i] = Q[i] * p + tempQ
				// 참고: 원본 uint64 코드의 Q[i]*p + Q_ 로직을 그대로 옮겼습니다.
				Q[i].Mul(Q[i], new(big.Int).SetUint64(p))
				Q[i].Add(Q[i], tempQ)
			}

		}
	}

	return Q, R
}

// ShallowCopy creates a shallow copy of this basis extender in which the read-only data-structures are
// shared with the receiver.
func (be *BasisExtender) ShallowCopy() *BasisExtender {
	if be == nil {
		return nil
	}
	return &BasisExtender{
		ringQ:            be.ringQ,
		ringP:            be.ringP,
		dicConstantsQtoP: be.dicConstantsQtoP,
		dicConstantsPtoQ: be.dicConstantsPtoQ,

		buffQ: be.ringQ.NewPoly(),
		buffP: be.ringP.NewPoly(),
	}
}

func (be *BasisExtender) ModSwitchPtoQ(levelP, levelQ int, polP, polQ ring.Poly) {

	// ringQ := be.ringQ.AtLevel(levelQ)
	// ringP := be.ringP.AtLevel(levelP)
	// buffP := be.buffP

	// PHalf := bignum.NewInt(ringP.ModulusAtLevel[levelP])
	// PHalf.Rsh(PHalf, 1)

	// ringP.AddScalarBigint(polP, PHalf, buffP)
	ModUpExact(polP.Coeffs[:levelP+1], polQ.Coeffs[:levelQ+1], be.ringP, be.ringQ, be.dicConstantsPtoQ[Key{levelP, levelQ}])
	// QHalf := bignum.NewInt(ringQ.ModulusAtLevel[levelQ])
	// QHalf.Rsh(QHalf, 1)
	// ringQ.SubScalarBigint(polQ, PHalf, polQ)
}

func (be *BasisExtender) ModSwitchQtoP(levelQ, levelP int, polQ, polP ring.Poly) {

	// ringQ := be.ringQ.AtLevel(levelQ)
	// ringP := be.ringP.AtLevel(levelP)
	// buffQ := be.buffQ

	// QHalf := bignum.NewInt(ringQ.ModulusAtLevel[levelQ])
	// QHalf.Rsh(QHalf, 1)

	// ringQ.AddScalarBigint(polQ, QHalf, buffQ)
	ModUpExact(polQ.Coeffs[:levelQ+1], polP.Coeffs[:levelP+1], be.ringQ, be.ringP, be.dicConstantsQtoP[Key{levelQ, levelP}])
	// PHalf := bignum.NewInt(ringP.ModulusAtLevel[levelP])
	// PHalf.Rsh(PHalf, 1)
	// ringP.SubScalarBigint(polP, QHalf, polP)
}

// ModUpExact takes p1 mod Q and switches its basis to P, returning the result on p2.
// Caution: values are not centered and returned values are in [0, 2P-1].
func ModUpExact(p1, p2 [][]uint64, ringQ, ringP *ring.Ring, MUC ModUpConstants) {

	var rlo, rhi [8]uint64
	var y0, y1, y2, y3, y4, y5, y6, y7 [64]uint64

	levelQ := len(p1) - 1
	levelP := len(p2) - 1

	Q := ringQ.ModuliChain()
	mredQ := ringQ.MRedConstants()
	// fmt.Println(mredQ)

	P := ringP.ModuliChain()
	mredP := ringP.MRedConstants()
	bredP := ringP.BRedConstants()

	qoverqiinvqi := MUC.qoverqiinvqi
	alphaimodp := MUC.alphaimodp
	betaioverqi := MUC.betaioverqi
	// We loop over each coefficient and apply the basis extension
	for x := 0; x < len(p1[0]); x = x + 8 {
		reconstructRNS(0, levelQ+1, x, p1, &y0, &y1, &y2, &y3, &y4, &y5, &y6, &y7, Q, mredQ, qoverqiinvqi)
		for j := 0; j < levelP+1; j++ {
			/* #nosec G103 -- behavior and consequences well understood, possible buffer overflow if len(p2[j])%8 != 0*/
			multSum(levelQ, (*[8]uint64)(unsafe.Pointer(&p2[j][x])), &rlo, &rhi, &y0, &y1, &y2, &y3, &y4, &y5, &y6, &y7, P[j], mredP[j], bredP[j][0], alphaimodp[j], betaioverqi)
		}
	}
}

func reconstructRNS(start, end, x int, p [][]uint64, y0, y1, y2, y3, y4, y5, y6, y7 *[64]uint64, Q, QInv, QbMont []uint64) {
	var qi, qiInv, qoverqiinvqi uint64
	_ = p[end-1][x+7] // p의 각 행에 최소 8개가 있다는 가정 이미 있으니 힌트용
	_ = (*y0)[end-1]
	_ = (*y1)[end-1]
	_ = (*y2)[end-1]
	_ = (*y3)[end-1]
	_ = (*y4)[end-1]
	_ = (*y5)[end-1]
	_ = (*y6)[end-1]
	_ = (*y7)[end-1]

	for i, j := start, 0; i < end; i, j = i+1, j+1 {
		qoverqiinvqi = QbMont[i]
		qi = Q[i]
		qiInv = QInv[i]

		/* #nosec G103 -- behavior and consequences well understood, possible buffer overflow if len(p[i])%8 != 0 */
		pTmp := (*[8]uint64)(unsafe.Pointer(&p[i][x]))

		y0[j] = ring.MRed(pTmp[0], qoverqiinvqi, qi, qiInv)
		y4[j] = ring.MRed(pTmp[4], qoverqiinvqi, qi, qiInv)
		y1[j] = ring.MRed(pTmp[1], qoverqiinvqi, qi, qiInv)
		y5[j] = ring.MRed(pTmp[5], qoverqiinvqi, qi, qiInv)
		y2[j] = ring.MRed(pTmp[2], qoverqiinvqi, qi, qiInv)
		y6[j] = ring.MRed(pTmp[6], qoverqiinvqi, qi, qiInv)
		y3[j] = ring.MRed(pTmp[3], qoverqiinvqi, qi, qiInv)
		y7[j] = ring.MRed(pTmp[7], qoverqiinvqi, qi, qiInv)
	}
}

func montLazyAdd(hi, lo, si, q, qInv uint64) uint64 {
	// t = (lo * qInv) * q;  hhi = high64(t)
	hhi, _ := bits.Mul64(lo*qInv, q)
	x := hi - hhi + q + si
	if x >= q {
		x -= q
	}
	return x
}

// Caution, returns the values in [0, 2q-1]
func multSum(level int, res, rlo, rhi *[8]uint64, y0, y1, y2, y3, y4, y5, y6, y7 *[64]uint64, q, qInv uint64, bredP uint64, alphaimodp []uint64, betaioverqi []float64) {
	var qqip uint64

	_ = alphaimodp[level]
	_ = betaioverqi[level]
	_ = (*y0)[level]
	_ = (*y1)[level]
	_ = (*y2)[level]
	_ = (*y3)[level]
	_ = (*y4)[level]
	_ = (*y5)[level]
	_ = (*y6)[level]
	_ = (*y7)[level]

	var s [8]float64
	for i := 0; i < level+1; i++ {

		beta := betaioverqi[i]

		s[0] = math.FMA(float64(y0[i]), beta, s[0])
		s[4] = math.FMA(float64(y4[i]), beta, s[4])
		s[1] = math.FMA(float64(y1[i]), beta, s[1])
		s[5] = math.FMA(float64(y5[i]), beta, s[5])
		s[2] = math.FMA(float64(y2[i]), beta, s[2])
		s[6] = math.FMA(float64(y6[i]), beta, s[6])
		s[3] = math.FMA(float64(y3[i]), beta, s[3])
		s[7] = math.FMA(float64(y7[i]), beta, s[7])
	}
	var si [8]uint64
	var s0 uint64

	si[0] = uint64(s[0] + 0.5)
	s0, _ = bits.Mul64(si[0], bredP)
	si[0] = si[0] - s0*q

	si[4] = uint64(s[4] + 0.5)
	s0, _ = bits.Mul64(si[4], bredP)
	si[4] = si[4] - s0*q

	si[1] = uint64(s[1] + 0.5)
	s0, _ = bits.Mul64(si[1], bredP)
	si[1] = si[1] - s0*q

	si[5] = uint64(s[5] + 0.5)
	s0, _ = bits.Mul64(si[5], bredP)
	si[5] = si[5] - s0*q

	si[2] = uint64(s[2] + 0.5)
	s0, _ = bits.Mul64(si[2], bredP)
	si[2] = si[2] - s0*q

	si[6] = uint64(s[6] + 0.5)
	s0, _ = bits.Mul64(si[6], bredP)
	si[6] = si[6] - s0*q

	si[3] = uint64(s[3] + 0.5)
	s0, _ = bits.Mul64(si[3], bredP)
	si[3] = si[3] - s0*q

	si[7] = uint64(s[7] + 0.5)
	s0, _ = bits.Mul64(si[7], bredP)
	si[7] = si[7] - s0*q

	qqip = alphaimodp[0]

	rhi[0], rlo[0] = bits.Mul64(y0[0], qqip)
	rhi[4], rlo[4] = bits.Mul64(y4[0], qqip)
	rhi[1], rlo[1] = bits.Mul64(y1[0], qqip)
	rhi[5], rlo[5] = bits.Mul64(y5[0], qqip)
	rhi[2], rlo[2] = bits.Mul64(y2[0], qqip)
	rhi[6], rlo[6] = bits.Mul64(y6[0], qqip)
	rhi[3], rlo[3] = bits.Mul64(y3[0], qqip)
	rhi[7], rlo[7] = bits.Mul64(y7[0], qqip)

	// Accumulates the sum on uint128 and does a lazy montgomery reduction at the end
	var mhi, mlo, c uint64
	for i := 1; i < level+1; i++ {

		qqip = alphaimodp[i]

		mhi, mlo = bits.Mul64(y0[i], qqip)
		rlo[0], c = bits.Add64(rlo[0], mlo, 0)
		rhi[0] += mhi + c

		mhi, mlo = bits.Mul64(y4[i], qqip)
		rlo[4], c = bits.Add64(rlo[4], mlo, 0)
		rhi[4] += mhi + c

		mhi, mlo = bits.Mul64(y1[i], qqip)
		rlo[1], c = bits.Add64(rlo[1], mlo, 0)
		rhi[1] += mhi + c

		mhi, mlo = bits.Mul64(y5[i], qqip)
		rlo[5], c = bits.Add64(rlo[5], mlo, 0)
		rhi[5] += mhi + c

		mhi, mlo = bits.Mul64(y2[i], qqip)
		rlo[2], c = bits.Add64(rlo[2], mlo, 0)
		rhi[2] += mhi + c

		mhi, mlo = bits.Mul64(y6[i], qqip)
		rlo[6], c = bits.Add64(rlo[6], mlo, 0)
		rhi[6] += mhi + c

		mhi, mlo = bits.Mul64(y3[i], qqip)
		rlo[3], c = bits.Add64(rlo[3], mlo, 0)
		rhi[3] += mhi + c

		mhi, mlo = bits.Mul64(y7[i], qqip)
		rlo[7], c = bits.Add64(rlo[7], mlo, 0)
		rhi[7] += mhi + c

		// rlo[0], rhi[0] = muladd64(y0[i], qqip, rlo[0], rhi[0])

		// rlo[4], rhi[4] = muladd64(y4[i], qqip, rlo[4], rhi[4])

		// rlo[1], rhi[1] = muladd64(y1[i], qqip, rlo[1], rhi[1])

		// rlo[5], rhi[5] = muladd64(y5[i], qqip, rlo[5], rhi[5])

		// rlo[2], rhi[2] = muladd64(y2[i], qqip, rlo[2], rhi[2])

		// rlo[6], rhi[6] = muladd64(y6[i], qqip, rlo[6], rhi[6])

		// rlo[3], rhi[3] = muladd64(y3[i], qqip, rlo[3], rhi[3])

		// rlo[7], rhi[7] = muladd64(y7[i], qqip, rlo[7], rhi[7])
	}

	res[0] = montLazyAdd(rhi[0], rlo[0], si[0], q, qInv)
	res[4] = montLazyAdd(rhi[4], rlo[4], si[4], q, qInv)
	res[1] = montLazyAdd(rhi[1], rlo[1], si[1], q, qInv)
	res[5] = montLazyAdd(rhi[5], rlo[5], si[5], q, qInv)
	res[2] = montLazyAdd(rhi[2], rlo[2], si[2], q, qInv)
	res[6] = montLazyAdd(rhi[6], rlo[6], si[6], q, qInv)
	res[3] = montLazyAdd(rhi[3], rlo[3], si[3], q, qInv)
	res[7] = montLazyAdd(rhi[7], rlo[7], si[7], q, qInv)
}
