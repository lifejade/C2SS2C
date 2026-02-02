package matmult

import (
	"math/bits"

	"github.com/tuneinsight/lattigo/v5/utils/bignum"
)

// MForm switches a to the Montgomery domain by computing
// a*2^32 mod q.
func MForm(a, q uint32, u []uint32) (r uint32) {
	mhi, _ := bits.Mul32(a, u[1])
	r = -(a*u[0] + mhi) * q
	if r >= q {
		r -= q
	}
	return
}

// MFormLazy switches a to the Montgomery domain by computing
// a*2^32 mod q in constant time.
// The result is between 0 and 2*q-1.
func MFormLazy(a, q uint32, u []uint32) (r uint32) {
	mhi, _ := bits.Mul32(a, u[1])
	r = -(a*u[0] + mhi) * q
	return
}

// IMForm switches a from the Montgomery domain back to the
// standard domain by computing a*(1/2^32) mod q.
func IMForm(a, q, qInv uint32) (r uint32) {
	r, _ = bits.Mul32(a*qInv, q)
	r = q - r
	if r >= q {
		r -= q
	}
	return
}

// IMFormLazy switches a from the Montgomery domain back to the
// standard domain by computing a*(1/2^32) mod q in constant time.
// The result is between 0 and 2*q-1.
func IMFormLazy(a, q, qInv uint32) (r uint32) {
	r, _ = bits.Mul32(a*qInv, q)
	r = q - r
	return
}

// MRedConstant computes the constant qInv = (q^-1) mod 2^32 required for MRed.
func MRedConstant(q uint32) (qInv uint32) {
	qInv = 1
	for i := 0; i < 63; i++ {
		qInv *= q
		q *= q
	}
	return
}

// MRed computes x * y * (1/2^32) mod q.
func MRed(x, y, q, qInv uint32) (r uint32) {
	mhi, mlo := bits.Mul32(x, y)
	hhi, _ := bits.Mul32(mlo*qInv, q)
	r = mhi - hhi + q
	if r >= q {
		r -= q
	}
	return
}

// MRedLazy computes x * y * (1/2^32) mod q in constant time.
// The result is between 0 and 2*q-1.
func MRedLazy(x, y, q, qInv uint32) (r uint32) {
	ahi, alo := bits.Mul32(x, y)
	H, _ := bits.Mul32(alo*qInv, q)
	r = ahi - H + q
	return
}

// BRedConstant computes the constant for the BRed algorithm.
// Returns ((2^128)/q)/(2^32) and (2^128)/q mod 2^32.
func BRedConstant(q uint32) (constant []uint32) {
	bigR := bignum.NewInt("0x100000000000000000000000000000000")
	bigR.Quo(bigR, bignum.NewInt(q))

	mlo := uint32(bigR.Uint64())
	mhi := uint32(bigR.Rsh(bigR, 32).Uint64())

	return []uint32{mhi, mlo}
}

// BRedAdd computes a mod q.
func BRedAdd(a, q uint32, u []uint32) (r uint32) {
	mhi, _ := bits.Mul32(a, u[0])
	r = a - mhi*q
	if r >= q {
		r -= q
	}
	return
}

// BRedAddLazy computes a mod q in constant time.
// The result is between 0 and 2*q-1.
func BRedAddLazy(x, q uint32, u []uint32) uint32 {
	s0, _ := bits.Mul32(x, u[0])
	return x - s0*q
}

// BRed computes x*y mod q.
func BRed(x, y, q uint32, u []uint32) (r uint32) {

	var mhi, mlo, lhi, hhi, hlo, s0, carry uint32

	mhi, mlo = bits.Mul32(x, y)

	// computes r = mhi * uhi + (mlo * uhi + mhi * ulo)<<32 + (mlo * ulo)) >> 128

	r = mhi * u[0] // r = mhi * uhi

	hhi, hlo = bits.Mul32(mlo, u[0]) // mlo * uhi

	r += hhi

	lhi, _ = bits.Mul32(mlo, u[1]) // mlo * ulo

	s0, carry = bits.Add32(hlo, lhi, 0)

	r += carry

	hhi, hlo = bits.Mul32(mhi, u[1]) // mhi * ulo

	r += hhi

	_, carry = bits.Add32(hlo, s0, 0)

	r += carry

	r = mlo - r*q

	if r >= q {
		r -= q
	}

	return
}

// BRedLazy computes x*y mod q in constant time.
// The result is between 0 and 2*q-1.
func BRedLazy(x, y, q uint32, u []uint32) (r uint32) {

	var mhi, mlo, lhi, hhi, hlo, s0, carry uint32

	mhi, mlo = bits.Mul32(x, y)

	// computes r = mhi * uhi + (mlo * uhi + mhi * ulo)<<32 + (mlo * ulo)) >> 128

	r = mhi * u[0] // r = mhi * uhi

	hhi, hlo = bits.Mul32(mlo, u[0]) // mlo * uhi

	r += hhi

	lhi, _ = bits.Mul32(mlo, u[1]) // mlo * ulo

	s0, carry = bits.Add32(hlo, lhi, 0)

	r += carry

	hhi, hlo = bits.Mul32(mhi, u[1]) // mhi * ulo

	r += hhi

	_, carry = bits.Add32(hlo, s0, 0)

	r += carry

	r = mlo - r*q

	return
}

// CRed reduce returns a mod q where a is between 0 and 2*q-1.
func CRed(a, q uint32) uint32 {
	if a >= q {
		return a - q
	}
	return a
}
