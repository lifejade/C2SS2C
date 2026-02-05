package matmult

import (
	"fmt"
	"math"
	"math/big"
	"math/bits"

	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils"
)

// Ring is a structure that keeps all the variables required to operate on a polynomial represented in this ring.
type Ring struct {
	SubRings []*SubRing

	// Product of the Moduli for each level
	ModulusAtLevel []*big.Int

	// Rescaling parameters (RNS division)
	RescaleConstants [][]uint64

	level int
}

// N returns the ring degree.
func (r Ring) N() int {
	return r.SubRings[0].N
}

// LogN returns log2(ring degree).
func (r Ring) LogN() int {
	return bits.Len64(uint64(r.N() - 1))
}

// LogModuli returns the size of the extended modulus P in bits
func (r Ring) LogModuli() (logmod float64) {
	for _, qi := range r.ModuliChain() {
		logmod += math.Log2(float64(qi))
	}
	return
}

// ModuliChainLength returns the number of primes in the RNS basis of the ring.
func (r Ring) ModuliChainLength() int {
	return len(r.SubRings)
}

// Level returns the level of the current ring.
func (r Ring) Level() int {
	return r.level
}

// AtLevel returns an instance of the target ring that operates at the target level.
// This instance is thread safe and can be use concurrently with the base ring.
func (r Ring) AtLevel(level int) *Ring {

	// Sanity check
	if level < 0 {
		panic("level cannot be negative")
	}

	// Sanity check
	if level > r.MaxLevel() {
		panic("level cannot be larger than max level")
	}

	return &Ring{
		SubRings:         r.SubRings,
		ModulusAtLevel:   r.ModulusAtLevel,
		RescaleConstants: r.RescaleConstants,
		level:            level,
	}
}

// MaxLevel returns the maximum level allowed by the ring (#NbModuli -1).
func (r Ring) MaxLevel() int {
	return r.ModuliChainLength() - 1
}

// ModuliChain returns the list of primes in the modulus chain.
func (r Ring) ModuliChain() (moduli []uint32) {
	moduli = make([]uint32, len(r.SubRings))
	for i := range r.SubRings {
		moduli[i] = r.SubRings[i].Modulus
	}

	return
}

// Modulus returns the modulus of the target ring at the currently
// set level in *big.Int.
func (r Ring) Modulus() *big.Int {
	return r.ModulusAtLevel[r.level]
}

// MRedConstants returns the concatenation of the Montgomery constants
// of the target ring.
func (r Ring) MRedConstants() (MRC []uint64) {
	MRC = make([]uint64, len(r.SubRings))
	for i := range r.SubRings {
		MRC[i] = r.SubRings[i].MRedConstant
	}

	return
}

// BRedConstants returns the concatenation of the Barrett constants
// of the target ring.
func (r Ring) BRedConstants() (BRC [][]uint64) {
	BRC = make([][]uint64, len(r.SubRings))
	for i := range r.SubRings {
		BRC[i] = r.SubRings[i].BRedConstant
	}

	return
}

func NewRing(N int, Moduli []uint32) (r *Ring, err error) {
	r = new(Ring)
	if len(Moduli) == 0 {
		return nil, fmt.Errorf("invalid ModuliChain (must be a non-empty []uint64)")
	}

	if !utils.AllDistinct(Moduli) {
		return nil, fmt.Errorf("invalid ModuliChain (moduli are not distinct)")
	}

	// Computes bigQ for all levels
	r.ModulusAtLevel = make([]*big.Int, len(Moduli))
	r.ModulusAtLevel[0] = new(big.Int).SetUint64(uint64(Moduli[0]))
	for i := 1; i < len(Moduli); i++ {
		r.ModulusAtLevel[i] = new(big.Int).Mul(r.ModulusAtLevel[i-1], new(big.Int).SetUint64(uint64(Moduli[i])))
	}
	r.SubRings = make([]*SubRing, len(Moduli))

	for i := range r.SubRings {
		if r.SubRings[i], err = NewSubRing(N, Moduli[i]); err != nil {
			return nil, err
		}
	}

	r.RescaleConstants = rewRescaleConstants(r.SubRings)

	r.level = len(Moduli) - 1

	return r, nil
}

// func (r *Ring) NewPoly() Poly {
// 	return newPoly(r.N(), r.Level())
// }

type SubRing struct {

	// Polynomial nb.Coefficients
	N int

	// Modulus
	Modulus uint32

	// Fast reduction constants
	BRedConstant []uint64 // Barrett Reduction
	MRedConstant uint64   // Montgomery Reduction
}

func NewSubRing(N int, Modulus uint32) (s *SubRing, err error) {

	s = &SubRing{}

	s.N = N

	s.Modulus = Modulus

	// Computes the fast modular reduction constants for the Ring
	s.BRedConstant = ring.BRedConstant(uint64(Modulus))

	// If qi is not a power of 2, we can compute the MRed (otherwise, it
	// would return an error as there is no valid Montgomery form mod a power of 2)
	if (Modulus&(Modulus-1)) != 0 && Modulus != 0 {
		s.MRedConstant = ring.MRedConstant(uint64(Modulus))
	}

	return
}

func rewRescaleConstants(subRings []*SubRing) (rescaleConstants [][]uint64) {

	rescaleConstants = make([][]uint64, len(subRings)-1)

	for j := len(subRings) - 1; j > 0; j-- {

		qj := uint64(subRings[j].Modulus)

		rescaleConstants[j-1] = make([]uint64, j)

		for i := 0; i < j; i++ {
			qi := uint64(subRings[i].Modulus)
			rescaleConstants[j-1][i] = ring.MForm(qi-ring.ModExp(qj, qi-2, qi), qi, subRings[i].BRedConstant)
		}
	}

	return
}
func (r *Ring) NewPoly() Poly {
	return newPoly(r.N(), r.Level())
}
