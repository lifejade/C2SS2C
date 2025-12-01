package sharedswtich

import (
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

func GenShAEvkNew(params hefloat.Parameters, kgen rlwe.KeyGenerator, skIn *rlwe.SecretKey, n int) (evk *rlwe.EvaluationKey) {
	levelQ, levelP := params.MaxLevelQ(), params.MaxLevelP()
	evk = &rlwe.EvaluationKey{GadgetCiphertext: *rlwe.NewGadgetCiphertext(params, 1, levelQ, levelP, 0)}

	skOutputs := make([]*rlwe.SecretKey, n)
	for i := range skOutputs {
		skOutputs[i] = kgen.GenSecretKeyNew()
	}
	GenShAEvaluationKey(params, skIn, skOutputs, evk)
	return
}

func GenShAEvaluationKey(params hefloat.Parameters, skIn *rlwe.SecretKey, skOuts []*rlwe.SecretKey, evk *rlwe.EvaluationKey) {

	ringQ := params.RingQ()
	ringP := params.RingP()

	buffQP := params.RingQP().NewPoly()
	buffQ := [2]ring.Poly{ringQ.NewPoly(), ringQ.NewPoly()}

	// Maps the smaller key to the largest with Y = X^{N/n}.
	for i := range skOuts {
		ring.MapSmallDimensionToLargerDimensionNTT(skOuts[i].Value.Q, buffQP.Q)
	}
	// Extends the modulus P of skOutput to the one of skInput
	if levelP := evk.LevelP(); levelP != -1 {
		rlwe.ExtendBasisSmallNormAndCenterNTTMontgomery(ringQ, ringP.AtLevel(levelP), buffQP.Q, buffQ[0], buffQP.P)
	}

	// Maps the smaller key to the largest dimension with Y = X^{N/n}.
	ring.MapSmallDimensionToLargerDimensionNTT(skIn.Value.Q, buffQ[0])
	for i := range skOuts {
		rlwe.ExtendBasisSmallNormAndCenterNTTMontgomery(ringQ, ringQ.AtLevel(skOuts[i].Value.Q.Level()), buffQ[0], buffQ[1], buffQ[0])
	}
	// kgen.genEvaluationKey(kgen.buffQ[0], kgen.buffQP, evk)
}

func genEvaluationKey() {

}
