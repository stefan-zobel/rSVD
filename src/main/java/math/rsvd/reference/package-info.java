/**
 * The Stage A algorithms of Halko, Martinsson and Tropp, transcribed as the
 * paper states them: one class per numbered algorithm, a Gaussian test matrix
 * as the bounds of the paper require, the number of iterations left to the
 * caller, and no step beyond the orthonormal basis {@code Q}.
 * <p>
 * Fidelity is the purpose of this package, which is why
 * {@link math.rsvd.reference.RanPowerIteration} (Algorithm 4.3) is here at all
 * although the paper recommends
 * {@link math.rsvd.reference.RanSubspaceIteration} (Algorithm 4.4) in its
 * place. Remark 4.3 of the paper explains that in floating point arithmetic
 * round-off extinguishes every singular value below
 * {@code eps^(1/(2q+1)) * ||A||}, and a transcription that dropped 4.3 would
 * drop the statement the paper makes about it.
 * <p>
 * Nathan Halko, Per-Gunnar Martinsson, and Joel A Tropp. Finding structure with
 * randomness: Probabilistic algorithms for constructing approximate matrix
 * decompositions. SIAM review, 53(2):217-288, 2011.
 * <p>
 * For the decompositions a caller is more likely to want, see
 * {@link math.rsvd}.
 */
package math.rsvd.reference;
