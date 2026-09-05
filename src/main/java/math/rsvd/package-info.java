/**
 * Randomized SVD for the caller: these classes return a decomposition rather
 * than a basis, and a rank that is determined rather than guessed.
 * <p>
 * The algorithms are those of Nathan Halko, Per-Gunnar Martinsson, and Joel A
 * Tropp. Finding structure with randomness: Probabilistic algorithms for
 * constructing approximate matrix decompositions. SIAM review, 53(2):217-288,
 * 2011. This package takes liberties with them where practice suggests it: the
 * test matrix of {@link math.rsvd.ApproximateBasis} is uniform rather than
 * Gaussian, its number of subspace iterations is fixed, and the rank criterion
 * behind {@code ApproximateBasis.decompose} comes from a different paper
 * altogether (Matan Gavish and David L. Donoho. The optimal hard threshold for
 * singular values is 4 / sqrt(3). IEEE Transactions on Information Theory,
 * 60(8):5040-5053, 2014).
 * <p>
 * For those algorithms transcribed as the paper states them, see
 * {@link math.rsvd.reference}.
 */
package math.rsvd;
