/**
 * Randomized SVD for the caller: these classes return a decomposition rather
 * than a basis, and a rank that is determined rather than guessed.
 * <p>
 * {@link math.rsvd.RandomizedSVD} is the way in. It asks what is known about the
 * matrix - an accuracy target, a rank, or neither - and the choice of algorithm
 * follows from the answer, with each setting named rather than positional. The
 * constructors of {@link math.rsvd.AdaRangeFinder} and
 * {@link math.rsvd.ApproximateBasis} remain public and are still the way to
 * reach what the facade deliberately leaves out, as is
 * {@link math.rsvd.Nystroem}, which is composed with a basis rather than
 * configured.
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
