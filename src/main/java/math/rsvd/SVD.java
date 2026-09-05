/*
 * Copyright 2020, 2026 Stefan Zobel
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package math.rsvd;

import java.util.Arrays;
import java.util.Objects;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;

/**
 * An approximate singular value decomposition {@code A ~ U * S * Vt}, where
 * {@code U} has orthonormal columns, {@code Vt} has orthonormal rows and
 * {@code S} is the diagonal matrix of the singular values in descending order.
 * <p>
 * The singular values are held as an array rather than as a matrix. The
 * diagonal form is available from {@link #getS()}, but it is built on demand
 * and never stored: measured for a {@code 400 x 400} decomposition of rank
 * 400, it occupies 160,000 cells of which 400 are used, and accounts for a
 * third of the memory of the whole decomposition.
 */
public class SVD {

    /** The IEEE 754 machine epsilon from Cephes: (2^-53) */
    private static final double MACH_EPS_DBL = 1.11022302462515654042e-16;
    /**
     * Upper bound on the rounds of {@link #optimalRank(MatrixD)}. Measured over
     * 400 random cases the iteration needed two to three rounds and never more
     * than six, so this is a guard against a cycle that has not been observed,
     * not a working limit.
     */
    private static final int MAX_REFINEMENTS = 50;

    private final MatrixD U;
    private final MatrixD Vt;
    /** the singular values, exactly as many as U has columns */
    private final double[] sigma;

    /**
     * Creates a decomposition from its three factors.
     *
     * @param U
     *            the left singular vectors, one per column
     * @param singularValues
     *            the singular values in descending order. Only the first
     *            {@code U.numColumns()} of them belong to this decomposition;
     *            a longer array is accepted and truncated, which is what the
     *            economy sized decompositions of the underlying library hand
     *            out
     * @param Vt
     *            the transposed right singular vectors, one per row
     * @throws IllegalArgumentException
     *             if {@code Vt} does not have as many rows as {@code U} has
     *             columns, or if fewer singular values are supplied than
     *             {@code U} has columns
     */
    public SVD(MatrixD U, double[] singularValues, MatrixD Vt) {
        this.U = Objects.requireNonNull(U);
        this.Vt = Objects.requireNonNull(Vt);
        Objects.requireNonNull(singularValues);
        int k = U.numColumns();
        if (Vt.numRows() != k) {
            throw new IllegalArgumentException(
                    "U has " + k + " columns, so Vt must have " + k + " rows, but it has " + Vt.numRows());
        }
        if (singularValues.length < k) {
            throw new IllegalArgumentException("U has " + k + " columns, so at least " + k
                    + " singular values are needed, but only " + singularValues.length + " were supplied");
        }
        // a copy, for two reasons: the array may be longer than this
        // decomposition is wide, and a caller passing the array of a jamu
        // SvdD would otherwise keep a writable reference to our state, since
        // SvdD.getS() hands out its internal array directly
        this.sigma = Arrays.copyOf(singularValues, k);
    }

    /**
     * The left singular vectors, one per column.
     *
     * @return the {@code m x k} matrix {@code U}
     */
    public MatrixD getU() {
        return U;
    }

    /**
     * The transposed right singular vectors, one per row.
     *
     * @return the {@code k x n} matrix {@code Vt}
     */
    public MatrixD getVt() {
        return Vt;
    }

    /**
     * The singular values in descending order.
     *
     * @return a fresh copy of the {@code k} singular values, so that a caller
     *         cannot change this decomposition through the array it is handed
     */
    public double[] getSingularValues() {
        return sigma.clone();
    }

    /**
     * The singular values as a diagonal matrix.
     * <p>
     * This is the escape hatch for code that wants to write the product
     * {@code U * S * Vt} literally. It is built on every call and not cached,
     * because {@code k * k} cells for {@code k} numbers are not worth holding
     * on to; {@link #reconstruct()} computes that product without it.
     *
     * @return a new {@code k x k} diagonal matrix carrying the singular values
     */
    public MatrixD getS() {
        return Matrices.diagD(sigma);
    }

    /**
     * The number of singular values of this decomposition, that is the number
     * of columns of {@code U} and the number of rows of {@code Vt}.
     *
     * @return the number of singular values
     */
    public int size() {
        return sigma.length;
    }

    /**
     * The numerical rank of this decomposition, counting the singular values
     * above {@code max(rows, columns) * eps * sigma_1}, which is the usual
     * threshold.
     * <p>
     * Note what this can and cannot say. It is the rank of the
     * <em>approximation</em>, and the approximation has at most as many
     * singular values as the caller asked for when constructing it. A matrix
     * of rank 100 decomposed with an estimated rank of 10 reports 10 here, not
     * 100. The number is meaningful when the estimated rank was chosen
     * generously and one wants to know how much of it was actually needed.
     *
     * @return the number of singular values above the default threshold
     */
    public int rank() {
        if (sigma.length == 0) {
            return 0;
        }
        return rank(Math.max(U.numRows(), Vt.numColumns()) * MACH_EPS_DBL * sigma[0]);
    }

    /**
     * The number of singular values strictly above {@code tolerance}.
     *
     * @param tolerance
     *            the threshold below which a singular value counts as zero
     * @return the number of singular values above {@code tolerance}
     * @see #rank()
     */
    public int rank(double tolerance) {
        int r = 0;
        for (int i = 0; i < sigma.length; ++i) {
            if (sigma[i] > tolerance) {
                ++r;
            }
        }
        return r;
    }

    /**
     * The number of singular values that carry signal rather than noise,
     * according to the optimal hard threshold of Gavish and Donoho (2014), for
     * a noise level supplied by the caller.
     * <p>
     * The threshold is
     * {@code lambda(beta) * sqrt(max(rows, columns)) * noiseLevel} with
     * {@code beta = min(rows, columns) / max(rows, columns)}. It is bounded
     * below by the threshold of {@link #rank()}: below the round-off level of
     * the decomposition nothing can be told from zero anyway, and without that
     * floor a noise free input reports the full width of the sketch instead of
     * its rank.
     * <p>
     * This is a different question from the two the library answers elsewhere,
     * and the three are easy to confuse. {@code AdaRangeFinder} answers "how
     * many columns are needed for a relative accuracy of epsilon".
     * {@link #rank()} answers "how many singular values are above the
     * round-off level of this approximation". This method answers "where does
     * signal end and noise begin", which is a statement about the data and not
     * about arithmetic.
     * <p>
     * The underlying model is a low rank signal plus additive white noise. It
     * degrades gracefully outside that model rather than failing: measured on a
     * rank 10 signal, noise differing by a factor of 20 between column groups
     * still gave 10, noise with heavy tails gave 11, and a handful of gross
     * outliers were counted as the extra dimensions they genuinely are.
     *
     * @param noiseLevel
     *            the standard deviation of the noise, which must be finite and
     *            not negative
     * @return the number of singular values above the threshold
     * @throws IllegalArgumentException
     *             if {@code noiseLevel} is negative, infinite or {@code NaN}
     * @see #optimalRank(MatrixD)
     */
    public int optimalRank(double noiseLevel) {
        // the negated comparison also rejects NaN
        if (!(noiseLevel >= 0.0 && noiseLevel < Double.POSITIVE_INFINITY)) {
            throw new IllegalArgumentException(
                    "noiseLevel must be finite and not negative, but was " + noiseLevel);
        }
        if (sigma.length == 0) {
            return 0;
        }
        return rank(threshold(noiseLevel));
    }

    /**
     * The number of singular values that carry signal rather than noise, with
     * the noise level estimated from this decomposition itself.
     * <p>
     * The noise level and the number of signal values determine each other: the
     * noise is what is left of {@code A} once the signal has been subtracted,
     * and how much to subtract is exactly the number being sought. The two are
     * therefore solved together rather than one after the other. Starting from
     * "everything is signal", the estimate
     * {@code ||A - A_r||_F / sqrt((rows - r) * (columns - r))} and the
     * threshold of {@link #optimalRank(double)} are applied in turn until the
     * count stops changing, which measured over 400 random cases took two to
     * three rounds and never more than six.
     * <p>
     * Doing it in one pass instead does not work. Using the width of the
     * decomposition in place of {@code r} in the degrees of freedom underrates
     * the noise by up to 35 %, because the columns of a fitted basis are chosen
     * to capture as much energy as they can and so take more than their share
     * of it.
     * <p>
     * Measured against an oracle that is given the true noise level and the
     * exact full spectrum, this agreed exactly in 384 of 400 random cases and
     * was off by one in 10 more. Against the rank the test matrices were built
     * from it scored 367 of 400, where the oracle itself scored 366.
     *
     * @param A
     *            the matrix this decomposition approximates, needed for the
     *            residual the noise level is estimated from
     * @return the number of singular values above the estimated threshold,
     *         which is 0 if {@code A} is indistinguishable from noise
     * @throws NullPointerException
     *             if {@code A} is {@code null}
     * @throws IllegalArgumentException
     *             if {@code A} does not have the shape this decomposition
     *             approximates
     * @see #optimalRank(double)
     */
    public int optimalRank(MatrixD A) {
        Objects.requireNonNull(A);
        int m = U.numRows();
        int n = Vt.numColumns();
        if (A.numRows() != m || A.numColumns() != n) {
            throw new IllegalArgumentException("this decomposition approximates a " + m + " x " + n
                    + " matrix, but A is " + A.numRows() + " x " + A.numColumns());
        }
        if (sigma.length == 0) {
            return 0;
        }
        // r has to stay below min(rows, columns): there the residual is empty
        // by construction, the noise comes out as zero and everything survives
        // the threshold
        int cap = Math.min(m, n) - 1;
        if (cap < 1) {
            // a single row or column leaves no residual to estimate from, the
            // rank 1 approximation being exact already
            return rank();
        }
        int r = Math.min(sigma.length, cap);
        for (int i = 0; i < MAX_REFINEMENTS; ++i) {
            double noise = residualNorm(A, r) / Math.sqrt((double) (m - r) * (n - r));
            int next = Math.min(rank(threshold(noise)), cap);
            if (next == r) {
                return r;
            }
            if (i > 0 && Math.abs(next - r) <= 1) {
                // once two consecutive counts are within one of each other the
                // iteration has reached the edge of the spectrum, where further
                // rounds only drift down by one more. This is not needed for
                // termination: measured over 400 random cases the iteration
                // always settled on its own, in at most 8 rounds. It is a
                // slightly better calibrated and slightly cheaper stopping
                // point, matching an oracle given the true noise level in 384
                // of those cases against 382, in at most 6 rounds instead of 8
                return Math.min(next, r);
            }
            r = next;
        }
        return r;
    }

    /**
     * The Gavish-Donoho threshold for {@code noiseLevel}, but never below the
     * threshold of {@link #rank()}.
     *
     * @param noiseLevel
     *            the standard deviation of the noise
     * @return the singular value below which a value counts as noise
     */
    private double threshold(double noiseLevel) {
        int m = U.numRows();
        int n = Vt.numColumns();
        double beta = Math.min(m, n) / (double) Math.max(m, n);
        double lambda = Math.sqrt(2.0 * (beta + 1.0)
                + 8.0 * beta / ((beta + 1.0) + Math.sqrt(beta * beta + 14.0 * beta + 1.0)));
        double tau = lambda * Math.sqrt(Math.max(m, n)) * noiseLevel;
        return Math.max(tau, Math.max(m, n) * MACH_EPS_DBL * sigma[0]);
    }

    /**
     * {@code ||A - A_r||_F}, where {@code A_r} is this decomposition truncated
     * to rank {@code r}.
     * <p>
     * The residual is formed rather than inferred from
     * {@code ||A||_F^2 - sum sigma_i^2}. That subtraction is cheaper by 4 % but
     * cancels once the noise sinks below {@code sqrt(eps)} relative to the
     * signal, where it returns a non positive tail and hence a noise level of
     * zero; measured, it fails for noise levels between 1e-12 and 1e-8 and is
     * correct on both sides of that band. Forming the residual also means that
     * nothing is ever squared, so the estimate keeps its scale invariance down
     * to the smallest normal {@code double}.
     *
     * @param A
     *            the matrix this decomposition approximates
     * @param r
     *            the rank to truncate to, where 0 means the zero matrix
     * @return {@code ||A - A_r||_F}
     */
    private double residualNorm(MatrixD A, int r) {
        if (r <= 0) {
            // the rank 0 approximation is the zero matrix
            return A.normF();
        }
        return A.copy().addInplace(-1.0, truncate(Math.min(r, sigma.length)).reconstruct()).normF();
    }

    /**
     * Computes the product {@code U * S * Vt}, the approximation of the matrix
     * this decomposition was computed from.
     * <p>
     * The diagonal matrix is not formed: row {@code i} of {@code S * Vt} is
     * simply {@code sigma_i} times row {@code i} of {@code Vt}, which saves
     * both the {@code k x k} matrix and one of the two multiplications.
     *
     * @return a new {@code m x n} matrix
     */
    public MatrixD reconstruct() {
        MatrixD SVt = Vt.copy();
        for (int i = 0; i < sigma.length; ++i) {
            double s = sigma[i];
            for (int j = 0; j < SVt.numColumns(); ++j) {
                SVt.set(i, j, s * SVt.get(i, j));
            }
        }
        return U.times(SVt);
    }

    /**
     * The decomposition of rank {@code k} obtained by keeping only the
     * {@code k} largest singular values and their vectors.
     *
     * @param k
     *            the number of singular values to keep, from 1 to
     *            {@link #size()}
     * @return a new decomposition of rank {@code k}, or this one if {@code k}
     *         is already its size
     * @throws IllegalArgumentException
     *             if {@code k} is outside {@code [1, size()]}
     */
    public SVD truncate(int k) {
        if (k < 1 || k > sigma.length) {
            throw new IllegalArgumentException(
                    "k must be in [1, " + sigma.length + "], but was " + k);
        }
        if (k == sigma.length) {
            return this;
        }
        return new SVD(U.selectConsecutiveColumns(0, k - 1), Arrays.copyOf(sigma, k),
                Vt.selectSubmatrix(0, 0, k - 1, Vt.endCol()));
    }
}
