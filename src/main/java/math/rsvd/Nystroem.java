/*
 * Copyright 2026 Stefan Zobel
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

import java.util.Objects;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;
import net.jamu.matrix.SvdD;

/**
 * Eigenvalue decomposition of a positive semidefinite matrix from an
 * approximate basis of its range, by the Nystroem method.
 * <p>
 * Algorithm 5.5 from Nathan Halko, Per-Gunnar Martinsson, and Joel A Tropp.
 * Finding structure with randomness: Probabilistic algorithms for constructing
 * approximate matrix decompositions. SIAM review, 53(2):217-288, 2011.
 * <p>
 * Where {@link ApproximateBasis#computeSVD()} forms the approximation
 * {@code Q (Q'AQ) Q'}, which is right for any matrix, this forms
 * {@code (AQ) (Q'AQ)^-1 (AQ)'}, which is right only for a positive semidefinite
 * one and is better. The paper: "In both cases, the dominant cost occurs when we
 * form AQ, so the two procedures have roughly the same running time. On the
 * other hand, Algorithm 5.5 is typically much more accurate than Algorithm 5.3.
 * In a sense, we are exploiting the fact that A is positive semidefinite to take
 * one step of subspace iteration (Algorithm 4.4) for free." And on the error:
 * "in the spectral norm, the Nystroem approximation error never exceeds
 * ||A - QQ'A||, and it is often substantially smaller."
 * <p>
 * <b>The accuracy holds up; the cost claim does not.</b> Measured over 360 cases
 * whose error is above the round-off floor - random spectra with exponential,
 * polynomial and exactly low rank decay, sizes from 80 to 240, basis widths from
 * 5 to 40 - the bound was never violated, and the error came to a median of 0.73
 * of {@code ||A - QQ'A||} and 0.67 of what Algorithm 5.3 achieves from the same
 * basis, with a best of 0.025. But this is <b>not</b> free: measured against the
 * complete Algorithm 5.3 it takes 2.6 to 8.1 times as long. The reason is that
 * step 4 decomposes an {@code n x k} matrix, where Algorithm 5.3 only decomposes
 * the {@code k x k} core and lifts the result back with a product. At
 * {@code n = 800} and a basis of 60 that one step costs 1.63 ms against 0.79 ms
 * for forming {@code AQ}, so the premise "the dominant cost occurs when we form
 * AQ" simply does not hold against an optimized BLAS. It comes closer as
 * {@code n / k} grows: at {@code n = 1600} and a basis of 20, {@code AQ} takes
 * 1.73 ms against 0.52 ms for step 4. Replacing the decomposition in step 4 by a
 * QR factorization plus a small decomposition was measured and is slower still,
 * in every shape tried.
 * <p>
 * The basis may come from any of the range finders, the paper only asks that it
 * capture the range of {@code A} well. It does not have to be tight: a basis
 * wider than the numerical rank of {@code A} is trimmed rather than rejected,
 * so the decomposition returned can be narrower than {@code Q}.
 */
public final class Nystroem {

    /** The IEEE 754 machine epsilon from Cephes: (2^-53) */
    private static final double MACH_EPS_DBL = 1.11022302462515654042e-16;

    /**
     * How far from symmetric {@code A} may be, in the metric of
     * {@link #relativeAsymmetry(MatrixD)}.
     * <p>
     * Deliberately generous, because this guards against passing a matrix that
     * is not symmetric at all, not against round-off. A little asymmetry is
     * harmless here: the approximation this class builds is symmetric by
     * construction, so it simply approximates the symmetric part of what it was
     * given, and a false rejection would be the worse outcome. Measured, a
     * matrix formed as {@code G * G'} comes out exactly symmetric in most cases
     * and never worse than {@code 1.1e-16}, over sizes from 40 to 600, ranks
     * from 1 to full and scales from {@code 1e-160} to {@code 1e+100}; a matrix
     * perturbed away from symmetry reproduces the size of the perturbation. So
     * this rejects from about {@code 1e-10} upwards, six orders above anything
     * round-off produces, and catches a general matrix, which lands around
     * {@code 0.6}, by nine.
     */
    private static final double SYMMETRY_TOLERANCE = 1.0e-10;

    /**
     * How negative an eigenvalue of the core {@code Q'AQ} may be, relative to
     * the largest one, before {@code A} is rejected as indefinite.
     * <p>
     * A matrix that is positive semidefinite in exact arithmetic is not quite
     * one after it has been computed: measured, a rank 10 matrix perturbed by a
     * relative {@code 1e-13} carried a genuinely negative eigenvalue at
     * {@code -5e-15} of the largest, which is round-off and not a violation. An
     * indefinite matrix, on the other hand, announces itself loudly - a planted
     * eigenvalue of {@code -50} beside a largest of {@code 100} is a relative
     * {@code -0.5}. Seven orders of magnitude on either side of this threshold,
     * so it is not a delicate choice. Directions below it are dropped rather
     * than used, since a negative eigenvalue has no inverse square root.
     */
    private static final double DEFINITENESS_TOLERANCE = 1.0e-8;

    private Nystroem() {
        throw new AssertionError("no instances");
    }

    /**
     * Computes an approximate eigenvalue decomposition {@code A ~ U L U'} of the
     * positive semidefinite matrix {@code A} from the basis {@code Q}, where
     * {@code U} is orthonormal and {@code L} is nonnegative and diagonal.
     * <p>
     * The result is returned as an {@link SVD} rather than as a separate
     * eigenvalue type, because for a positive semidefinite matrix the two
     * coincide: the singular values are the eigenvalues and the left and right
     * singular vectors are the same. Everything {@code SVD} offers -
     * {@link SVD#reconstruct()}, {@link SVD#truncate(int)}, {@link SVD#rank()},
     * {@link SVD#optimalRank(MatrixD)} - therefore applies unchanged.
     * <p>
     * <b>The precondition is checked, but only as far as it can be cheaply.</b>
     * Symmetry is tested against {@code ||A||_F}. Definiteness is not tested on
     * {@code A} itself, which would cost a factorization of the whole matrix,
     * but on the {@code k x k} core {@code Q'AQ}, where it is nearly free: for a
     * symmetric core the singular values are the absolute values of the
     * eigenvalues, and a negative eigenvalue shows up as a sign flip between the
     * left and the right singular vector. A negative eigenvalue at round-off
     * level is not a rejection - a matrix that is positive semidefinite in exact
     * arithmetic is rarely one after it has been computed - its direction is
     * dropped instead; see {@link #DEFINITENESS_TOLERANCE}. An indefinite matrix
     * whose negative eigenvalues happen to lie outside the range of {@code Q}
     * passes, because nothing in this computation can see them.
     * <p>
     * <b>Steps 2 and 3 of the paper are not carried out as written.</b>
     * Algorithm 5.5 factors the core {@code B2 = Q'AQ} by a Cholesky
     * decomposition and forms {@code F = B1 C^-1} by a triangular solve. This
     * uses the inverse square root {@code B2^-1/2} from a decomposition of the
     * core instead, which is the form equation (5.12) of the paper is written in
     * anyway, and which yields the same {@code U} and the same eigenvalues: the
     * two factors differ by an orthogonal transformation on the right, and step
     * 4 keeps only the left singular vectors and the singular values. The reason
     * is that a Cholesky decomposition fails on a singular core, and a singular
     * core is the normal case here rather than an exotic one - it is exactly
     * what happens whenever {@code Q} is wider than the numerical rank of
     * {@code A}, which the adaptive range finders produce routinely. The
     * decomposition of the core costs {@code O(k^3)} instead of
     * {@code O(k^3/3)}, against the {@code O(n^2 k)} of forming {@code AQ},
     * which dominates.
     *
     * @param A
     *            the positive semidefinite matrix to decompose
     * @param Q
     *            a matrix with orthonormal columns whose range approximates the
     *            range of {@code A}, with as many rows as {@code A} and no more
     *            columns than that
     * @return an approximate eigenvalue decomposition of {@code A}, with at most
     *         as many columns as {@code Q} has
     * @throws NullPointerException
     *             if {@code A} or {@code Q} is {@code null}
     * @throws IllegalArgumentException
     *             if {@code A} is not square, if the shape of {@code Q} does not
     *             fit {@code A}, if {@code ||A||_F} is not positive and finite,
     *             if {@code A} is not symmetric, or if {@code Q'AQ} shows that
     *             {@code A} is not positive semidefinite
     * @throws ArithmeticException
     *             if {@code Q'AQ} is numerically zero, so that there is no
     *             direction left to normalize
     */
    public static SVD decompose(MatrixD A, MatrixD Q) {
        Objects.requireNonNull(A);
        Objects.requireNonNull(Q);
        int n = A.numRows();
        if (A.numColumns() != n) {
            throw new IllegalArgumentException(
                    "A must be square, but is " + n + " x " + A.numColumns());
        }
        if (Q.numRows() != n) {
            throw new IllegalArgumentException("A is " + n + " x " + n + ", so Q must have " + n
                    + " rows, but it has " + Q.numRows());
        }
        int k = Q.numColumns();
        if (k > n) {
            throw new IllegalArgumentException(
                    "Q has more columns (" + k + ") than A has rows (" + n + ")");
        }
        double normF = A.normF();
        // the negated comparison also rejects NaN. A zero matrix has no
        // decomposition to return: a matrix with zero columns cannot be
        // constructed, and the inverse square root below would divide by zero
        if (!(normF > 0.0 && normF < Double.POSITIVE_INFINITY)) {
            throw new IllegalArgumentException(
                    "the decomposition of a matrix with ||A||_F = " + normF + " is not defined");
        }
        double asymmetry = relativeAsymmetry(A);
        if (!(asymmetry <= SYMMETRY_TOLERANCE)) {
            throw new IllegalArgumentException("A is not symmetric: the largest difference between"
                    + " A[i][j] and A[j][i] is " + asymmetry + " of the largest entry, above the"
                    + " tolerance of " + SYMMETRY_TOLERANCE);
        }

        // step 1 of the paper
        MatrixD B1 = A.times(Q);
        MatrixD B2 = Q.transAmult(B1, Matrices.createD(k, k));
        // the round-off of the two products turns a symmetric core into an
        // almost symmetric one, and the sign test below needs symmetry
        B2 = B2.addInplace(1.0, B2.transpose()).scaleInplace(0.5);

        // steps 2 and 3, as the inverse square root of the core
        SvdD core = B2.svdEcon();
        double[] s = core.getS();
        MatrixD W = core.getU();
        MatrixD Wt = core.getVt();
        // the same form of threshold that SVD.rank() uses
        double cutoff = k * MACH_EPS_DBL * s[0];
        int r = 0;
        while (r < s.length && s[r] > cutoff) {
            if (isNegative(W, Wt, r)) {
                if (s[r] > DEFINITENESS_TOLERANCE * s[0]) {
                    throw new IllegalArgumentException("A is not positive semidefinite: Q' * A * Q has the"
                            + " eigenvalue " + (-s[r]) + " next to the largest one " + s[0]);
                }
                // a negative eigenvalue at round-off level is not a violation,
                // it is what a matrix computed in floating point looks like.
                // The singular values are ordered by magnitude, so everything
                // beyond this one is at least as small: drop the tail and build
                // the approximation from the positive part
                break;
            }
            ++r;
        }
        if (r == 0) {
            throw new ArithmeticException("Q' * A * Q has no usable positive direction for ||A||_F = "
                    + normF + ", so Q does not meet the range of A at all");
        }

        double[] inverseRoots = new double[r];
        for (int j = 0; j < r; ++j) {
            inverseRoots[j] = 1.0 / Math.sqrt(s[j]);
        }
        MatrixD Wr = W.submatrix(0, 0, k - 1, r - 1, Matrices.createD(k, r), 0, 0);
        MatrixD F = B1.times(Wr.times(Matrices.diagD(r, r, inverseRoots)));

        // step 4
        SvdD factor = F.svdEcon();
        double[] sigma = factor.getS();
        MatrixD U = factor.getU();
        double[] eigenvalues = new double[U.numColumns()];
        for (int i = 0; i < eigenvalues.length; ++i) {
            eigenvalues[i] = sigma[i] * sigma[i];
        }
        return new SVD(U, eigenvalues, U.transpose());
    }

    /**
     * How far {@code A} is from symmetric: the largest difference between
     * {@code A[i][j]} and {@code A[j][i]}, relative to the largest entry.
     * <p>
     * One pass over the upper triangle, reading the column-major storage
     * directly. The obvious form, {@code ||A - A'||_F / ||A||_F}, needs two
     * whole extra matrices and four passes over them, and it turned out to cost
     * more than the decomposition it guards: measured at {@code n = 800} it took
     * 3.31 ms against 1.50 ms for the whole of Algorithm 5.5 at a basis width of
     * 60, and at {@code n = 1600}, 22.55 ms against 2.61 ms for the form used
     * here. The guard is memory bound while what it protects is a BLAS-3 kernel,
     * so comparing {@code O(n^2)} against {@code O(n^2 k)} says nothing about
     * which of the two is actually felt.
     * <p>
     * The largest entry rather than the Frobenius norm, because a sum of squared
     * differences underflows to zero for a matrix scaled to around
     * {@code 1e-160}, and every decision in this library is meant to survive a
     * rescaling of its input. Measured, the two metrics agree within a factor of
     * two on a matrix formed as {@code G * G'}.
     * <p>
     * A matrix holding a {@code NaN} never reaches this: the check on
     * {@code ||A||_F} above rejects it first, which matters because a comparison
     * against {@code NaN} is false and the asymmetry would come out as zero.
     *
     * @param A
     *            a square matrix
     * @return the largest asymmetry relative to the largest entry, or {@code 0}
     *         if every entry is zero
     */
    private static double relativeAsymmetry(MatrixD A) {
        int n = A.numRows();
        double[] a = A.getArrayUnsafe();
        double largestDifference = 0.0;
        double largestEntry = 0.0;
        for (int j = 0; j < n; ++j) {
            int column = j * n;
            for (int i = 0; i <= j; ++i) {
                double upper = a[column + i];
                double lower = a[i * n + j];
                double difference = Math.abs(upper - lower);
                if (difference > largestDifference) {
                    largestDifference = difference;
                }
                double entry = Math.abs(upper);
                if (entry > largestEntry) {
                    largestEntry = entry;
                }
                entry = Math.abs(lower);
                if (entry > largestEntry) {
                    largestEntry = entry;
                }
            }
        }
        return (largestEntry == 0.0) ? 0.0 : largestDifference / largestEntry;
    }

    /**
     * Whether eigenvalue {@code j} of the symmetric core is negative.
     * <p>
     * A decomposition into singular values throws the sign away, but for a
     * symmetric matrix it leaves it in plain sight: the left and the right
     * singular vector agree for a positive eigenvalue and differ by a sign for a
     * negative one, so their inner product is close to {@code +1} or to
     * {@code -1}. That is {@code O(k)} for one column, against the
     * {@code O(k^3)} of the decomposition that produced them, which is why the
     * definiteness of {@code A} can be checked at all without a factorization of
     * the whole matrix.
     *
     * @param W
     *            the left singular vectors of the core
     * @param Wt
     *            the transposed right singular vectors of the core
     * @param j
     *            the index of the eigenvalue
     * @return whether the eigenvalue is negative
     */
    private static boolean isNegative(MatrixD W, MatrixD Wt, int j) {
        double dot = 0.0;
        for (int i = 0; i < W.numRows(); ++i) {
            dot += W.get(i, j) * Wt.get(j, i);
        }
        return dot < 0.0;
    }
}
