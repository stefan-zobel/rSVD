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

import java.util.ArrayList;
import java.util.Objects;
import java.util.Random;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;

/**
 * Adaptive Randomized Range Finder.
 * <p>
 * Algorithm 4.2 from Nathan Halko, Per-Gunnar Martinsson, and Joel A Tropp.
 * Finding structure with randomness: Probabilistic algorithms for constructing
 * approximate matrix decompositions. SIAM review, 53(2):217-288, 2011.
 */
public class AdaRangeFinder {

    /**
     * The default relative accuracy target used by
     * {@link #AdaRangeFinder(MatrixD)}.
     */
    public static final double DEFAULT_EPSILON = 1.0e-3;

    /** The IEEE 754 machine epsilon from Cephes: (2^-53) */
    private static final double MACH_EPS_DBL = 1.11022302462515654042e-16;
    private static final int r = 10;
    /** The constant 1 / (10 * sqrt(2 / pi)) from Algorithm 4.2 */
    private static final double BOUND_FACTOR = 1.0 / (10.0 * Math.sqrt(2.0 / Math.PI));

    private final MatrixD A;
    private final int n;
    /** epsilon * ||A||_F / (10 * sqrt(2 / pi)) */
    private final double bound;
    /**
     * MACH_EPS_DBL * ||A||_F, below which a vector computed from {@code A} is
     * indistinguishable from zero at working precision. Like {@link #bound}
     * this is relative to the scale of {@code A}, so that no decision of this
     * class depends on the absolute magnitude of the entries
     */
    private final double zeroTol;
    /** the numerical rank of A cannot exceed min(rows, columns) */
    private final int maxCols;
    /** whether {@link #seed} was supplied by the caller */
    private final boolean seeded;
    /** starting point of the seed sequence, only meaningful if {@link #seeded} */
    private final long seed;

    /**
     * Creates a range finder for {@code A} which uses {@link #DEFAULT_EPSILON}
     * as its relative accuracy target. The test vectors are drawn from an
     * unspecified source of randomness, so repeated runs generally differ.
     *
     * @param A
     *            the matrix whose approximate range is sought
     * @throws IllegalArgumentException
     *             if {@code ||A||_F} is not positive and finite
     */
    public AdaRangeFinder(MatrixD A) {
        this(A, DEFAULT_EPSILON);
    }

    /**
     * Creates a range finder for {@code A} which stops as soon as the estimated
     * residual {@code ||(I - Q * Q') * A||} has dropped below
     * {@code epsilon * ||A||_F}, or as soon as {@code Q} has
     * {@code min(rows, columns)} columns, whichever happens first. The
     * criterion is relative to the Frobenius norm of {@code A} and is therefore
     * invariant under a rescaling of {@code A}. That invariance holds as long
     * as the entries of {@code A} stay in the normal range of a {@code double};
     * once they turn denormal the products {@code A * omega} lose accuracy and
     * fewer columns are found.
     *
     * @param A
     *            the matrix whose approximate range is sought
     * @param epsilon
     *            the relative accuracy target, must be in the range
     *            {@code (0.0, 1.0]}
     * @throws IllegalArgumentException
     *             if {@code epsilon} is not in the range {@code (0.0, 1.0]}, or
     *             if {@code ||A||_F} is not positive and finite
     */
    public AdaRangeFinder(MatrixD A, double epsilon) {
        this(A, epsilon, false, 0L);
    }

    /**
     * Creates a reproducible range finder for {@code A}. Two range finders
     * constructed with the same matrix, the same {@code epsilon} and the same
     * {@code seed} draw the same test vectors and stop after the same number of
     * them, and repeated {@link #computeQ()} calls on one instance do so as
     * well.
     * <p>
     * The bases they return agree to round-off rather than bit for bit, and no
     * seed can change that. The BLAS and LAPACK routines underneath are free to
     * divide their work differently from one run to the next, and floating point
     * addition is not associative, so the same operands can add up to a slightly
     * different sum. Measured over 21 runs per case, the column count was
     * identical in every case - on inputs that stop early as well as on inputs
     * that run to the {@code min(rows, columns)} cap - and the entries of
     * {@code Q} agreed to {@code 1.2e-14} absolute.
     * <p>
     * Note that there is deliberately no {@code (MatrixD, long)} overload: it
     * would be chosen over {@code (MatrixD, double)} for an integer literal, so
     * {@code new AdaRangeFinder(A, 1)} would silently mean a seed rather than
     * an accuracy target. Use {@link #DEFAULT_EPSILON} explicitly instead.
     *
     * @param A
     *            the matrix whose approximate range is sought
     * @param epsilon
     *            the relative accuracy target, must be in the range
     *            {@code (0.0, 1.0]}
     * @param seed
     *            the starting point of the sequence of test vectors
     * @throws IllegalArgumentException
     *             if {@code epsilon} is not in the range {@code (0.0, 1.0]}, or
     *             if {@code ||A||_F} is not positive and finite
     */
    public AdaRangeFinder(MatrixD A, double epsilon, long seed) {
        this(A, epsilon, true, seed);
    }

    private AdaRangeFinder(MatrixD A, double epsilon, boolean seeded, long seed) {
        this.A = Objects.requireNonNull(A);
        // the negated comparison also rejects NaN
        if (!(epsilon > 0.0 && epsilon <= 1.0)) {
            throw new IllegalArgumentException("epsilon: " + epsilon);
        }
        double normF = A.normF();
        // the negated comparison also rejects NaN. A zero matrix has to be
        // rejected here because its range is the zero subspace, and a matrix
        // with zero columns cannot be constructed
        if (!(normF > 0.0 && normF < Double.POSITIVE_INFINITY)) {
            throw new IllegalArgumentException(
                    "the range of a matrix with ||A||_F = " + normF + " is not defined");
        }
        this.n = A.numColumns();
        this.bound = epsilon * normF * BOUND_FACTOR;
        this.zeroTol = MACH_EPS_DBL * normF;
        this.maxCols = Math.min(A.numRows(), A.numColumns());
        this.seeded = seeded;
        this.seed = seed;
    }

    /**
     * Draws the next {@code n x 1} test vector.
     * <p>
     * Note that the seeded factory methods of {@code Matrices} construct a new
     * {@code Random} from the seed on every call, so passing one and the same
     * seed to all of them would yield identical vectors and would silently
     * degenerate the algorithm. Each draw therefore consumes a fresh seed from
     * {@code seeds}.
     *
     * @param seeds
     *            the seed sequence, or {@code null} for an unseeded run
     * @return a new {@code n x 1} standard normal random vector
     */
    private MatrixD nextTestVector(Random seeds) {
        return (seeds == null) ? Matrices.randomNormalD(n, 1) : Matrices.randomNormalD(n, 1, seeds.nextLong());
    }

    private static double norm(MatrixD y) {
        // y is always a column vector, so normF() is its euclidean length
        // (norm2() would yield the same value but computes a full SVD)
        return y.normF();
    }

    private static double getMax(ArrayList<MatrixD> vectors) {
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < vectors.size(); ++i) {
            double norm = norm(vectors.get(i));
            if (norm > max) {
                max = norm;
            }
        }
        return max;
    }

    /**
     * Applies the orthogonal projector {@code I - Q * Q'} to the column vector
     * {@code y}, overwriting {@code y} with {@code y - Q * (Q' * y)}.
     * <p>
     * The projector is never formed explicitly. Doing so would need a
     * {@code rows x rows} matrix and {@code O(rows * rows)} work per
     * application, whereas the two matrix-vector products used here need
     * {@code O(rows * k)} work and no storage beyond the {@code k x 1} vector
     * of coefficients, where {@code k} is the number of columns of {@code Q}.
     *
     * @param Q
     *            a matrix with orthonormal columns
     * @param y
     *            the column vector to project, overwritten with the result
     */
    private static void project(MatrixD Q, MatrixD y) {
        // c = Q' * y
        MatrixD c = Matrices.createD(Q.numColumns(), 1);
        Q.transAmult(y, c);
        // y = y - Q * c (multAdd accumulates into y, it does not clear it)
        Q.multAdd(-1.0, c, y);
    }

    /**
     * Computes a matrix {@code Q} with orthonormal columns whose range
     * approximates the range of {@code A} to the accuracy target given at
     * construction time. {@code Q} has at least one and at most
     * {@code min(rows, columns)} columns and is never {@code null}.
     *
     * @return an orthonormal basis of the approximate range of {@code A}
     * @throws ArithmeticException
     *             if the products {@code A * omega} underflow to zero, which
     *             can only happen for a matrix whose nonzero entries are all
     *             denormal
     */
    public MatrixD computeQ() {

        // a local seed sequence keeps computeQ() idempotent and reentrant
        Random seeds = seeded ? new Random(seed) : null;

        ArrayList<MatrixD> vectors = new ArrayList<>(r);
        for (int k = 0; k < r; ++k) {
            vectors.add(A.times(nextTestVector(seeds)));
        }

        MatrixD y = vectors.get(0);
        double norm = norm(y);
        if (norm <= zeroTol) {
            // A itself is nonzero, the constructor rejects a zero matrix, so
            // the product A * omega must have underflowed. There is no basis
            // to return in that case: a matrix with zero columns cannot be
            // constructed, and normalizing y would divide by zero
            throw new ArithmeticException(
                    "A * omega underflowed to " + norm + " for ||A||_F = " + A.normF());
        }

        MatrixD q = Matrices.sameDimD(y);
        q = y.scale(1.0 / norm, q);

        MatrixD Q = q.copy();

        shift(vectors, q, Q, seeds);

        // the loop test has to see the test vectors in the state they are in
        // now, that is projected onto the orthogonal complement of the range of
        // the Q that has just been built. Taking the maximum before the shift
        // above would compare unprojected norms against the bound and would
        // always spend one iteration too many
        double max = getMax(vectors);

        // stop as soon as the residual estimate has dropped below
        // epsilon * ||A||_F / (10 * sqrt(2 / pi)), and never use more columns
        // than the numerical rank of A can possibly have
        while (max > bound && Q.numColumns() < maxCols) {

            // project the oldest test vector onto the orthogonal complement of
            // the range of Q. This overwrites vectors.get(0) in place, which is
            // safe because the shift() call below discards that element anyway
            y = vectors.get(0);
            project(Q, y);

            norm = norm(y);
            if (norm <= zeroTol) {
                break;
            }
            q = y.scale(1.0 / norm, q);
            Q = Q.appendColumn(q);

            shift(vectors, q, Q, seeds);

            max = getMax(vectors);
        }

        return Q;
    }

    private void shift(ArrayList<MatrixD> vectors, MatrixD q, MatrixD Q, Random seeds) {
        vectors.remove(0);
        MatrixD omega = nextTestVector(seeds);
        // yr is retained in the vectors list, so it must be a fresh matrix and
        // must not be written into a shared scratch buffer
        MatrixD yr = A.times(omega);
        project(Q, yr);
        vectors.add(yr);
        // re-orthogonalize the remaining test vectors against the newest basis
        // vector. That is the same projection with a single-column Q; the last
        // element is yr, which was already projected above
        for (int i = 0; i < vectors.size() - 1; ++i) {
            project(q, vectors.get(i));
        }
    }
}
