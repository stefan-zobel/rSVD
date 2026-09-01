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
    /** the numerical rank of A cannot exceed min(rows, columns) */
    private final int maxCols;

    /**
     * Creates a range finder for {@code A} which uses {@link #DEFAULT_EPSILON}
     * as its relative accuracy target.
     *
     * @param A
     *            the matrix whose approximate range is sought
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
     * invariant under a rescaling of {@code A}.
     *
     * @param A
     *            the matrix whose approximate range is sought
     * @param epsilon
     *            the relative accuracy target, must be in the range
     *            {@code (0.0, 1.0]}
     * @throws IllegalArgumentException
     *             if {@code epsilon} is not in the range {@code (0.0, 1.0]}
     */
    public AdaRangeFinder(MatrixD A, double epsilon) {
        this.A = Objects.requireNonNull(A);
        // the negated comparison also rejects NaN
        if (!(epsilon > 0.0 && epsilon <= 1.0)) {
            throw new IllegalArgumentException("epsilon: " + epsilon);
        }
        this.n = A.numColumns();
        this.bound = epsilon * A.normF() * BOUND_FACTOR;
        this.maxCols = Math.min(A.numRows(), A.numColumns());
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

    public MatrixD computeQ() {

        ArrayList<MatrixD> vectors = new ArrayList<>(r);
        for (int k = 0; k < r; ++k) {
            vectors.add(A.times(Matrices.randomNormalD(n, 1)));
        }

        MatrixD y = vectors.get(0);
        double norm = norm(y);
        if (norm <= MACH_EPS_DBL) {
            // this also covers the case of a zero matrix (where bound == 0.0)
            return null;
        }
        double max = getMax(vectors);

        MatrixD q = Matrices.sameDimD(y);
        q = y.scale(1.0 / norm, q);

        MatrixD Q = q.copy();

        shift(vectors, q, Q);

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
            if (norm <= MACH_EPS_DBL) {
                break;
            }
            q = y.scale(1.0 / norm, q);
            Q = Q.appendColumn(q);

            shift(vectors, q, Q);

            max = getMax(vectors);
        }

        return Q;
    }

    private void shift(ArrayList<MatrixD> vectors, MatrixD q, MatrixD Q) {
        vectors.remove(0);
        MatrixD omega = Matrices.randomNormalD(n, 1);
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
