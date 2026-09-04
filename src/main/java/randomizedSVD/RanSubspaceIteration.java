/*
 * Copyright 2021 Stefan Zobel
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
package randomizedSVD;

import java.util.Objects;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;

/**
 * Subspace iteration scheme for the fixed-rank problem. For matrices whose
 * singular values decay slowly. This is substantially more accurate in
 * floating-point arithmetic than algorithm 4.3
 * <p>
 * Algorithm 4.4 from Nathan Halko, Per-Gunnar Martinsson, and Joel A Tropp.
 * Finding structure with randomness: Probabilistic algorithms for constructing
 * approximate matrix decompositions. SIAM review, 53(2):217-288, 2011.
 */
public class RanSubspaceIteration {

    // Oversampling parameter
    private static final int P = 10;

    private final MatrixD A;
    private final int m;
    private final int n;
    /** the number of columns of the random test matrix, see the constructor */
    private final int sketchWidth;
    private final int q;

    /**
     * Creates a subspace iteration for {@code A}.
     * <p>
     * Both size arguments are capped rather than rejected.
     * {@code estimatedRank} is a statement about {@code A} and cannot exceed
     * {@code min(rows, columns)}, which is the largest rank {@code A} can have.
     * The width of the sketch is capped in addition at
     * {@code max(rows, columns)}: the sketch {@code Y} has that many rows in
     * either shape branch, and a QR decomposition needs at least as many rows
     * as columns.
     *
     * @param A
     *            the matrix whose approximate range is sought
     * @param estimatedRank
     *            the target rank, must not be negative
     * @param q
     *            the number of subspace iterations, must be at least 1
     * @throws IllegalArgumentException
     *             if {@code estimatedRank} is negative or {@code q} is less
     *             than 1
     */
    public RanSubspaceIteration(MatrixD A, int estimatedRank, int q) {
        if (estimatedRank < 0) {
            throw new IllegalArgumentException("negative target rank: " + estimatedRank);
        }
        if (q < 1) {
            throw new IllegalArgumentException("q must be at least 1. q = " + q);
        }
        this.A = Objects.requireNonNull(A);
        this.m = A.numRows();
        this.n = A.numColumns();
        // the target rank is a statement about A and cannot exceed the largest
        // rank A can have. The sketch is oversampled on top of that, but only
        // up to the row count of Y, which is what a QR can still factor
        int cappedRank = Math.min(estimatedRank, Math.min(m, n));
        this.sketchWidth = Math.min(cappedRank + P, Math.max(m, n));
        this.q = q;
    }

    public MatrixD computeQ() {
        MatrixD AT = A.transpose();
        MatrixD Omega = null;
        MatrixD Y = null;
        if (m >= n) {
            Omega = Matrices.randomNormalD(n, sketchWidth);
            Y = A.times(Omega);
        } else {
            Omega = Matrices.randomNormalD(sketchWidth, m);
            Y = Omega.times(A).transpose();
        }
        // for q == 1 the loop below never runs, so this first step is also the
        // last one and has to produce an orthonormal basis
        MatrixD Q = orthonormalize(Y);

        if (m >= n) {
            for (int j = 1; j < q; ++j) {
                Q = intermediateBasis(AT.times(Q));
                Q = orthonormalize(A.times(Q));
            }
        } else {
            // for a wide A the sweep runs on the transpose: Y above is
            // (Omega * A)' = A' * Omega', so Q spans the row space of A and
            // stays n x sketchWidth throughout. The order of A and AT is
            // therefore swapped with respect to the branch above
            for (int j = 1; j < q; ++j) {
                Q = intermediateBasis(A.times(Q));
                Q = orthonormalize(AT.times(Q));
            }
        }

        return Q;
    }

    /**
     * Computes an orthonormal basis of the range of {@code Y}.
     * <p>
     * Every sweep of the iteration ends here, and so does the whole
     * computation, so that the returned basis is orthonormal for every
     * {@code q}. The sketch width is capped such that {@code Y} is never wider
     * than tall at this point, which is what makes the QR possible.
     *
     * @param Y
     *            the sketch, with at least as many rows as columns
     * @return an orthonormal basis of the range of {@code Y}
     */
    private static MatrixD orthonormalize(MatrixD Y) {
        return Y.qrd().getQ();
    }

    /**
     * Computes a basis of the range of {@code Y} for an intermediate step of
     * the sweep.
     * <p>
     * Remark 4.1 of the paper permits a cheaper, non-orthogonal basis here, so
     * an LU decomposition is used where a QR is not possible anyway. That is
     * the case whenever {@code Y} is wider than tall, which the cap on the
     * sketch width does not rule out for the intermediate steps: the sketch
     * width is bounded by {@code max(rows, columns)}, while the intermediate
     * {@code Y} has only {@code min(rows, columns)} rows.
     *
     * @param Y
     *            the intermediate sketch
     * @return a basis of the range of {@code Y}, orthonormal only if
     *         {@code Y} has at least as many rows as columns
     */
    private static MatrixD intermediateBasis(MatrixD Y) {
        return (Y.numRows() < Y.numColumns()) ? Y.lud().getPL() : Y.qrd().getQ();
    }
}
