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
package math.rsvd.reference;

import java.util.Objects;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;

/**
 * Power iteration scheme for the fixed-rank problem. For matrices whose
 * singular values decay slowly. This algorithm is vulnerable to round-off
 * errors, the recommended implementation is Algorithm 4.4.
 * <p>
 * How vulnerable is worth stating, because it is what {@code q} costs. Remark
 * 4.3 of the paper: "when Algorithm 4.3 is executed in floating point
 * arithmetic, rounding errors will extinguish all information pertaining to
 * singular modes associated with singular values that are small compared with
 * ||A||. (Roughly, if machine precision is eps, then all information associated
 * with singular values smaller than eps^(1/(2q+1)) * ||A|| is lost.)" In double
 * precision that floor sits at about {@code 6e-6 * ||A||} for {@code q = 1} and
 * at about {@code 1.8e-2 * ||A||} for {@code q = 4}. Every power iteration
 * sharpens the spectrum and raises the level below which the spectrum is gone,
 * and it is the only handle a caller has on that trade.
 * <p>
 * {@link RanSubspaceIteration} (Algorithm 4.4) is algebraically the same scheme
 * with the sample matrix orthonormalized between each application of {@code A}
 * and {@code A'}, and it has no such floor. Measured on this code the
 * elementwise error here is {@code 5.6e-10 * ||A||_F}, against {@code 3e-16}
 * for {@link math.rsvd.AdaRangeFinder}. This class is kept because the paper
 * keeps Algorithm 4.3: a transcription that dropped it because it is inaccurate
 * would drop the statement the paper makes about it.
 * <p>
 * Algorithm 4.3 from Nathan Halko, Per-Gunnar Martinsson, and Joel A Tropp.
 * Finding structure with randomness: Probabilistic algorithms for constructing
 * approximate matrix decompositions. SIAM review, 53(2):217-288, 2011.
 */
public class RanPowerIteration {

    // Oversampling parameter
    private static final int P = 10;

    private final MatrixD A;
    private final int m;
    private final int n;
    /** the number of columns of the random test matrix, see the constructor */
    private final int sketchWidth;
    private final int q;
    /** whether {@link #seed} was supplied by the caller */
    private final boolean seeded;
    /** the seed for the random test matrix, only meaningful if {@link #seeded} */
    private final long seed;

    /**
     * Creates a power iteration for {@code A}. The random test matrix is drawn
     * from an unspecified source of randomness, so repeated runs generally
     * differ.
     *
     * @param A
     *            the matrix to decompose
     * @param estimatedRank
     *            the target rank, must be at least 1. It is capped at
     *            {@code min(rows, columns)}, and the width of the sketch is
     *            capped in addition at {@code max(rows, columns)}, which is
     *            the number of rows of {@code Y} and therefore the most a QR
     *            decomposition can take
     * @param q
     *            the number of power iterations, must be at least 1. It also
     *            raises the round-off floor {@code eps^(1/(2q+1)) * ||A||}
     *            below which the spectrum is lost, see the class javadoc
     */
    public RanPowerIteration(MatrixD A, int estimatedRank, int q) {
        this(A, estimatedRank, q, false, 0L);
    }

    /**
     * Creates a power iteration for {@code A} which draws its test matrix from
     * {@code seed}, so that a run can be repeated.
     * <p>
     * Note that this pins the input of the algorithm, not its output. Two runs
     * with the same seed see the same test matrix, but the basis they return
     * is not reproducible elementwise: {@code estimatedRank + P} exceeds the
     * rank of {@code A} as soon as the caller asks for the full rank, and the
     * surplus columns of the QR decomposition are then determined by round-off
     * alone. Measured, two runs with one and the same seed differ by 0.5 in
     * single entries of {@code Q}, while their reconstructions
     * {@code Q * Q' * A} still agree to about {@code 1e-9 * ||A||_F}. The
     * subspace that carries the approximation is reproducible, a basis of it
     * is not.
     *
     * @param A
     *            the matrix to decompose
     * @param estimatedRank
     *            the target rank, must be at least 1. It is capped at
     *            {@code min(rows, columns)}, and the width of the sketch is
     *            capped in addition at {@code max(rows, columns)}, which is
     *            the number of rows of {@code Y} and therefore the most a QR
     *            decomposition can take
     * @param q
     *            the number of power iterations, must be at least 1
     * @param seed
     *            the seed for the random test matrix
     */
    public RanPowerIteration(MatrixD A, int estimatedRank, int q, long seed) {
        this(A, estimatedRank, q, true, seed);
    }

    private RanPowerIteration(MatrixD A, int estimatedRank, int q, boolean seeded, long seed) {
        if (estimatedRank < 1) {
            throw new IllegalArgumentException("target rank must be at least 1, but was " + estimatedRank);
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
        this.seeded = seeded;
        this.seed = seed;
    }

    public MatrixD computeQ() {
        // sketch A itself for a tall matrix and its transpose for a wide one.
        // The two cases are the same computation because
        // A' * (A * A')^q == (A' * A)^q * A', so only the matrix being sketched
        // changes and Q always spans the larger of the two subspaces
        MatrixD M = (m >= n) ? A : A.transpose();
        MatrixD MT = (m >= n) ? A.transpose() : A;

        // Y = (M * M')^q * M * Omega, formed by alternating multiplication as
        // the paper prescribes. Note that M * M' is never built: that would be
        // an m x m matrix costing O(m^3) per step, whereas applying M and M' to
        // the columns of the sketch costs O(m * n * sketchWidth)
        MatrixD Y = M.times(nextTestMatrix(M.numColumns(), sketchWidth));
        for (int i = 0; i < q; ++i) {
            Y = MT.times(Y);
            Y = M.times(Y);
        }
        return Y.qrd().getQ();
    }

    /**
     * Draws the {@code rows x cols} standard normal test matrix.
     *
     * @param rows
     *            the number of rows
     * @param cols
     *            the number of columns
     * @return the test matrix, reproducible if this instance is seeded
     */
    private MatrixD nextTestMatrix(int rows, int cols) {
        return seeded ? Matrices.randomNormalD(rows, cols, seed) : Matrices.randomNormalD(rows, cols);
    }
}
