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
 * Randomized Range Finder. Fixed-rank problem, where the target rank of the
 * input matrix is specified in advance.
 * <p>
 * Algorithm 4.1 from Nathan Halko, Per-Gunnar Martinsson, and Joel A Tropp.
 * Finding structure with randomness: Probabilistic algorithms for constructing
 * approximate matrix decompositions. SIAM review, 53(2):217-288, 2011.
 */
public class RanRangeFinder {

    // Oversampling parameter
    private static final int P = 10;

    private final MatrixD A;
    private final int m;
    private final int n;
    /** the number of columns of the random test matrix, see the constructor */
    private final int sketchWidth;

    /**
     * Creates a range finder for {@code A}.
     * <p>
     * Both arguments are capped rather than rejected. {@code estimatedRank} is
     * a statement about {@code A} and cannot exceed {@code min(rows, columns)},
     * which is the largest rank {@code A} can have. The width of the sketch is
     * capped in addition at {@code max(rows, columns)}: the sketch {@code Y}
     * has that many rows in either shape branch, and a QR decomposition needs
     * at least as many rows as columns.
     *
     * @param A
     *            the matrix whose approximate range is sought
     * @param estimatedRank
     *            the target rank, must not be negative
     * @throws IllegalArgumentException
     *             if {@code estimatedRank} is negative
     */
    public RanRangeFinder(MatrixD A, int estimatedRank) {
        if (estimatedRank < 0) {
            throw new IllegalArgumentException("negative target rank: " + estimatedRank);
        }
        this.A = Objects.requireNonNull(A);
        this.m = A.numRows();
        this.n = A.numColumns();
        // the target rank is a statement about A and cannot exceed the largest
        // rank A can have. The sketch is oversampled on top of that, but only
        // up to the row count of Y, which is what a QR can still factor
        int cappedRank = Math.min(estimatedRank, Math.min(m, n));
        this.sketchWidth = Math.min(cappedRank + P, Math.max(m, n));
    }

    public MatrixD computeQ() {
        if (m >= n) {
            MatrixD Omega = Matrices.randomNormalD(n, sketchWidth);
            MatrixD Y = A.times(Omega);
            MatrixD Q = decompose(Y);
            return Q;
        } else {
            MatrixD Omega = Matrices.randomNormalD(sketchWidth, m);
            MatrixD Y = Omega.times(A).transpose();
            MatrixD Q = decompose(Y);
            return Q;
        }
    }

    private MatrixD decompose(MatrixD Y) {
        return Y.qrd().getQ();
    }
}
