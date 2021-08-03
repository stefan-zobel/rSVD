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
 * Power iteration scheme for the fixed-rank problem. For matrices whose
 * singular values decay slowly. This algorithm is vulnerable to round-off
 * errors, the recommended implementation is Algorithm 4.4.
 * <p>
 * Algorithm 4.3 from Nathan Halko, Per-Gunnar Martinsson, and Joel A Tropp.
 * Finding structure with randomness: Probabilistic algorithms for constructing
 * approximate matrix decompositions. SIAM review, 53(2):217–288, 2011.
 */
public class RanPowerIteration {

    // Oversampling parameter
    private static final int P = 10;

    private final MatrixD A;
    private final int n;
    private final int targetRank;
    private final int q;

    public RanPowerIteration(MatrixD A, int estimatedRank, int q) {
        if (estimatedRank < 0) {
            throw new IllegalArgumentException("negative target rank: " + estimatedRank);
        }
        if (q < 1) {
            throw new IllegalArgumentException("q must be at least 1. q = " + q);
        }
        this.A = Objects.requireNonNull(A);
        this.n = A.numColumns();
        this.targetRank = estimatedRank;
        this.q = q;
    }

    public MatrixD computeQ() {
        // AT: n x m
        MatrixD AT = A.transpose();
        // tmp2: m x m
        MatrixD tmp2 = Matrices.createD(A.numRows(), A.numRows());
        // m x m
        MatrixD B = A.mult(AT, tmp2);
        // tmp1: n x m
        MatrixD tmp1 = Matrices.createD(n, A.numRows());
        for (int i = 2; i <= q; ++i) {
            // B = AT.times(B);
            // (n x m) * (m x m) = (n x m)
            tmp1 = AT.mult(B, tmp1);
            // B = A.times(B);
            // (m x n) * (n x m) = m x m
            B = A.mult(tmp1, tmp2);
        }
        // (m x m) * (m x n) = (m x n)
        B = B.times(A);
        MatrixD Omega = Matrices.randomNormalD(n, targetRank + P);
        MatrixD Y = B.times(Omega);
        MatrixD Q = decompose(Y);
        return Q;
    }

    private MatrixD decompose(MatrixD Y) {
        return Y.qrd().getQ();
    }
}
