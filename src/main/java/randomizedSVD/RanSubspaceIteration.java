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
 * approximate matrix decompositions. SIAM review, 53(2):217–288, 2011.
 */
public class RanSubspaceIteration {

    // Oversampling parameter
    private static final int P = 10;

    private final MatrixD A;
    private final int m;
    private final int n;
    private final int targetRank;
    private final int q;

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
        this.targetRank = estimatedRank;
        this.q = q;
    }

    public MatrixD computeQ() {
        MatrixD AT = A.transpose();
        MatrixD Omega = null;
        MatrixD Y = null;
        if (m >= n) {
            Omega = Matrices.randomNormalD(n, targetRank + P);
            Y = A.times(Omega);
        } else {
            Omega = Matrices.randomNormalD(targetRank + P, m);
            Y = Omega.times(A).transpose();
        }
        MatrixD Q = decompose(Y);

        if (m >= n) {
            for (int j = 1; j < q; ++j) {
                Y = AT.times(Q);
                Q = decompose(Y);
                Y = A.times(Q);
                Q = decompose(Y);
            }
        } else {
            // XXX ???
            //throw new UnsupportedOperationException("m < k not yet implemented");
            for (int j = 1; j < q; ++j) {
                Y = A.times(Q);
                Q = decompose(Y);
                Y = AT.times(Q);
                Q = decompose(Y);
            }
        }

        return Q;
    }

    private MatrixD decompose(MatrixD Y) {
        MatrixD Q = null;
        if (Y.numRows() < Y.numColumns()) {
            Q = Y.lud().getPL();
        } else {
            Q = Y.qrd().getQ();
        }
        return Q;
    }
}
