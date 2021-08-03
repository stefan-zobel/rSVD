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
 * approximate matrix decompositions. SIAM review, 53(2):217–288, 2011.
 */
public class RanRangeFinder {

    // Oversampling parameter
    private static final int P = 10;

    private final MatrixD A;
    private final int n;
    private final int targetRank;

    public RanRangeFinder(MatrixD A, int estimatedRank) {
        if (estimatedRank < 0) {
            throw new IllegalArgumentException("negative target rank: " + estimatedRank);
        }
        this.A = Objects.requireNonNull(A);
        this.n = A.numColumns();
        this.targetRank = estimatedRank;
    }

    public MatrixD computeQ() {
        MatrixD Omega = Matrices.randomNormalD(n, targetRank + P);
        MatrixD Y = A.times(Omega);
        MatrixD Q = qrDecompose(Y);
        return Q;
    }

    private MatrixD qrDecompose(MatrixD Y) {
        MatrixD Q = null;
        if (Y.numRows() < Y.numColumns()) {
            Y = Y.transpose();
            MatrixD RT = Y.qrd().getR();
            Q = RT.transpose();
        } else {
            Q = Y.qrd().getQ();
        }
        return Q;
    }
}
