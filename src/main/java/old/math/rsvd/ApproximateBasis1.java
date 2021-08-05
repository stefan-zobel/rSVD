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
package old.math.rsvd;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;

public final class ApproximateBasis1 {

    // Oversampling parameter
    private static final int P = 5;

    private final MatrixD A;
    private final MatrixD AT;
    private final int columnsA;
    private final int targetRank;
    private final boolean transposed;

    public ApproximateBasis1(MatrixD A, int estimatedRank) {
        if (estimatedRank < 0) {
            throw new IllegalArgumentException("estimatedRank: " + estimatedRank);
        }
        int m = A.numRows();
        int n = A.numColumns();
        if (m < n) {
            this.A = A.transpose();
            AT = A;
            columnsA = m;
            transposed = true;
        } else {
            this.A = A;
            AT = A.transpose();
            columnsA = n;
            transposed = false;
        }
        targetRank = estimatedRank;
    }

    public MatrixD computeQ() {
        MatrixD Q = Matrices.randomUniformD(columnsA, targetRank + P, -1.0, 1.0);

        for (int i = 0; i < 4; ++i) {
            Q = A.times(Q).lud().getPL();
            Q = AT.times(Q).lud().getPL();
        }
        Q = A.times(Q).qrd().getQ();

        if (transposed) {
            Q = Q.transpose();
        }
        return Q;
    }
}
