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

public final class ApproximateBasis2 {

    // Oversampling parameter
    private static final int P = 5;

    private final MatrixD A;
    private final int m;
    private final int n;
    private final int targetRank;
    private final boolean transpose;

    public ApproximateBasis2(MatrixD A, int estimatedRank) {
        if (estimatedRank < 0) {
            throw new IllegalArgumentException("estimatedRank: " + estimatedRank);
        }
        m = A.numRows();
        n = A.numColumns();
        transpose = (m < n) ? true : false;
        this.A = A;
        targetRank = estimatedRank;
    }

    public MatrixD computeQ() {
        MatrixD Q = getRandomMatrix();
        if (transpose) {
            Q = loopWide(Q, A.transpose());
        } else {
            Q = loopTall(Q, A.transpose());
        }
        return Q;
    }

    private MatrixD loopWide(MatrixD Q, MatrixD AT) {
        for (int i = 0; i < 4; ++i) {
            Q = A.times(Q).lud().getPL();
            Q = AT.times(Q).lud().getPL();
        }
        return A.times(Q).qrd().getQ();
    }

    private MatrixD loopTall(MatrixD Q, MatrixD AT) {
        for (int i = 0; i < 4; ++i) {
            Q = AT.times(Q).lud().getPL();
            Q = A.times(Q).lud().getPL();
        }
        return Q = AT.times(Q).qrd().getQ();
    }

    private MatrixD getRandomMatrix() {
        MatrixD Omega = null;
        if (transpose) {
            Omega = Matrices.randomUniformD(targetRank + P, m, -1.0, 1.0);
            return Omega.times(A).transpose();
        }
        Omega = Matrices.randomUniformD(n, targetRank + P, -1.0, 1.0);
        return A.times(Omega);
    }
}
