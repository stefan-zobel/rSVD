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
package math.rsvd;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;
import net.jamu.matrix.SvdD;

public final class ApproximateBasis {

    // Oversampling parameter
    private static final int P = 5;

    private final MatrixD A;
    private final int m;
    private final int n;
    private final int targetRank;
    private final boolean transpose;

    public ApproximateBasis(MatrixD A, int estimatedRank) {
        if (estimatedRank < 0) {
            throw new IllegalArgumentException("estimatedRank: " + estimatedRank);
        }
        m = A.numRows();
        n = A.numColumns();
        transpose = (m < n) ? true : false;
        this.A = A;
        targetRank = estimatedRank;
    }

    public SVD computeSVD() {
        MatrixD[] BQ = computeBQ();
        MatrixD B = BQ[0];
        MatrixD Q = BQ[1];

        SvdD svd = B.svdEcon();
        MatrixD U_tilde = svd.getU();
        double[] sigma = svd.getS();
        MatrixD Vt = svd.getVt();

        MatrixD U = null;
        if (transpose) {
            U = Q.times(U_tilde);
        } else {
            U = U_tilde;
            Vt = Vt.times(Q.transpose());
        }
        return createSVD(U, sigma, Vt);
    }

    private SVD createSVD(MatrixD U, double[] sigma, MatrixD Vt) {
        if (U.numColumns() > targetRank) {
            U = U.selectConsecutiveColumns(0, targetRank - 1);
        }
        if (Vt.numRows() > targetRank) {
            Vt = Vt.selectSubmatrix(0, 0, targetRank - 1, Vt.endCol());
        }
        if (U.numRows() > m) {
            U = U.selectSubmatrix(0, 0, m - 1, U.endCol());
        }
        if (Vt.numColumns() > n) {
            Vt = Vt.selectConsecutiveColumns(0, n - 1);
        }
        MatrixD S = Matrices.diagD(U.numColumns(), Vt.numRows(), sigma);
        return new SVD(U, S, Vt);
    }

    private MatrixD[] computeBQ() {
        MatrixD Q = computeQ();
        if (transpose) {
            return new MatrixD[] { Q.transpose().times(A), Q };
        }
        return new MatrixD[] { A.times(Q), Q };
    }

    private MatrixD computeQ() {
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
