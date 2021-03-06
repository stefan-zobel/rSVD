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
        targetRank = Math.min(estimatedRank, Math.min(m, n));
    }

    public SVD computeSVD() {
        MatrixD[] BQ = computeBQ();
        MatrixD B = BQ[0];
        MatrixD Q = BQ[1];
        MatrixD QT = BQ[2];

        SvdD svd = B.svdEcon();
        MatrixD U_tilde = svd.getU();
        double[] sigma = svd.getS();
        MatrixD Vt = svd.getVt();

        MatrixD U = null;
        if (transpose) {
            U = Q.times(U_tilde);
        } else {
            U = U_tilde;
            Vt = Vt.times(QT);
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
            // should never happen
            U = U.selectSubmatrix(0, 0, m - 1, U.endCol());
        }
        if (Vt.numColumns() > n) {
            // should never happen
            Vt = Vt.selectConsecutiveColumns(0, n - 1);
        }
        MatrixD S = Matrices.diagD(U.numColumns(), Vt.numRows(), sigma);
        return new SVD(U, S, Vt);
    }

    private MatrixD[] computeBQ() {
        MatrixD Q = computeQ();
        MatrixD QT = Q.transpose();
        if (transpose) {
            return new MatrixD[] { QT.times(A), Q, QT };
        }
        return new MatrixD[] { A.times(Q), Q, QT };
    }

    private MatrixD computeQ() {
        MatrixD Q = getRandomMatrix();
        if (transpose) {
            Q = loopWideSaveAllocations(Q, A.transpose());
        } else {
            Q = loopTallSaveAllocations(Q, A.transpose());
        }
        return Q;
    }

    protected MatrixD loopWide(MatrixD Q, MatrixD AT) {
        for (int i = 0; i < 4; ++i) {
            Q = A.times(Q).lud().getPL();
            Q = AT.times(Q).lud().getPL();
        }
        return A.times(Q).qrd().getQ();
    }

    protected MatrixD loopTall(MatrixD Q, MatrixD AT) {
        for (int i = 0; i < 4; ++i) {
            Q = AT.times(Q).lud().getPL();
            Q = A.times(Q).lud().getPL();
        }
        return AT.times(Q).qrd().getQ();
    }

    private MatrixD loopWideSaveAllocations(MatrixD Q, MatrixD AT) {
        MatrixD C1 = Matrices.createD(A.numRows(), Q.numColumns());
        MatrixD C2 = null;

        Q = A.mult(Q, C1).lud().getPL();
        if (Q.numColumns() != C1.numColumns()) {
            C2 = Matrices.createD(AT.numRows(), Q.numColumns());
        } else {
            C2 = Matrices.createD(AT.numRows(), C1.numColumns());
        }
        Q = AT.mult(Q, C2).lud().getPL();
        if (Q.numColumns() != C1.numColumns()) {
            C1 = Matrices.createD(A.numRows(), A.numRows());
        }

        for (int i = 0; i < 3; ++i) {
            Q = A.mult(Q, C1).lud().getPL();
            Q = AT.mult(Q, C2).lud().getPL();
        }
        return A.mult(Q, C1).qrd().getQ();
    }

    private MatrixD loopTallSaveAllocations(MatrixD Q, MatrixD AT) {
        MatrixD C1 = Matrices.createD(AT.numRows(), Q.numColumns());
        MatrixD C2 = null;

        Q = AT.mult(Q, C1).lud().getPL();
        if (Q.numColumns() != AT.numRows()) {
            C2 = Matrices.createD(A.numRows(), Q.numColumns());
        } else {
            C2 = Matrices.createD(A.numRows(), AT.numRows());
        }
        Q = A.mult(Q, C2).lud().getPL();
        if (Q.numColumns() != C1.numColumns()) {
            C1 = Matrices.createD(AT.numRows(), AT.numRows());
        }

        for (int i = 0; i < 3; ++i) {
            Q = AT.mult(Q, C1).lud().getPL();
            Q = A.mult(Q, C2).lud().getPL();
        }
        return AT.mult(Q, C1).qrd().getQ();
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
