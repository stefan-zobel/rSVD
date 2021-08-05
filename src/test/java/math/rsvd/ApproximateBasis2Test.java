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

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;
import net.jamu.matrix.SvdD;
import old.math.rsvd.ApproximateBasis2;

public class ApproximateBasis2Test {

    private static final double TOLERANCE = 1.0e-8;

    @Test
    public void testNaturalNumbersTall() {
        int m = 220;
        int n = 150;
        int estimatedRank = 2;
        MatrixD A = Matrices.naturalNumbersD(m, n);
        MatrixD Q = getQ(A, estimatedRank);
        MatrixD B = checkApproximation(Q, A, TOLERANCE);

        SVD svd = createSVD(B, A, Q, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test
    public void testNaturalNumbersWide() {
        int m = 150;
        int n = 220;
        int estimatedRank = 2;
        MatrixD A = Matrices.naturalNumbersD(m, n);
        MatrixD Q = getQ(A, estimatedRank);
        MatrixD B = checkApproximation(Q, A, TOLERANCE);

        SVD svd = createSVD(B, A, Q, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalTall() {
        int m = 220;
        int n = 150;
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomNormalD(m, n);
        MatrixD Q = getQ(A, estimatedRank);
        MatrixD B = checkApproximation(Q, A, TOLERANCE);

        SVD svd = createSVD(B, A, Q, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalWide() {
        int m = 150;
        int n = 220;
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomNormalD(m, n);
        MatrixD Q = getQ(A, estimatedRank);
        MatrixD B = checkApproximation(Q, A, TOLERANCE);

        SVD svd = createSVD(B, A, Q, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformTall() {
        int m = 220;
        int n = 150;
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomUniformD(m, n);
        MatrixD Q = getQ(A, estimatedRank);
        MatrixD B = checkApproximation(Q, A, TOLERANCE);

        SVD svd = createSVD(B, A, Q, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformWide() {
        int m = 150;
        int n = 220;
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomUniformD(m, n);
        MatrixD Q = getQ(A, estimatedRank);
        MatrixD B = checkApproximation(Q, A, TOLERANCE);

        SVD svd = createSVD(B, A, Q, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    private MatrixD checkApproximation(MatrixD Q, MatrixD A, double tolerance) {
        boolean transpose = A.numRows() < A.numColumns();
        MatrixD QT = Q.transpose();
        MatrixD QQT = Q.times(QT);
        if (QQT.numColumns() < A.numRows()) {
            MatrixD I = Matrices.identityD(A.numRows());
            QQT = I.setSubmatrixInplace(0, 0, QQT, 0, 0, QQT.endRow(), QQT.endCol());
        }
        MatrixD A_approx = QQT.times(A);
        boolean equal = Matrices.approxEqual(A_approx, A, tolerance);
        assertTrue("A_approx and A should be approximately equal", equal);

        if (transpose) {
            return QT.times(A);
        }
        return A.times(Q);
    }

    private SVD createSVD(MatrixD B, MatrixD A_expected, MatrixD Q, int estimatedRank) {
        boolean transpose = A_expected.numRows() < A_expected.numColumns();

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

        if (U.numColumns() > estimatedRank) {
            U = U.selectConsecutiveColumns(0, estimatedRank - 1);
        }
        if (U.numRows() > A_expected.numRows()) {
            U = U.selectSubmatrix(0, 0, A_expected.numRows() - 1, U.endCol());
        }
        if (Vt.numRows() > estimatedRank) {
            Vt = Vt.selectSubmatrix(0, 0, estimatedRank - 1, Vt.endCol());
        }
        if (Vt.numColumns() > A_expected.numColumns()) {
            Vt = Vt.selectConsecutiveColumns(0, A_expected.numColumns() - 1);
        }

        MatrixD S = Matrices.diagD(U.numColumns(), Vt.numRows(), sigma);

        return new SVD(U, S, Vt);
    }

    private void checkSVD(SVD svd, MatrixD A_expected, double tolerance) {
        MatrixD U = svd.U;
        MatrixD S = svd.S;
        MatrixD Vt = svd.Vt;

        MatrixD A_approx = U.timesTimes(S, Vt);
        System.out.println("A_approx: " + A_approx.numRows() + "x" + A_approx.numColumns());
        boolean equal = Matrices.approxEqual(A_approx, A_expected, tolerance);
        System.out.println(equal ? "EQUAL" : "NOT EQUAL");
        System.out.println("***");
        assertTrue("A and reconstruction of A should be approximately equal", equal);
    }

    private MatrixD getQ(MatrixD A, int estimatedRank) {
        return new ApproximateBasis2(A, estimatedRank).computeQ();
    }
}
