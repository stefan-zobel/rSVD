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

import static org.junit.Assert.assertTrue;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;
import net.jamu.matrix.SvdD;

public final class Checks {

    public static MatrixD checkFactorization(MatrixD Q, MatrixD A, double tolerance) {
        if (A.numRows() >= A.numColumns()) { // m >= n
            MatrixD B = Q.transpose().times(A);
            MatrixD A_approx = Q.times(B);
            boolean equal = Matrices.approxEqual(A_approx, A, tolerance);
            assertTrue("A_approx and A should be approximately equal", equal);
            return B;
        } else { // m < n
            MatrixD B = A.times(Q);
            MatrixD A_approx = B.times(Q.transpose());
            boolean equal = Matrices.approxEqual(A_approx, A, tolerance);
            assertTrue("A_approx and A should be approximately equal", equal);
            return B;
        }
    }

    public static void checkSVD(MatrixD B, MatrixD Q, MatrixD A_expected, double tolerance) {
        SvdD svdReduced = B.svd(true);
        if (A_expected.numRows() >= A_expected.numColumns()) { // m >= n
            // U
            MatrixD U_lowrank = Q.times(svdReduced.getU());
            MatrixD U_approx = Matrices.embed(A_expected.numRows(), A_expected.numColumns(), U_lowrank);
            // Sigma
            MatrixD tmp = Matrices.diagD(svdReduced.getS());
            MatrixD Sigma = Matrices.embed(A_expected.numColumns(), A_expected.numColumns(), tmp);
            // Vt
            MatrixD Vt = svdReduced.getVt();
            // A_approx
            MatrixD A_approx = U_approx.timesTimes(Sigma, Vt);
            boolean equal = Matrices.approxEqual(A_approx, A_expected, tolerance);
            assertTrue("A and reconstruction of A should be approximately equal", equal);
        } else { // m < n
            // U
            MatrixD U_lowrank = svdReduced.getU();
            MatrixD U_approx = Matrices.embed(A_expected.numRows(), A_expected.numColumns(), U_lowrank);
            // Sigma
            MatrixD tmp = Matrices.diagD(svdReduced.getS());
            MatrixD Sigma = Matrices.embed(A_expected.numColumns(), A_expected.numColumns(), tmp);
            // Vt
            MatrixD Vt = svdReduced.getVt().times(Q.transpose());
            Vt = Matrices.embed(A_expected.numColumns(), A_expected.numColumns(), Vt);
            // A_approx
            MatrixD A_approx = U_approx.timesTimes(Sigma, Vt);
            boolean equal = Matrices.approxEqual(A_approx, A_expected, tolerance);
            assertTrue("A and reconstruction of A should be approximately equal", equal);
        }
    }
}
