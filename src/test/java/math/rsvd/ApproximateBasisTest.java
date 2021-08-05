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

public class ApproximateBasisTest {

    private static final double TOLERANCE = 1.0e-8;

    @Test
    public void testNaturalNumbersTall() {
        int m = 220;
        int n = 150;
        int estimatedRank = 2;
        MatrixD A = Matrices.naturalNumbersD(m, n);
        SVD svd = getSVD(A, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test
    public void testNaturalNumbersWide() {
        int m = 150;
        int n = 220;
        int estimatedRank = 2;
        MatrixD A = Matrices.naturalNumbersD(m, n);
        SVD svd = getSVD(A, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalTall() {
        int m = 220;
        int n = 150;
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomNormalD(m, n);
        SVD svd = getSVD(A, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalWide() {
        int m = 150;
        int n = 220;
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomNormalD(m, n);
        SVD svd = getSVD(A, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformTall() {
        int m = 220;
        int n = 150;
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomUniformD(m, n);
        SVD svd = getSVD(A, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformWide() {
        int m = 150;
        int n = 220;
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomUniformD(m, n);
        SVD svd = getSVD(A, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    private void checkSVD(SVD svd, MatrixD A_expected, double tolerance) {
        MatrixD U = svd.U;
        MatrixD S = svd.S;
        MatrixD Vt = svd.Vt;

        MatrixD A_approx = U.timesTimes(S, Vt);
        boolean equal = Matrices.approxEqual(A_approx, A_expected, tolerance);
        assertTrue("A and reconstruction of A should be approximately equal", equal);
    }

    private SVD getSVD(MatrixD A, int estimatedRank) {
        return new ApproximateBasis(A, estimatedRank).computeSVD();
    }
}
