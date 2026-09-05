/*
 * Copyright 2021, 2026 Stefan Zobel
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
    // seeded so that a failure can be reproduced
    private static final long SEED = 7L;

    @Test
    public void testNaturalNumbersTall() {
        int m = 220;
        int n = 150;
        int estimatedRank = 2;
        MatrixD A = Matrices.naturalNumbersD(m, n);
        Checks.assertTall(A);
        SVD svd = getSVD(A, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test
    public void testNaturalNumbersWide() {
        int m = 150;
        int n = 220;
        int estimatedRank = 2;
        MatrixD A = Matrices.naturalNumbersD(m, n);
        Checks.assertWide(A);
        SVD svd = getSVD(A, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalTall() {
        int m = 220;
        int n = 150;
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomNormalD(m, n, 1L);
        Checks.assertTall(A);
        SVD svd = getSVD(A, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalWide() {
        int m = 150;
        int n = 220;
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomNormalD(m, n, 2L);
        Checks.assertWide(A);
        SVD svd = getSVD(A, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformTall() {
        int m = 220;
        int n = 150;
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomUniformD(m, n, 3L);
        Checks.assertTall(A);
        SVD svd = getSVD(A, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformWide() {
        int m = 150;
        int n = 220;
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomUniformD(m, n, 4L);
        Checks.assertWide(A);
        SVD svd = getSVD(A, estimatedRank);
        checkSVD(svd, A, TOLERANCE);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testEstimatedRankZero() {
        // a rank 0 subspace cannot be represented: jamu has no matrix with zero
        // columns. Asking for it used to reach createSVD and fail there with
        // "Illegal column index -1", which says nothing about the cause
        new ApproximateBasis(Matrices.randomNormalD(60, 40, 1L), 0, SEED);
    }

    @Test(expected = NullPointerException.class)
    public void testNullMatrixRejected() {
        new ApproximateBasis(null, 2, SEED);
    }

    private void checkSVD(SVD svd, MatrixD A_expected, double tolerance) {
        MatrixD A_approx = svd.reconstruct();
        boolean equal = Matrices.approxEqual(A_approx, A_expected, tolerance, Checks.absTol(A_expected));
        assertTrue("A and reconstruction of A should be approximately equal", equal);
    }

    private SVD getSVD(MatrixD A, int estimatedRank) {
        return new ApproximateBasis(A, estimatedRank, SEED).computeSVD();
    }
}
