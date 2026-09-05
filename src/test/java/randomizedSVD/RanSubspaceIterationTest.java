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

import org.junit.Test;

import math.rsvd.Checks;

/**
 * This is the recommended algorithm for the fixed rank problem.
 */
public class RanSubspaceIterationTest {

    private static final int m = 220;
    private static final int n = 150;
    // tolerance requirements can be more strict here
    private static final double TOLERANCE = 1.0e-8;
    private static final int q = 4;

    @Test
    public void testNaturalNumbersTall() {
        // this is really low rank
        int estimatedRank = 2;
        MatrixD A = Matrices.naturalNumbersD(m, n);
        Checks.assertTall(A);
        MatrixD Q = getQ(A, estimatedRank, q);
        Checks.assertOrthonormal(Q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testNaturalNumbersWide() {
        // this is really low rank
        int estimatedRank = 2;
        MatrixD A = Matrices.naturalNumbersD(n, m);
        Checks.assertWide(A);
        MatrixD Q = getQ(A, estimatedRank, q);
        Checks.assertOrthonormal(Q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalTall() {
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomNormalD(m, n);
        Checks.assertTall(A);
        MatrixD Q = getQ(A, estimatedRank, q);
        Checks.assertOrthonormal(Q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalWide() {
        // high rank random noise
        int estimatedRank = Math.min(n, m);
        MatrixD A = Matrices.randomNormalD(n, m);
        Checks.assertWide(A);
        MatrixD Q = getQ(A, estimatedRank, q);
        Checks.assertOrthonormal(Q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformTall() {
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomUniformD(m, n);
        Checks.assertTall(A);
        MatrixD Q = getQ(A, estimatedRank, q);
        Checks.assertOrthonormal(Q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformWide() {
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomUniformD(n, m);
        Checks.assertWide(A);
        MatrixD Q = getQ(A, estimatedRank, q);
        Checks.assertOrthonormal(Q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testSinglePowerIteration() {
        // q == 1 skips the sweep entirely, so the very first decomposition is
        // also the last one. It used to be an LU whenever the sketch was wider
        // than tall, and the basis returned for a nearly square matrix then
        // deviated from orthonormality by 2.5e+01 instead of 1.3e-15
        MatrixD A = Matrices.randomNormalD(150, 145, 13L);
        MatrixD Q = getQ(A, 145, 1);
        Checks.assertOrthonormal(Q);
        Checks.checkFactorization(Q, A, TOLERANCE);
    }

    @Test
    public void testNearlySquare() {
        // the sketch width used to exceed the row count of Y here. This class
        // did not throw like the other two, it silently fell back to LU
        for (int[] shape : new int[][] { { 150, 145 }, { 145, 150 }, { 150, 150 } }) {
            int rows = shape[0];
            int cols = shape[1];
            MatrixD A = Matrices.randomNormalD(rows, cols, rows + cols);
            MatrixD Q = getQ(A, Math.min(rows, cols), q);
            Checks.assertOrthonormal(Q);
            assertTrue("Q has " + Q.numColumns() + " columns for a " + rows + "x" + cols + " matrix",
                    Q.numColumns() <= Math.max(rows, cols));
            Checks.checkFactorization(Q, A, TOLERANCE);
        }
    }

    @Test
    public void testEstimatedRankTooLarge() {
        // a rank beyond min(rows, columns) cannot exist, so it is capped rather
        // than allowed to blow up the sketch
        MatrixD A = Matrices.randomNormalD(60, 40, 12L);
        MatrixD Q = getQ(A, 1000, q);
        Checks.assertOrthonormal(Q);
        assertTrue("Q has " + Q.numColumns() + " columns", Q.numColumns() <= 60);
        Checks.checkFactorization(Q, A, TOLERANCE);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testEstimatedRankZero() {
        // a rank 0 subspace cannot be represented: jamu has no matrix with zero
        // columns, so asking for it is a caller error rather than something to
        // silently turn into a basis of P columns
        new RanSubspaceIteration(Matrices.randomNormalD(60, 40, 1L), 0, q);
    }

    private MatrixD getQ(MatrixD A, int estimatedRank, int q) {
        return new RanSubspaceIteration(A, estimatedRank, q).computeQ();
    }
}
