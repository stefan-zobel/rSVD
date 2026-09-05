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

    /** the shapes that matter here: the basis used to depend on which one it was */
    private static final int[][] SHAPES = { { 220, 150 }, { 150, 220 }, { 160, 160 }, { 121, 120 },
            { 400, 60 }, { 60, 400 } };

    private static MatrixD signalPlusNoise(int rows, int cols, int rank, long seed) {
        return Matrices.randomNormalD(rows, rank, seed).times(Matrices.randomNormalD(rank, cols, seed + 1000L))
                .addInplace(1.0, Matrices.randomNormalD(rows, cols, seed + 2000L).scaleInplace(0.5));
    }

    private static double residual(MatrixD Q, MatrixD A) {
        return A.copy().addInplace(-1.0, Q.times(Q.transpose().times(A))).normF() / A.normF();
    }

    @Test
    public void testTheBasisHasTheRowsOfTheMatrixWhateverItsShape() {
        // the iteration works in the smaller dimension, so on a tall matrix it
        // ends in the row space. computeQ() has to carry that over, and this is
        // the check that says it did: the basis is of the range of A in every
        // shape, not of whichever space happened to be cheaper
        for (int[] shape : SHAPES) {
            for (int rank : new int[] { 3, 12, 40 }) {
                if (rank > Math.min(shape[0], shape[1])) {
                    continue;
                }
                MatrixD A = signalPlusNoise(shape[0], shape[1], rank, 11L + rank);
                MatrixD Q = new ApproximateBasis(A, rank, SEED).computeQ();
                String what = shape[0] + "x" + shape[1] + " rank " + rank;
                org.junit.Assert.assertEquals(what, shape[0], Q.numRows());
                Checks.assertOrthonormal(Q);
            }
        }
    }

    @Test
    public void testTheBasisCarriesTheApproximation() {
        // spanning the right space is not enough, it has to capture A as well
        // as the decomposition built from it does. It captures slightly more,
        // because it still carries the oversampled width that computeSVD()
        // truncates away
        for (int[] shape : SHAPES) {
            MatrixD A = signalPlusNoise(shape[0], shape[1], 12, 31L);
            MatrixD Q = new ApproximateBasis(A, 12, SEED).computeQ();
            SVD svd = new ApproximateBasis(A, 12, SEED).computeSVD();
            double byBasis = residual(Q, A);
            double byDecomposition = residual(svd.getU(), A);
            assertTrue(shape[0] + "x" + shape[1] + ": " + byBasis + " against " + byDecomposition,
                    byBasis <= byDecomposition * 1.05);
        }
    }

    @Test
    public void testTheBasisIsWiderThanTheRankThatWasAskedFor() {
        // the oversampling is deliberate and documented, so it is pinned here
        // rather than left to surprise a caller who compares with computeSVD()
        MatrixD A = signalPlusNoise(220, 150, 12, 41L);
        MatrixD Q = new ApproximateBasis(A, 12, SEED).computeQ();
        assertTrue("width was " + Q.numColumns(), Q.numColumns() > 12);
        assertTrue("width was " + Q.numColumns(), Q.numColumns() <= 150);
        org.junit.Assert.assertEquals(12, new ApproximateBasis(A, 12, SEED).computeSVD().size());
    }

    @Test
    public void testARankAtOrBeyondTheSmallerDimension() {
        // there the LU steps cap the width, and the QR factorization that
        // follows needs at least as many rows as columns
        for (int[] shape : new int[][] { { 20, 20 }, { 5, 7 }, { 7, 5 }, { 40, 30 } }) {
            int cap = Math.min(shape[0], shape[1]);
            for (int rank : new int[] { cap, cap + 8 }) {
                MatrixD A = Matrices.randomNormalD(shape[0], shape[1], 51L);
                MatrixD Q = new ApproximateBasis(A, rank, SEED).computeQ();
                String what = shape[0] + "x" + shape[1] + " rank " + rank;
                org.junit.Assert.assertEquals(what, shape[0], Q.numRows());
                assertTrue(what, Q.numColumns() <= cap);
                Checks.assertOrthonormal(Q);
                // at the full width the approximation is exact
                assertTrue(what, residual(Q, A) < 1.0e-12);
            }
        }
    }

    @Test
    public void testASingleRowAndASingleColumn() {
        for (int rank : new int[] { 1, 9 }) {
            MatrixD row = Matrices.randomNormalD(1, 40, 61L);
            MatrixD Q = new ApproximateBasis(row, rank, SEED).computeQ();
            org.junit.Assert.assertEquals(1, Q.numRows());
            Checks.assertOrthonormal(Q);
            assertTrue(residual(Q, row) < 1.0e-12);

            MatrixD column = Matrices.randomNormalD(40, 1, 71L);
            Q = new ApproximateBasis(column, rank, SEED).computeQ();
            org.junit.Assert.assertEquals(40, Q.numRows());
            Checks.assertOrthonormal(Q);
            assertTrue(residual(Q, column) < 1.0e-12);
        }
    }

    @Test
    public void testTheBasisIsReproducible() {
        MatrixD A = signalPlusNoise(220, 150, 9, 81L);
        ApproximateBasis basis = new ApproximateBasis(A, 9, SEED);
        MatrixD first = basis.computeQ();
        MatrixD second = basis.computeQ();
        org.junit.Assert.assertEquals(first.numColumns(), second.numColumns());
        assertTrue(Matrices.approxEqual(first, second, 1.0e-10, Checks.absTol(first)));
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
