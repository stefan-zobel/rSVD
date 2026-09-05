/*
 * Copyright 2026 Stefan Zobel
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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;

import org.junit.Test;

/**
 * The blocked organization of Algorithm 4.2, {@code computeQ(int)}, against the
 * column-by-column one it has to agree with.
 */
public class AdaRangeFinderBlockTest {

    // seeded so that a failure can be reproduced
    private static final long SEED = 7L;

    private static final int[] BLOCKS = { 1, 2, 4, 8, 16, 32 };

    private static final int[][] SHAPES = { { 220, 150 }, { 150, 220 }, { 160, 160 }, { 121, 120 },
            { 400, 60 } };

    private static MatrixD lowRank(int rows, int cols, int rank, long seed) {
        return Matrices.randomNormalD(rows, rank, seed).times(Matrices.randomNormalD(rank, cols, seed + 1000L));
    }

    private static MatrixD signalPlusNoise(int rows, int cols, int rank, double level, long seed) {
        return lowRank(rows, cols, rank, seed)
                .addInplace(1.0, Matrices.randomNormalD(rows, cols, seed + 2000L).scaleInplace(level));
    }

    /** the projection onto the range of {@code Q}, which is what a basis promises */
    private static MatrixD projection(MatrixD Q, MatrixD A) {
        return Q.times(Q.transpose().times(A));
    }

    private static double relativeResidual(MatrixD Q, MatrixD A) {
        return A.copy().addInplace(-1.0, projection(Q, A)).normF() / A.normF();
    }

    @Test
    public void testBlockSizeOneReproducesTheColumnByColumnPath() {
        // the strongest check there is: at a block size of one the samples are
        // drawn one at a time, so both paths see exactly the same test vectors
        // in the same order and have to stop in the same place
        for (int[] shape : SHAPES) {
            for (double epsilon : new double[] { 1.0e-3, 1.0e-1, 0.3 }) {
                MatrixD A = signalPlusNoise(shape[0], shape[1], 8, 0.5, 21L);
                MatrixD byColumn = new AdaRangeFinder(A, epsilon, SEED).computeQ();
                MatrixD byBlock = new AdaRangeFinder(A, epsilon, SEED).computeQ(1);
                String what = shape[0] + "x" + shape[1] + " at epsilon " + epsilon;
                assertEquals(what, byColumn.numColumns(), byBlock.numColumns());
                // the same subspace, but not necessarily the same basis of it:
                // a QR decomposition of a single column may flip its sign
                assertTrue(what, Matrices.approxEqual(projection(byColumn, A), projection(byBlock, A), 1.0e-8,
                        Checks.absTol(A)));
            }
        }
    }

    @Test
    public void testTheAccuracyTargetIsMet() {
        for (int[] shape : SHAPES) {
            for (double epsilon : new double[] { 1.0e-3, 1.0e-1, 0.3 }) {
                MatrixD A = signalPlusNoise(shape[0], shape[1], 8, 0.5, 31L);
                for (int block : BLOCKS) {
                    MatrixD Q = new AdaRangeFinder(A, epsilon, SEED).computeQ(block);
                    if (Q.numColumns() < Math.min(shape[0], shape[1])) {
                        // the target only binds while there is room left to grow
                        double residual = relativeResidual(Q, A);
                        assertTrue(shape[0] + "x" + shape[1] + " block " + block + " at epsilon " + epsilon
                                + ": residual " + residual, residual <= epsilon);
                    }
                }
            }
        }
    }

    @Test
    public void testOrthonormalColumns() {
        for (int[] shape : SHAPES) {
            for (int block : BLOCKS) {
                MatrixD A = signalPlusNoise(shape[0], shape[1], 8, 0.5, 41L);
                Checks.assertOrthonormal(new AdaRangeFinder(A, 1.0e-3, SEED).computeQ(block));
            }
        }
    }

    @Test
    public void testADeficientBlockDoesNotDestroyTheOrthonormality() {
        // the regression test for the one defect this path had. A block wider
        // than the rank that is left over is deficient, and the directions a QR
        // decomposition then invents for the deficient part are orthonormal
        // among themselves but not orthogonal to the basis built so far.
        // Appending them unchanged took |Q'Q - I| to 1.0 and made the range
        // finder run to the full width on an input of rank 12. The diagonal of
        // R does not separate them: it showed 4e-13 to 2.5e-11 for the invented
        // directions where the threshold for an exactly vanished one was 6e-14
        for (int rank : new int[] { 4, 12 }) {
            for (int block : new int[] { 8, 16, 32 }) {
                MatrixD A = lowRank(160, 160, rank, 51L + rank);
                MatrixD Q = new AdaRangeFinder(A, 1.0e-3, SEED).computeQ(block);
                String what = "rank " + rank + " with block " + block;
                Checks.assertOrthonormal(Q);
                // it must still recognize that the matrix is of low rank rather
                // than running to the full 160 columns
                assertTrue(what + ": " + Q.numColumns() + " columns", Q.numColumns() <= 2 * block);
                assertTrue(what + " is reconstructed",
                        Matrices.approxEqual(projection(Q, A), A, 1.0e-8, Checks.absTol(A)));
            }
        }
    }

    @Test
    public void testNeverFewerColumnsThanTheColumnByColumnPath() {
        // blocking tests the stopping criterion less often and on a different
        // window of look-ahead samples, so the two do not stop in the same
        // place. Measured over 2040 cases the blocked path never returned fewer
        // columns, but the excess is not bounded by the block size: where the
        // residual decays slowly the stopping point is sensitive to which
        // samples the estimator sees, and the worst excess measured was
        // 3.5 times the block size. The bound below is a regression guard with
        // headroom over that, not a proven invariant - "never fewer" and the
        // accuracy target are the properties that hold
        for (int[] shape : SHAPES) {
            for (double epsilon : new double[] { 1.0e-3, 0.3 }) {
                MatrixD A = signalPlusNoise(shape[0], shape[1], 8, 0.5, 61L);
                int byColumn = new AdaRangeFinder(A, epsilon, SEED).computeQ().numColumns();
                for (int block : BLOCKS) {
                    int byBlock = new AdaRangeFinder(A, epsilon, SEED).computeQ(block).numColumns();
                    String what = shape[0] + "x" + shape[1] + " block " + block + " at epsilon " + epsilon
                            + ": " + byBlock + " against " + byColumn;
                    assertTrue(what, byBlock >= byColumn);
                    assertTrue(what, byBlock - byColumn <= 4 * block + 8);
                }
            }
        }
    }

    @Test
    public void testNoiseFreeLowRankInputIsReconstructed() {
        for (int rank : new int[] { 1, 3, 12 }) {
            for (int block : BLOCKS) {
                MatrixD A = lowRank(220, 150, rank, 71L + rank);
                MatrixD Q = new AdaRangeFinder(A, 1.0e-3, SEED).computeQ(block);
                assertTrue("exact rank " + rank + " with block " + block,
                        Matrices.approxEqual(projection(Q, A), A, 1.0e-8, Checks.absTol(A)));
            }
        }
    }

    @Test
    public void testSameSeedGivesSameResult() {
        MatrixD A = signalPlusNoise(220, 150, 9, 0.5, 81L);
        for (int block : BLOCKS) {
            MatrixD first = new AdaRangeFinder(A, 1.0e-3, SEED).computeQ(block);
            MatrixD second = new AdaRangeFinder(A, 1.0e-3, SEED).computeQ(block);
            assertEquals("block " + block, first.numColumns(), second.numColumns());
            // not bit for bit: the seed fixes the test vectors, but the LAPACK
            // routines underneath are not reproducible to the last bit
            assertTrue("block " + block,
                    Matrices.approxEqual(first, second, 1.0e-10, Checks.absTol(first)));
        }
    }

    @Test
    public void testRepeatedCallsOnOneInstanceAgree() {
        MatrixD A = signalPlusNoise(220, 150, 9, 0.5, 91L);
        AdaRangeFinder finder = new AdaRangeFinder(A, 1.0e-3, SEED);
        MatrixD first = finder.computeQ(8);
        MatrixD second = finder.computeQ(8);
        assertEquals(first.numColumns(), second.numColumns());
        assertTrue(Matrices.approxEqual(first, second, 1.0e-10, Checks.absTol(first)));
    }

    @Test
    public void testABlockWiderThanTheCap() {
        // a block size beyond min(rows, columns) has to be capped rather than
        // ask the QR decomposition for more columns than the matrix has rows
        MatrixD A = signalPlusNoise(80, 40, 6, 0.5, 101L);
        MatrixD Q = new AdaRangeFinder(A, 1.0e-3, SEED).computeQ(500);
        Checks.assertOrthonormal(Q);
        assertTrue("columns: " + Q.numColumns(), Q.numColumns() >= 1 && Q.numColumns() <= 40);
    }

    @Test
    public void testASingleColumnAndASingleRow() {
        for (int block : new int[] { 1, 8 }) {
            Checks.assertOrthonormal(
                    new AdaRangeFinder(Matrices.randomNormalD(50, 1, 111L), 1.0e-3, SEED).computeQ(block));
            Checks.assertOrthonormal(
                    new AdaRangeFinder(Matrices.randomNormalD(1, 50, 121L), 1.0e-3, SEED).computeQ(block));
        }
    }

    @Test
    public void testScaleInvariance() {
        // every decision of this class is relative to ||A||_F, so scaling the
        // input must not change the number of columns
        MatrixD A = signalPlusNoise(150, 100, 7, 0.5, 131L);
        int expected = new AdaRangeFinder(A, 1.0e-3, SEED).computeQ(8).numColumns();
        for (double scale : new double[] { 1.0e+100, 1.0e+10, 1.0e-10, 1.0e-100, 1.0e-300 }) {
            MatrixD scaled = A.copy().scaleInplace(scale);
            assertEquals("scale " + scale, expected,
                    new AdaRangeFinder(scaled, 1.0e-3, SEED).computeQ(8).numColumns());
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBlockSizeZeroRejected() {
        new AdaRangeFinder(Matrices.randomNormalD(60, 40, 141L), 1.0e-3, SEED).computeQ(0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBlockSizeNegativeRejected() {
        new AdaRangeFinder(Matrices.randomNormalD(60, 40, 151L), 1.0e-3, SEED).computeQ(-8);
    }

    @Test(expected = ArithmeticException.class)
    public void testUnderflowingMatrixThrows() {
        // the same guard the column-by-column path has, and the same matrix it
        // is tested with: ||A||_F is the smallest positive double, so this
        // passes the constructor, but every product A * omega then underflows
        // to zero. A merely denormal matrix does not do it - its products stay
        // denormal but nonzero
        MatrixD A = Matrices.createD(60, 40);
        A.set(7, 3, Double.MIN_VALUE);
        new AdaRangeFinder(A, 1.0e-3, SEED).computeQ(8);
    }
}
