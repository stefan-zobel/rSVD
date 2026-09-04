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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;
import randomizedSVD.Checks;

import org.junit.Test;

public class SVDTest {

    private static final int m = 220;
    private static final int n = 150;
    // seeded so that a failure can be reproduced
    private static final long SEED = 7L;

    private static SVD decompose(MatrixD A, int estimatedRank) {
        return new ApproximateBasis(A, estimatedRank, SEED).computeSVD();
    }

    /** an exactly rank r matrix, the same construction as in AdaRangeFinderTest */
    private static MatrixD lowRank(int rows, int cols, int rank, long seed) {
        return Matrices.randomNormalD(rows, rank, seed).times(Matrices.randomNormalD(rank, cols, seed + 1000L));
    }

    private static double maxAbsDifference(MatrixD X, MatrixD Y) {
        assertEquals(X.numRows(), Y.numRows());
        assertEquals(X.numColumns(), Y.numColumns());
        double[] a = X.getArrayUnsafe();
        double[] b = Y.getArrayUnsafe();
        double worst = 0.0;
        for (int i = 0; i < a.length; ++i) {
            worst = Math.max(worst, Math.abs(a[i] - b[i]));
        }
        return worst;
    }

    @Test
    public void testSizeAndShapes() {
        SVD svd = decompose(Matrices.randomNormalD(m, n, 1L), 40);
        assertEquals(40, svd.size());
        assertEquals(40, svd.getSingularValues().length);
        assertEquals(m, svd.getU().numRows());
        assertEquals(40, svd.getU().numColumns());
        assertEquals(40, svd.getVt().numRows());
        assertEquals(n, svd.getVt().numColumns());
    }

    @Test
    public void testSingularValuesAreDescending() {
        double[] sigma = decompose(Matrices.randomNormalD(m, n, 2L), 40).getSingularValues();
        for (int i = 1; i < sigma.length; ++i) {
            assertTrue("singular value " + i + " is larger than its predecessor: " + sigma[i] + " > "
                    + sigma[i - 1], sigma[i] <= sigma[i - 1]);
        }
        assertTrue("the largest singular value should be positive, but was " + sigma[0], sigma[0] > 0.0);
    }

    @Test
    public void testSingularValuesAreACopy() {
        SVD svd = decompose(Matrices.randomNormalD(m, n, 3L), 20);
        double[] first = svd.getSingularValues();
        double original = first[0];
        first[0] = -1.0;
        // a caller must not be able to corrupt the decomposition through the
        // array it was handed, and jamu's SvdD.getS() does hand out its own
        assertEquals(original, svd.getSingularValues()[0], 0.0);
        assertEquals(original, svd.getS().get(0, 0), 0.0);
    }

    @Test
    public void testDiagonalFormCarriesTheSingularValues() {
        SVD svd = decompose(Matrices.randomNormalD(m, n, 4L), 25);
        double[] sigma = svd.getSingularValues();
        MatrixD S = svd.getS();
        assertEquals(sigma.length, S.numRows());
        assertEquals(sigma.length, S.numColumns());
        for (int i = 0; i < S.numRows(); ++i) {
            for (int j = 0; j < S.numColumns(); ++j) {
                assertEquals((i == j) ? sigma[i] : 0.0, S.get(i, j), 0.0);
            }
        }
    }

    @Test
    public void testReconstructMatchesTheExplicitChain() {
        // reconstruct() skips the dense diagonal matrix, so it is held against
        // the literal product it replaces. Measured, the two agree to about
        // 2.2e-15, which is the different summation order and nothing else
        for (int[] shape : new int[][] { { m, n }, { n, m }, { 150, 150 }, { 60, 40 } }) {
            int rows = shape[0];
            int cols = shape[1];
            MatrixD A = Matrices.randomNormalD(rows, cols, rows + cols);
            SVD svd = decompose(A, Math.min(rows, cols));
            MatrixD chain = svd.getU().timesTimes(svd.getS(), svd.getVt());
            double worst = maxAbsDifference(chain, svd.reconstruct());
            assertTrue("reconstruct() and U * S * Vt disagree by " + worst + " for a " + rows + "x" + cols
                    + " matrix", worst <= 1.0e-12 * A.normF());
        }
    }

    @Test
    public void testReconstructApproximatesTheInput() {
        for (int rank = 1; rank <= 8; ++rank) {
            MatrixD A = lowRank(m, n, rank, rank);
            MatrixD approx = decompose(A, rank).reconstruct();
            double worst = maxAbsDifference(approx, A);
            assertTrue("rank " + rank + " input is reconstructed with an error of " + worst,
                    worst <= Checks.absTol(A));
        }
    }

    @Test
    public void testRankOfAnExactlyLowRankInput() {
        // the estimated rank is chosen generously, so the numerical rank has
        // room to come out below it
        for (int rank = 1; rank <= 8; ++rank) {
            assertEquals("exact rank " + rank + " input", rank, decompose(lowRank(m, n, rank, rank), 20).rank());
        }
    }

    @Test
    public void testRankIsCappedByTheEstimatedRank() {
        // rank() reports the rank of the approximation, not of the input: a
        // full rank matrix decomposed with an estimated rank of 5 has nothing
        // more to report than 5
        assertEquals(5, decompose(Matrices.randomNormalD(m, n, 9L), 5).rank());
    }

    @Test
    public void testRankWithAnExplicitTolerance() {
        SVD svd = decompose(Matrices.randomNormalD(m, n, 10L), 30);
        double[] sigma = svd.getSingularValues();
        assertEquals(30, svd.rank(0.0));
        assertEquals(0, svd.rank(sigma[0]));
        // a threshold just below the tenth value keeps exactly ten
        assertEquals(10, svd.rank(Math.nextDown(sigma[9])));
    }

    @Test
    public void testTruncate() {
        SVD svd = decompose(Matrices.randomNormalD(m, n, 11L), 30);
        double[] sigma = svd.getSingularValues();
        SVD small = svd.truncate(7);
        assertEquals(7, small.size());
        assertEquals(m, small.getU().numRows());
        assertEquals(7, small.getU().numColumns());
        assertEquals(7, small.getVt().numRows());
        assertEquals(n, small.getVt().numColumns());
        double[] expected = new double[7];
        System.arraycopy(sigma, 0, expected, 0, 7);
        assertArrayEquals(expected, small.getSingularValues(), 0.0);
        // the vectors have to be the leading ones, not just any seven
        assertEquals(0.0, maxAbsDifference(small.getU(), svd.getU().selectConsecutiveColumns(0, 6)), 0.0);
    }

    @Test
    public void testTruncateToItsOwnSizeReturnsTheSameInstance() {
        SVD svd = decompose(Matrices.randomNormalD(m, n, 12L), 30);
        assertSame(svd, svd.truncate(svd.size()));
    }

    @Test
    public void testTruncateKeepsTheApproximationOfALowRankInput() {
        // 8 of the 20 computed singular values carry the whole matrix, so
        // throwing the other 12 away must not change the reconstruction
        MatrixD A = lowRank(m, n, 8, 13L);
        SVD svd = decompose(A, 20);
        double worst = maxAbsDifference(svd.truncate(8).reconstruct(), A);
        assertTrue("truncating a rank 8 input to 8 gives an error of " + worst, worst <= Checks.absTol(A));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTruncateToZero() {
        decompose(Matrices.randomNormalD(m, n, 14L), 30).truncate(0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTruncateBeyondTheSize() {
        decompose(Matrices.randomNormalD(m, n, 15L), 30).truncate(31);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMismatchedFactorsRejected() {
        MatrixD U = Matrices.randomNormalD(60, 5, 1L);
        MatrixD Vt = Matrices.randomNormalD(4, 40, 2L);
        new SVD(U, new double[] { 5.0, 4.0, 3.0, 2.0, 1.0 }, Vt);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTooFewSingularValuesRejected() {
        MatrixD U = Matrices.randomNormalD(60, 5, 1L);
        MatrixD Vt = Matrices.randomNormalD(5, 40, 2L);
        new SVD(U, new double[] { 5.0, 4.0 }, Vt);
    }

    @Test
    public void testLongerSingularValueArrayIsTruncated() {
        // the economy sized decomposition of the underlying library hands out
        // an array that can be longer than the decomposition is wide
        MatrixD U = Matrices.randomNormalD(60, 3, 1L);
        MatrixD Vt = Matrices.randomNormalD(3, 40, 2L);
        SVD svd = new SVD(U, new double[] { 5.0, 4.0, 3.0, 2.0, 1.0 }, Vt);
        assertEquals(3, svd.size());
        assertArrayEquals(new double[] { 5.0, 4.0, 3.0 }, svd.getSingularValues(), 0.0);
    }
}
