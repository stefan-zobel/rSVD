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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;
import randomizedSVD.Checks;

import org.junit.Test;

import math.rsvd.AdaRangeFinder;

public class AdaRangeFinderTest {

    private static final int m = 220;
    private static final int n = 150;
    // the adaptive algorithm has quite good tolerance even for matrices which
    // are not that large
    private static final double TOLERANCE = 1.0e-7;
    // seeding the input matrix alone would not be enough: the algorithm draws
    // its own test vectors, so the finder has to be seeded as well
    private static final long SEED = 7L;

    @Test
    public void testNaturalNumbersTall() {
        MatrixD A = Matrices.naturalNumbersD(m, n);
        Checks.assertTall(A);
        MatrixD Q = getQ(A);
        MatrixD B = Checks.checkFactorization2(Q, A, TOLERANCE);
        Checks.checkSVD2(B, Q, A, TOLERANCE);
    }

    @Test
    public void testNaturalNumbersWide() {
        MatrixD A = Matrices.naturalNumbersD(n, m);
        Checks.assertWide(A);
        MatrixD Q = getQ(A);
        MatrixD B = Checks.checkFactorization2(Q, A, TOLERANCE);
        Checks.checkSVD2(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalTall() {
        MatrixD A = Matrices.randomNormalD(m, n, 1L);
        Checks.assertTall(A);
        MatrixD Q = getQ(A);
        MatrixD B = Checks.checkFactorization2(Q, A, TOLERANCE);
        Checks.checkSVD2(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalWide() {
        MatrixD A = Matrices.randomNormalD(n, m, 2L);
        Checks.assertWide(A);
        MatrixD Q = getQ(A);
        MatrixD B = Checks.checkFactorization2(Q, A, TOLERANCE);
        Checks.checkSVD2(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformTall() {
        MatrixD A = Matrices.randomUniformD(m, n, 3L);
        Checks.assertTall(A);
        MatrixD Q = getQ(A);
        MatrixD B = Checks.checkFactorization2(Q, A, TOLERANCE);
        Checks.checkSVD2(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformWide() {
        MatrixD A = Matrices.randomUniformD(n, m, 4L);
        Checks.assertWide(A);
        MatrixD Q = getQ(A);
        MatrixD B = Checks.checkFactorization2(Q, A, TOLERANCE);
        Checks.checkSVD2(B, Q, A, TOLERANCE);
    }

    @Test
    public void testOrthonormalColumns() {
        Checks.assertOrthonormal(getQ(Matrices.randomNormalD(m, n, 5L)));
    }

    @Test
    public void testScaleInvariance() {
        // the natural numbers matrix has rank 2 no matter how it is scaled
        assertEquals(2, getQ(Matrices.naturalNumbersD(m, n)).numColumns());
        assertEquals(2, getQ(Matrices.naturalNumbersD(m, n).scaleInplace(1.0e8)).numColumns());
        // scaling down used to collapse the rank and eventually return null,
        // because the guard against a numerically zero vector was an absolute
        // comparison against the machine epsilon while the stopping criterion
        // is relative to ||A||_F. Measured, the rank is now recovered all the
        // way down to the scale at which the entries of A themselves turn
        // denormal, which for this matrix is below a factor of about 1.0e-308
        assertEquals(2, getQ(Matrices.naturalNumbersD(m, n).scaleInplace(1.0e-8)).numColumns());
        assertEquals(2, getQ(Matrices.naturalNumbersD(m, n).scaleInplace(1.0e-100)).numColumns());
        assertEquals(2, getQ(Matrices.naturalNumbersD(m, n).scaleInplace(1.0e-200)).numColumns());
    }

    @Test
    public void testRankIsNotOvershot() {
        // the loop test used to run on the norms of the unprojected test
        // vectors, which spent one column too many whenever a single column
        // already sufficed. Measured, the residual after the first column of a
        // rank 1 matrix is around 1.0e-14 while the bound is around 5.0e-3, so
        // the column count is a deterministic decision with a wide margin
        for (int rank = 1; rank <= 8; ++rank) {
            MatrixD A = Matrices.randomNormalD(60, rank, rank)
                    .times(Matrices.randomNormalD(rank, 40, rank + 1000L));
            assertEquals("exact rank " + rank + " input", rank, getQ(A).numColumns());
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void testZeroMatrixRejected() {
        // the range of the zero matrix is the zero subspace, which cannot be
        // represented: jamu has no matrix with zero columns
        new AdaRangeFinder(Matrices.createD(m, n), AdaRangeFinder.DEFAULT_EPSILON, SEED);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNaNMatrixRejected() {
        MatrixD A = Matrices.randomNormalD(m, n, 8L);
        A.set(3, 3, Double.NaN);
        new AdaRangeFinder(A, AdaRangeFinder.DEFAULT_EPSILON, SEED);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInfiniteMatrixRejected() {
        MatrixD A = Matrices.randomNormalD(m, n, 9L);
        A.set(3, 3, Double.POSITIVE_INFINITY);
        new AdaRangeFinder(A, AdaRangeFinder.DEFAULT_EPSILON, SEED);
    }

    @Test(expected = ArithmeticException.class)
    public void testUnderflowingMatrixThrows() {
        // ||A||_F is the smallest positive double, so this matrix passes the
        // constructor, but every product A * omega then underflows to zero
        MatrixD A = Matrices.createD(m, n);
        A.set(7, 3, Double.MIN_VALUE);
        new AdaRangeFinder(A, AdaRangeFinder.DEFAULT_EPSILON, SEED).computeQ();
    }

    @Test
    public void testColumnCap() {
        // an epsilon this small can never be met, so only the cap can stop the
        // iteration
        MatrixD Q = new AdaRangeFinder(Matrices.randomNormalD(m, n, 6L), 1.0e-15, SEED).computeQ();
        assertTrue(Q.numColumns() <= Math.min(m, n));
    }

    /**
     * Tolerance for the reproducibility tests.
     * <p>
     * A seeded run repeats the same sequence of test vectors, but the result is
     * not bit-identical: the underlying BLAS is not run-to-run reproducible.
     * Measured, {@code A.times(v)} alone returns a different last bit in about
     * half of all repetitions on identical inputs, because a freshly allocated
     * output lands at a different memory alignment and MKL then dispatches a
     * different vectorized kernel, which sums in a different order. The
     * observed spread over a whole {@code computeQ()} is around 1.5e-14, so
     * this tolerance leaves two orders of headroom while still being far below
     * the O(1) difference between two genuinely different bases.
     */
    private static final double REPRODUCIBILITY_TOLERANCE = 1.0e-12;

    @Test
    public void testSameSeedGivesSameResult() {
        MatrixD A = Matrices.randomNormalD(m, n, 11L);
        MatrixD Q1 = new AdaRangeFinder(A, AdaRangeFinder.DEFAULT_EPSILON, 99L).computeQ();
        MatrixD Q2 = new AdaRangeFinder(A, AdaRangeFinder.DEFAULT_EPSILON, 99L).computeQ();
        // a broken seed sequence would draw the same test vector every time and
        // collapse the basis, which this catches before the elementwise check
        assertEquals(Math.min(m, n), Q1.numColumns());
        assertEquals(Q1.numColumns(), Q2.numColumns());
        assertArrayEquals(Q1.getArrayUnsafe(), Q2.getArrayUnsafe(), REPRODUCIBILITY_TOLERANCE);
    }

    @Test
    public void testDifferentSeedsGiveDifferentResults() {
        MatrixD A = Matrices.randomNormalD(m, n, 11L);
        MatrixD Q1 = new AdaRangeFinder(A, AdaRangeFinder.DEFAULT_EPSILON, 99L).computeQ();
        MatrixD Q2 = new AdaRangeFinder(A, AdaRangeFinder.DEFAULT_EPSILON, 100L).computeQ();
        double maxDiff = 0.0;
        double[] a = Q1.getArrayUnsafe();
        double[] b = Q2.getArrayUnsafe();
        for (int i = 0; i < a.length; ++i) {
            maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
        }
        // two different seeds span the same subspace but with different basis
        // vectors, so they must disagree by far more than floating point noise
        assertTrue("different seeds should produce a different basis, but the largest difference was " + maxDiff,
                maxDiff > REPRODUCIBILITY_TOLERANCE);
    }

    @Test
    public void testEpsilonDefaultMatchesExplicitValue() {
        MatrixD A = Matrices.naturalNumbersD(m, n);
        int kDefault = new AdaRangeFinder(A).computeQ().numColumns();
        int kExplicit = new AdaRangeFinder(A, AdaRangeFinder.DEFAULT_EPSILON).computeQ().numColumns();
        assertEquals(kDefault, kExplicit);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testEpsilonZero() {
        new AdaRangeFinder(Matrices.randomNormalD(m, n), 0.0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testEpsilonNegative() {
        new AdaRangeFinder(Matrices.randomNormalD(m, n), -1.0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testEpsilonTooLarge() {
        new AdaRangeFinder(Matrices.randomNormalD(m, n), 1.5);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testEpsilonNaN() {
        new AdaRangeFinder(Matrices.randomNormalD(m, n), Double.NaN);
    }

    private MatrixD getQ(MatrixD A) {
        // seeded so that a failure can be reproduced
        return new AdaRangeFinder(A, AdaRangeFinder.DEFAULT_EPSILON, SEED).computeQ();
    }
}
