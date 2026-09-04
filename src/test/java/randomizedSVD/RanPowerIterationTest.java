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

/**
 * Algorithm 4.3 has very poor accuracy.
 */
public class RanPowerIterationTest {

    private static final int m = 220;
    private static final int n = 150;
    // this one must have a very generous tolerance and hand-picked q values to
    // pass
    private static final double TOLERANCE = 1.0e-2;
    /**
     * Absolute tolerance floor for this class, as a fraction of
     * {@code ||A||_F}.
     * <p>
     * Algorithm 4.3 raises the ratio {@code sigma_1 / sigma_min} of the input
     * to the power {@code 2 * q + 1}, so its accuracy depends strongly on the
     * spectrum of the test matrix. The uniform cases are the demanding ones:
     * {@code Matrices.randomUniformD} draws from {@code [0, 1)}, so the matrix
     * carries a dominant rank one component - its mean - and reaches a ratio
     * around 114, against about 10 for a standard normal matrix.
     * <p>
     * Measured over 900 decompositions of uniform matrices at {@code q = 2},
     * the largest elementwise error of the reconstruction was
     * {@code 1.7e-9 * ||A||_F} with a median of {@code 5.6e-10 * ||A||_F},
     * against at most {@code 9e-12 * ||A||_F} for the natural number and
     * standard normal cases. The default floor of {@code Checks.absTol} is
     * calibrated for the accurate algorithms and sits at {@code 1e-10}, which
     * this class exceeds in about 2.5 % of all runs. This floor carries a
     * factor of about 6 of headroom over the measured worst case and leaves
     * the strictest assertion at 6 % of its budget, while the class already
     * accepts a relative tolerance of one percent.
     */
    private static final double ABS_TOL_FACTOR = 1.0e-8;
    /**
     * Seed for the test matrix of the algorithm, so that a failure can be
     * reproduced.
     * <p>
     * Unlike {@code AdaRangeFinderTest} this class has no
     * {@code testSameSeedGivesSameResult}: for this algorithm such a test
     * cannot say anything. Two runs with the same seed return bases that
     * differ by 0.5 in single entries, because the surplus QR columns beyond
     * the rank of the input are round-off, and their reconstructions agree to
     * about 1e-9 * ||A||_F - which two <em>different</em> seeds do as well,
     * since both approximate the same matrix. The seed pins the input of the
     * algorithm, which is what makes a red run reproducible; it does not pin
     * the basis.
     */
    private static final long SEED = 7L;

    @Test
    public void testNaturalNumbersTall() {
        int q = 1;
        // this is really low rank
        int estimatedRank = 2;
        MatrixD A = Matrices.naturalNumbersD(m, n);
        Checks.assertTall(A);
        MatrixD Q = getQ(A, estimatedRank, q);
        Checks.assertOrthonormal(Q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE, ABS_TOL_FACTOR);
        Checks.checkSVD(B, Q, A, TOLERANCE, ABS_TOL_FACTOR);
    }

    @Test
    public void testNaturalNumbersWide() {
        int q = 1;
        // this is really low rank
        int estimatedRank = 2;
        MatrixD A = Matrices.naturalNumbersD(n, m);
        Checks.assertWide(A);
        MatrixD Q = getQ(A, estimatedRank, q);
        Checks.assertOrthonormal(Q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE, ABS_TOL_FACTOR);
        Checks.checkSVD(B, Q, A, TOLERANCE, ABS_TOL_FACTOR);
    }

    @Test
    public void testRandomNormalTall() {
        int q = 3;
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomNormalD(m, n, 1L);
        Checks.assertTall(A);
        MatrixD Q = getQ(A, estimatedRank, q);
        Checks.assertOrthonormal(Q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE, ABS_TOL_FACTOR);
        Checks.checkSVD(B, Q, A, TOLERANCE, ABS_TOL_FACTOR);
    }

    @Test
    public void testRandomNormalWide() {
        int q = 3;
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomNormalD(n, m, 2L);
        Checks.assertWide(A);
        MatrixD Q = getQ(A, estimatedRank, q);
        Checks.assertOrthonormal(Q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE, ABS_TOL_FACTOR);
        Checks.checkSVD(B, Q, A, TOLERANCE, ABS_TOL_FACTOR);
    }

    @Test
    public void testRandomUniformTall() {
        int q = 2;
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomUniformD(m, n, 3L);
        Checks.assertTall(A);
        MatrixD Q = getQ(A, estimatedRank, q);
        Checks.assertOrthonormal(Q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE, ABS_TOL_FACTOR);
        Checks.checkSVD(B, Q, A, TOLERANCE, ABS_TOL_FACTOR);
    }

    @Test
    public void testRandomUniformWide() {
        int q = 2;
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomUniformD(n, m, 4L);
        Checks.assertWide(A);
        MatrixD Q = getQ(A, estimatedRank, q);
        Checks.assertOrthonormal(Q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE, ABS_TOL_FACTOR);
        Checks.checkSVD(B, Q, A, TOLERANCE, ABS_TOL_FACTOR);
    }

    @Test
    public void testNearlySquare() {
        // the sketch width used to exceed the row count of Y here, which the QR
        // cannot factor. The 220 x 150 shapes above are just far enough apart
        // to hide that: 150 + 10 is still below 220
        for (int[] shape : new int[][] { { 150, 145 }, { 145, 150 }, { 150, 150 } }) {
            int rows = shape[0];
            int cols = shape[1];
            MatrixD A = Matrices.randomNormalD(rows, cols, rows + cols);
            MatrixD Q = getQ(A, Math.min(rows, cols), 2);
            Checks.assertOrthonormal(Q);
            assertTrue("Q has " + Q.numColumns() + " columns for a " + rows + "x" + cols + " matrix",
                    Q.numColumns() <= Math.max(rows, cols));
            Checks.checkFactorization(Q, A, TOLERANCE, ABS_TOL_FACTOR);
        }
    }

    @Test
    public void testEstimatedRankTooLarge() {
        // a rank beyond min(rows, columns) cannot exist, so it is capped rather
        // than allowed to blow up the sketch
        MatrixD A = Matrices.randomNormalD(60, 40, 12L);
        MatrixD Q = getQ(A, 1000, 2);
        Checks.assertOrthonormal(Q);
        assertTrue("Q has " + Q.numColumns() + " columns", Q.numColumns() <= 60);
        Checks.checkFactorization(Q, A, TOLERANCE, ABS_TOL_FACTOR);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testEstimatedRankZero() {
        // a rank 0 subspace cannot be represented: jamu has no matrix with zero
        // columns, so asking for it is a caller error rather than something to
        // silently turn into a basis of P columns
        new RanPowerIteration(Matrices.randomNormalD(60, 40, 1L), 0, 2, SEED);
    }

    private MatrixD getQ(MatrixD A, int estimatedRank, int q) {
        return new RanPowerIteration(A, estimatedRank, q, SEED).computeQ();
    }
}
