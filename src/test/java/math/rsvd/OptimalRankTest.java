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

public class OptimalRankTest {

    private static final int m = 200;
    private static final int n = 120;
    // seeded so that a failure can be reproduced
    private static final long SEED = 7L;

    /** an exactly rank r matrix, the same construction as in SVDTest */
    private static MatrixD lowRank(int rows, int cols, int rank, long seed) {
        return Matrices.randomNormalD(rows, rank, seed).times(Matrices.randomNormalD(rank, cols, seed + 1000L));
    }

    private static MatrixD noise(int rows, int cols, double level, long seed) {
        return Matrices.randomNormalD(rows, cols, seed).scaleInplace(level);
    }

    /** a rank r signal buried in white noise, the model the threshold assumes */
    private static MatrixD signalPlusNoise(int rows, int cols, int rank, double level, long seed) {
        return lowRank(rows, cols, rank, seed).addInplace(1.0, noise(rows, cols, level, seed + 2000L));
    }

    private static SVD decompose(MatrixD A, int estimatedRank) {
        return new ApproximateBasis(A, estimatedRank, SEED).computeSVD();
    }

    /**
     * A decomposition of the given shape carrying exactly these singular
     * values. Only the shape and the values matter to the threshold, but the
     * factors are made properly orthonormal so that the object is a decomposition
     * of something.
     */
    private static SVD withSpectrum(int rows, int cols, double[] singularValues) {
        int k = singularValues.length;
        MatrixD U = Matrices.randomNormalD(rows, k, 11L).qrd().getQ();
        MatrixD V = Matrices.randomNormalD(cols, k, 12L).qrd().getQ();
        return new SVD(U, singularValues, V.transpose());
    }

    /**
     * Recovers {@code lambda(beta)} from the public behavior: for a single
     * singular value of 1 the count flips from 1 to 0 exactly when the
     * threshold reaches 1, that is at
     * {@code noiseLevel = 1 / (lambda * sqrt(max(rows, cols)))}.
     */
    private static double lambdaOf(int rows, int cols) {
        SVD svd = withSpectrum(rows, cols, new double[] { 1.0 });
        double lo = 0.0;
        double hi = 1.0;
        while (svd.optimalRank(hi) != 0) {
            hi *= 2.0;
        }
        for (int i = 0; i < 100; ++i) {
            double mid = 0.5 * (lo + hi);
            if (svd.optimalRank(mid) == 0) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        return 1.0 / (hi * Math.sqrt(Math.max(rows, cols)));
    }

    @Test
    public void testLambdaAtASquareShape() {
        // at beta = 1 the expression collapses to a closed form,
        // sqrt(2 * 2 + 8 / (2 + sqrt(16))) = sqrt(16 / 3), which is an
        // independent check on the formula rather than a copy of it
        assertEquals(Math.sqrt(16.0 / 3.0), lambdaOf(150, 150), 1.0e-12);
    }

    @Test
    public void testLambdaAtOtherShapes() {
        // as beta goes to 0 the expression tends to sqrt(2), and it grows
        // monotonically with beta up to the square case
        double square = Math.sqrt(16.0 / 3.0);
        double previous = Math.sqrt(2.0);
        int[][] shapes = { { 2000, 20 }, { 1000, 60 }, { 400, 100 }, { 200, 120 }, { 300, 260 }, { 150, 150 } };
        for (int[] shape : shapes) {
            double lambda = lambdaOf(shape[0], shape[1]);
            assertTrue(shape[0] + "x" + shape[1] + " gave " + lambda,
                    lambda > previous && lambda <= square + 1.0e-12);
            previous = lambda;
        }
        // and the published value at beta = 0.6
        assertEquals(2.0533062558327178, lambdaOf(200, 120), 1.0e-12);
    }

    @Test
    public void testThresholdOnAHandBuiltSpectrum() {
        // 200 x 120 at noise 1.0 puts the threshold at 29.0381..., between the
        // third and the fourth value
        double[] sigma = { 100.0, 50.0, 29.5, 28.5, 1.0 };
        assertEquals(3, withSpectrum(m, n, sigma).optimalRank(1.0));
        // at noise 1.5 the threshold is 43.557 and only the two large values
        // are left, at noise 2.0 it is 58.076 and only the largest survives
        assertEquals(2, withSpectrum(m, n, sigma).optimalRank(1.5));
        assertEquals(1, withSpectrum(m, n, sigma).optimalRank(2.0));
        // and at a noise level of zero everything above round-off survives
        assertEquals(5, withSpectrum(m, n, sigma).optimalRank(0.0));
    }

    @Test
    public void testTheRoundOffFloorIsNeverUndercut() {
        // at a noise level of zero the threshold would be zero as well, and
        // then a singular value at round-off level would count as signal. The
        // floor sits at max(rows, columns) * eps * sigma[0], here 2.22e-12
        double[] sigma = { 100.0, 50.0, 1.0e-16 };
        assertEquals(2, withSpectrum(m, n, sigma).optimalRank(0.0));
        // a value just above the floor does count
        double floor = Math.max(m, n) * 1.11022302462515654042e-16 * 100.0;
        assertEquals(3, withSpectrum(m, n, new double[] { 100.0, 50.0, 2.0 * floor }).optimalRank(0.0));
    }

    @Test
    public void testTheIterationStopsAtTheEdgeOfTheSpectrum() {
        // a signal that has sunk into the noise, where the count walks down one
        // value at a time and there is no clean edge to find. The iteration
        // stops once two consecutive counts are within one of each other;
        // without that rule it drifts on to 4. This pins the stopping rule, it
        // is not a claim that 5 is the true rank - the planted rank is 12 and
        // is not recoverable at this noise level
        MatrixD A = signalPlusNoise(78, 138, 12, 5.0, 5L);
        assertEquals(5, decompose(A, 36).optimalRank(A));
    }

    @Test
    public void testFindsThePlantedRank() {
        for (int rank : new int[] { 3, 5, 10, 20, 30 }) {
            for (double level : new double[] { 0.05, 0.2, 0.5, 1.0 }) {
                MatrixD A = signalPlusNoise(m, n, rank, level, 17L + rank);
                // the sketch is generous but nowhere near the full spectrum
                assertEquals("rank " + rank + " at noise " + level, rank, decompose(A, 3 * rank).optimalRank(A));
            }
        }
    }

    @Test
    public void testSuppliedNoiseLevelAgreesWithTheEstimate() {
        for (int rank : new int[] { 5, 20 }) {
            MatrixD A = signalPlusNoise(m, n, rank, 0.5, 17L + rank);
            SVD svd = decompose(A, 3 * rank);
            assertEquals("rank " + rank, svd.optimalRank(0.5), svd.optimalRank(A));
        }
    }

    @Test
    public void testScaleInvariance() {
        // the noise scales with the data, so the threshold has to as well. The
        // squaring inside the estimate is what threatens this at the ends of
        // the range, which is why it is scaled by the largest singular value
        MatrixD base = signalPlusNoise(m, n, 5, 0.5, 61L);
        for (double factor : new double[] { 1.0e100, 1.0e10, 1.0, 1.0e-10, 1.0e-100, 1.0e-200, 1.0e-300 }) {
            MatrixD A = base.copy().scaleInplace(factor);
            assertEquals("scaled by " + factor, 5, decompose(A, 30).optimalRank(A));
        }
    }

    @Test
    public void testNoiseFreeInputReportsItsRank() {
        // with no noise there is nothing for the threshold to cut, so it falls
        // back on the round-off floor. Without that floor this reports the full
        // width of the sketch
        for (int rank = 1; rank <= 8; ++rank) {
            MatrixD A = lowRank(m, n, rank, rank);
            assertEquals("exact rank " + rank, rank, decompose(A, 30).optimalRank(A));
        }
    }

    @Test
    public void testPureNoiseReportsZero() {
        for (double level : new double[] { 0.01, 0.7, 5.0 }) {
            MatrixD A = noise(m, n, level, 11L);
            assertEquals("noise level " + level, 0, decompose(A, 30).optimalRank(A));
        }
    }

    @Test
    public void testASketchOfTheFullWidthStillReportsARank() {
        // at r = min(rows, columns) the residual is empty by construction, the
        // estimated noise comes out as zero and everything would survive the
        // threshold. The count has to stay below that width
        MatrixD A = signalPlusNoise(120, 120, 10, 0.5, 81L);
        SVD svd = decompose(A, 120);
        assertEquals(120, svd.size());
        assertEquals(10, svd.optimalRank(A));
    }

    @Test
    public void testASingleRowOrColumn() {
        // no residual is available at all here, and the answer is the numerical
        // rank of a shape that cannot have more than rank 1
        MatrixD row = Matrices.randomNormalD(1, 40, 5L);
        assertEquals(1, decompose(row, 1).optimalRank(row));
        MatrixD column = Matrices.randomNormalD(40, 1, 6L);
        assertEquals(1, decompose(column, 1).optimalRank(column));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testShapeMismatchRejected() {
        MatrixD A = signalPlusNoise(m, n, 5, 0.5, 21L);
        decompose(A, 20).optimalRank(Matrices.randomNormalD(n, m, 22L));
    }

    @Test(expected = NullPointerException.class)
    public void testNullMatrixRejected() {
        decompose(signalPlusNoise(m, n, 5, 0.5, 23L), 20).optimalRank((MatrixD) null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNegativeNoiseLevelRejected() {
        withSpectrum(m, n, new double[] { 1.0 }).optimalRank(-1.0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNaNNoiseLevelRejected() {
        withSpectrum(m, n, new double[] { 1.0 }).optimalRank(Double.NaN);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInfiniteNoiseLevelRejected() {
        withSpectrum(m, n, new double[] { 1.0 }).optimalRank(Double.POSITIVE_INFINITY);
    }
}
