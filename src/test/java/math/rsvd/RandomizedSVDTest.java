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
import net.jamu.matrix.SvdD;

import org.junit.Test;

/**
 * The facade must not be a second implementation. Almost every case here is an
 * equivalence against the direct route, because that is the property that makes
 * a facade one.
 */
public class RandomizedSVDTest {

    // seeded so that a failure can be reproduced
    private static final long SEED = 7L;

    private static final int M = 220;
    private static final int N = 150;

    /**
     * A seeded run repeats the same test vectors, but not the same bits: the
     * BLAS underneath is not run-to-run reproducible, for the reasons set out in
     * {@code AdaRangeFinderTest}. Two orders of headroom over the spread
     * measured there, and still far below the O(1) difference between two
     * genuinely different results.
     */
    private static final double TOLERANCE = 1.0e-12;

    private static MatrixD signalPlusNoise(int rows, int cols, int rank, double level, long seed) {
        return Matrices.randomNormalD(rows, rank, seed).times(Matrices.randomNormalD(rank, cols, seed + 1000L))
                .addInplace(1.0, Matrices.randomNormalD(rows, cols, seed + 2000L).scaleInplace(level));
    }

    /**
     * Compares two decompositions by what a seeded run actually promises: the
     * width, the singular values, and the approximation they carry.
     * <p>
     * Deliberately <em>not</em> {@code U} and {@code Vt} entrywise. Where the
     * sketch is wider than the rank of the input, the surplus columns are
     * determined by round-off alone, so two runs with one and the same seed
     * return genuinely different bases of the same subspace - this test
     * asserted it anyway at first and failed on roughly one run in four, at a
     * width of 40 on an input of rank 9. The seeded constructor of
     * {@link ApproximateBasis} says exactly this much and no more: "the width of
     * the decomposition came out identical every time and the singular values
     * agreed to 3.5e-15 relative".
     */
    private static void assertSame(String what, SVD expected, SVD actual) {
        assertEquals(what + ": size", expected.size(), actual.size());
        double[] a = expected.getSingularValues();
        double[] b = actual.getSingularValues();
        for (int i = 0; i < expected.size(); ++i) {
            assertEquals(what + ": singular value " + i, a[i], b[i], TOLERANCE * a[0]);
        }
        MatrixD one = expected.reconstruct();
        MatrixD other = actual.reconstruct();
        double difference = one.copy().addInplace(-1.0, other).normF() / one.normF();
        assertTrue(what + ": the approximations differ by " + difference, difference <= TOLERANCE);
    }

    @Test
    public void testFixedWidthMatchesTheConstructor() {
        MatrixD A = signalPlusNoise(M, N, 9, 0.5, 11L);
        for (int rank : new int[] { 3, 12, 40 }) {
            assertSame("rank " + rank + " unseeded is at least the same size",
                    new ApproximateBasis(A, rank, SEED).computeSVD(),
                    RandomizedSVD.of(A).toRank(rank).seed(SEED).decompose());
        }
    }

    @Test
    public void testAutomaticMatchesTheStaticFactory() {
        MatrixD A = signalPlusNoise(M, N, 9, 0.5, 21L);
        assertSame("automatic", ApproximateBasis.decompose(A, SEED),
                RandomizedSVD.of(A).findingTheRank().seed(SEED).decompose());
    }

    @Test
    public void testTheBasisMatchesTheRangeFinder() {
        MatrixD A = signalPlusNoise(M, N, 9, 0.5, 31L);
        for (double epsilon : new double[] { 1.0e-3, 1.0e-1, 0.3 }) {
            MatrixD direct = new AdaRangeFinder(A, epsilon, SEED).computeQ();
            MatrixD viaFacade = RandomizedSVD.of(A).toAccuracy(epsilon).seed(SEED).basis();
            assertEquals("epsilon " + epsilon, direct.numColumns(), viaFacade.numColumns());
            assertTrue("epsilon " + epsilon,
                    Matrices.approxEqual(direct, viaFacade, TOLERANCE, Checks.absTol(direct)));
        }
    }

    @Test
    public void testTheBlockedBasisMatchesTheRangeFinder() {
        MatrixD A = signalPlusNoise(M, N, 9, 0.5, 41L);
        for (int block : new int[] { 1, 8, 16, 32 }) {
            MatrixD direct = new AdaRangeFinder(A, 1.0e-3, SEED).computeQ(block);
            MatrixD viaFacade = RandomizedSVD.of(A).toAccuracy(1.0e-3).blockSize(block).seed(SEED).basis();
            assertEquals("block " + block, direct.numColumns(), viaFacade.numColumns());
            assertTrue("block " + block,
                    Matrices.approxEqual(direct, viaFacade, TOLERANCE, Checks.absTol(direct)));
        }
    }

    @Test
    public void testWithoutABlockSizeItIsTheColumnByColumnPath() {
        // the facade must not quietly pick a block size: the width would grow
        // by up to 3.5 times it, and that width is felt again in decompose()
        MatrixD A = signalPlusNoise(M, N, 9, 0.5, 51L);
        assertEquals(new AdaRangeFinder(A, 0.3, SEED).computeQ().numColumns(),
                RandomizedSVD.of(A).toAccuracy(0.3).seed(SEED).basis().numColumns());
    }

    @Test
    public void testTheCompletionMatchesTheHandWrittenOne() {
        // the one piece of arithmetic the facade adds, against the way a caller
        // would have to write it today - the same lines LowRankBenchmark carries
        MatrixD A = signalPlusNoise(M, N, 9, 0.5, 61L);
        for (double epsilon : new double[] { 1.0e-3, 0.3 }) {
            MatrixD Q = new AdaRangeFinder(A, epsilon, SEED).computeQ();
            SvdD small = Q.transpose().times(A).svdEcon();
            SVD byHand = new SVD(Q.times(small.getU()), small.getS(), small.getVt());
            assertSame("epsilon " + epsilon, byHand,
                    RandomizedSVD.of(A).toAccuracy(epsilon).seed(SEED).decompose());
        }
    }

    @Test
    public void testTheAccuracyPathMeetsItsTarget() {
        MatrixD A = signalPlusNoise(M, N, 9, 0.5, 71L);
        for (double epsilon : new double[] { 1.0e-3, 1.0e-1, 0.3 }) {
            SVD svd = RandomizedSVD.of(A).toAccuracy(epsilon).blockSize(16).seed(SEED).decompose();
            Checks.assertOrthonormal(svd.getU());
            assertTrue("epsilon " + epsilon + " produced " + svd.size() + " columns",
                    svd.size() <= Math.min(M, N));
            double residual = A.copy().addInplace(-1.0, svd.reconstruct()).normF() / A.normF();
            if (svd.size() < Math.min(M, N)) {
                assertTrue("epsilon " + epsilon + ": residual " + residual, residual <= epsilon);
            }
        }
    }

    @Test
    public void testTheOrderOfTheSettingsDoesNotMatter() {
        MatrixD A = signalPlusNoise(M, N, 9, 0.5, 81L);
        assertSame("order", RandomizedSVD.of(A).toAccuracy(1.0e-3).seed(SEED).blockSize(16).decompose(),
                RandomizedSVD.of(A).toAccuracy(1.0e-3).blockSize(16).seed(SEED).decompose());
    }

    @Test
    public void testATerminalMayBeCalledTwice() {
        MatrixD A = signalPlusNoise(M, N, 9, 0.5, 91L);
        RandomizedSVD.Accuracy stage = RandomizedSVD.of(A).toAccuracy(1.0e-3).seed(SEED);
        assertSame("twice", stage.decompose(), stage.decompose());
        MatrixD first = stage.basis();
        MatrixD second = stage.basis();
        assertEquals(first.numColumns(), second.numColumns());
        assertTrue(Matrices.approxEqual(first, second, TOLERANCE, Checks.absTol(first)));
    }

    @Test
    public void testAnUnseededRunWorks() {
        MatrixD A = signalPlusNoise(M, N, 9, 0.5, 101L);
        assertTrue(RandomizedSVD.of(A).toAccuracy(1.0e-3).basis().numColumns() >= 1);
        assertTrue(RandomizedSVD.of(A).toAccuracy(1.0e-3).blockSize(8).decompose().size() >= 1);
        assertEquals(12, RandomizedSVD.of(A).toRank(12).decompose().size());
        assertEquals(9, RandomizedSVD.of(A).findingTheRank().decompose().size());
    }

    @Test
    public void testRankAndSeedAreNotConfusable() {
        // the reason this facade exists. Both take a plain 16 with no suffix,
        // and they cannot be mixed up, because they are named rather than
        // positional - which is exactly what a (MatrixD, double, int) next to a
        // (MatrixD, double, long) constructor could not have offered
        MatrixD A = signalPlusNoise(M, N, 9, 0.5, 111L);
        assertEquals(16, RandomizedSVD.of(A).toRank(16).seed(16).decompose().size());
        assertEquals(RandomizedSVD.of(A).toRank(16).seed(16L).decompose().size(),
                RandomizedSVD.of(A).toRank(16).seed(16).decompose().size());
    }

    @Test
    public void testTheFixedWidthBasisMatchesTheConstructor() {
        MatrixD A = signalPlusNoise(M, N, 9, 0.5, 161L);
        for (int rank : new int[] { 3, 12, 40 }) {
            MatrixD direct = new ApproximateBasis(A, rank, SEED).computeQ();
            MatrixD viaFacade = RandomizedSVD.of(A).toRank(rank).seed(SEED).basis();
            assertEquals("rank " + rank, direct.numColumns(), viaFacade.numColumns());
            assertEquals("rank " + rank, M, viaFacade.numRows());
            Checks.assertOrthonormal(viaFacade);
            assertTrue("rank " + rank,
                    Matrices.approxEqual(direct, viaFacade, TOLERANCE, Checks.absTol(direct)));
        }
    }

    @Test
    public void testTheAutomaticBasisIsTheLeftFactorOfTheDecomposition() {
        MatrixD A = signalPlusNoise(M, N, 9, 0.5, 171L);
        MatrixD direct = ApproximateBasis.decompose(A, SEED).getU();
        MatrixD viaFacade = RandomizedSVD.of(A).findingTheRank().seed(SEED).basis();
        assertEquals(direct.numColumns(), viaFacade.numColumns());
        assertEquals(M, viaFacade.numRows());
        Checks.assertOrthonormal(viaFacade);
        // on this path the basis is exactly as wide as the rank that was found
        assertEquals(RandomizedSVD.of(A).findingTheRank().seed(SEED).decompose().size(),
                viaFacade.numColumns());
    }

    @Test
    public void testNystroemComposesWithEveryPath() {
        MatrixD G = Matrices.randomNormalD(N, 8, 121L);
        MatrixD A = G.times(G.transpose());
        SVD byAccuracy = Nystroem.decompose(A, RandomizedSVD.of(A).toAccuracy(1.0e-3).seed(SEED).basis());
        SVD byRank = Nystroem.decompose(A, RandomizedSVD.of(A).toRank(12).seed(SEED).basis());
        SVD automatic = Nystroem.decompose(A, RandomizedSVD.of(A).findingTheRank().seed(SEED).basis());
        for (SVD evd : new SVD[] { byAccuracy, byRank, automatic }) {
            assertEquals(8, evd.size());
            assertTrue(Matrices.approxEqual(evd.reconstruct(), A, 1.0e-8, Checks.absTol(A)));
        }
    }

    @Test(expected = NullPointerException.class)
    public void testNullMatrixRejectedAtTheEntryPoint() {
        RandomizedSVD.of(null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidEpsilonIsPassedThrough() {
        RandomizedSVD.of(signalPlusNoise(60, 40, 4, 0.5, 131L)).toAccuracy(0.0).basis();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidRankIsPassedThrough() {
        RandomizedSVD.of(signalPlusNoise(60, 40, 4, 0.5, 141L)).toRank(0).decompose();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvalidBlockSizeIsPassedThrough() {
        RandomizedSVD.of(signalPlusNoise(60, 40, 4, 0.5, 151L)).toAccuracy(1.0e-3).blockSize(0).basis();
    }
}
