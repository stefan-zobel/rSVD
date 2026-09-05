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
 * Algorithm 5.5, the Nystroem postprocessing for a positive semidefinite input.
 */
public class NystroemTest {

    // seeded so that a failure can be reproduced
    private static final long SEED = 7L;

    private static final int N = 120;

    /** a positive semidefinite matrix of exactly the given rank */
    private static MatrixD lowRankPsd(int n, int rank, long seed) {
        MatrixD G = Matrices.randomNormalD(n, rank, seed);
        return G.times(G.transpose());
    }

    /** a positive semidefinite matrix with a prescribed spectrum */
    private static MatrixD psd(int n, double[] eigenvalues, long seed) {
        MatrixD V = Matrices.randomNormalD(n, n, seed).qrd().getQ();
        MatrixD A = V.times(Matrices.diagD(n, n, eigenvalues)).times(V.transpose());
        // the construction is symmetric in exact arithmetic, and this makes it
        // symmetric in floating point too, so that the tests measure the method
        // rather than the round-off of their own setup
        return A.addInplace(1.0, A.transpose()).scaleInplace(0.5);
    }

    private static double[] decaying(int n, double rate) {
        double[] e = new double[n];
        for (int i = 0; i < n; ++i) {
            e[i] = Math.exp(-rate * i);
        }
        return e;
    }

    /** a Gaussian sketch plus QR, the plain Stage A of the paper */
    private static MatrixD sketch(MatrixD A, int width, long seed) {
        return A.times(Matrices.randomNormalD(A.numColumns(), width, seed)).qrd().getQ();
    }

    private static double spectralError(MatrixD A, MatrixD approximation) {
        return A.copy().addInplace(-1.0, approximation).norm2() / A.norm2();
    }

    @Test
    public void testTheErrorNeverExceedsTheStageABound() {
        // the one statement the paper proves: "in the spectral norm, the
        // Nystroem approximation error never exceeds ||A - QQ'A||". Everything
        // else about this method is a matter of degree, this is an inequality.
        // Measured over 360 cases above the round-off floor it held every time,
        // with a median of 0.73 and a best of 0.025, so the slack below is for
        // round-off and not for the claim
        for (double rate : new double[] { 0.5, 0.1, 0.02 }) {
            MatrixD A = psd(N, decaying(N, rate), 11L);
            for (int width : new int[] { 5, 10, 20, 40 }) {
                MatrixD Q = sketch(A, width, 31L + width);
                double stageA = spectralError(A, Q.times(Q.transpose().times(A)));
                double nystroem = spectralError(A, Nystroem.decompose(A, Q).reconstruct());
                assertTrue("rate " + rate + " width " + width + ": " + nystroem + " against " + stageA,
                        nystroem <= stageA * 1.001 + 1.0e-12);
            }
        }
    }

    @Test
    public void testExactlyLowRankInputIsReconstructed() {
        for (int rank : new int[] { 1, 5, 20 }) {
            MatrixD A = lowRankPsd(N, rank, 41L + rank);
            MatrixD Q = sketch(A, rank, SEED);
            SVD e = Nystroem.decompose(A, Q);
            assertTrue("rank " + rank,
                    Matrices.approxEqual(e.reconstruct(), A, 1.0e-8, Checks.absTol(A)));
        }
    }

    @Test
    public void testTheOvershootOfTheBasisIsTrimmed() {
        // a basis wider than the numerical rank makes Q'AQ singular, which is
        // where a Cholesky factorization - step 2 of the paper as written -
        // fails. Measured, dpotrf failed in every one of these cases while the
        // inverse square root used here runs through and returns the rank
        for (int rank : new int[] { 5, 12 }) {
            MatrixD A = lowRankPsd(N, rank, 51L + rank);
            for (int width : new int[] { rank, rank + 4, 2 * rank, 4 * rank }) {
                SVD e = Nystroem.decompose(A, sketch(A, width, SEED));
                assertEquals("rank " + rank + " from a basis of " + width, rank, e.size());
                assertTrue(Matrices.approxEqual(e.reconstruct(), A, 1.0e-8, Checks.absTol(A)));
            }
        }
    }

    @Test
    public void testTheEigenvaluesAreNonnegativeAndDescending() {
        MatrixD A = psd(N, decaying(N, 0.2), 61L);
        double[] lambda = Nystroem.decompose(A, sketch(A, 30, SEED)).getSingularValues();
        for (int i = 0; i < lambda.length; ++i) {
            assertTrue("eigenvalue " + i + " is " + lambda[i], lambda[i] >= 0.0);
            if (i > 0) {
                assertTrue("eigenvalue " + i + " is above its predecessor", lambda[i] <= lambda[i - 1]);
            }
        }
    }

    @Test
    public void testUIsOrthonormal() {
        for (int width : new int[] { 5, 20, 50 }) {
            MatrixD A = psd(N, decaying(N, 0.15), 71L);
            SVD e = Nystroem.decompose(A, sketch(A, width, SEED));
            Checks.assertOrthonormal(e.getU());
            // for a positive semidefinite matrix the decomposition is its own
            // eigendecomposition, so the right factor is the transpose of the
            // left one rather than an independent matrix
            assertTrue(Matrices.approxEqual(e.getVt(), e.getU().transpose(), 1.0e-12, 1.0e-12));
        }
    }

    @Test
    public void testAgainstTheExactEigenvalues() {
        // a basis that spans the whole range must recover the eigenvalues of A
        int rank = 8;
        MatrixD A = lowRankPsd(N, rank, 81L);
        double[] exact = A.svdEcon().getS();
        double[] found = Nystroem.decompose(A, sketch(A, rank + 6, SEED)).getSingularValues();
        assertEquals(rank, found.length);
        for (int i = 0; i < rank; ++i) {
            assertEquals("eigenvalue " + i, exact[i], found[i], 1.0e-8 * exact[0]);
        }
    }

    @Test
    public void testScaleInvariance() {
        MatrixD A = lowRankPsd(N, 6, 91L);
        MatrixD Q = sketch(A, 10, SEED);
        double[] reference = Nystroem.decompose(A, Q).getSingularValues();
        for (double scale : new double[] { 1.0e+100, 1.0e+10, 1.0e-10, 1.0e-100, 1.0e-300 }) {
            MatrixD scaled = A.copy().scaleInplace(scale);
            double[] found = Nystroem.decompose(scaled, sketch(scaled, 10, SEED)).getSingularValues();
            assertEquals("scale " + scale, reference.length, found.length);
            for (int i = 0; i < found.length; ++i) {
                // the eigenvalues scale with A, so they are compared as ratios
                assertEquals("scale " + scale + ", eigenvalue " + i, reference[i] / reference[0],
                        found[i] / found[0], 1.0e-8);
            }
        }
    }

    @Test
    public void testANearlySymmetricMatrixIsAccepted() {
        // the symmetry check guards against a matrix that is not symmetric at
        // all, not against round-off. Measured, a matrix built as G * G' comes
        // out exactly symmetric in nearly every case and never worse than
        // 7.7e-17 relative, so a perturbation well above that must still pass
        MatrixD A = lowRankPsd(N, 10, 101L);
        MatrixD noise = Matrices.randomNormalD(N, N, 102L).scaleInplace(1.0e-13 * A.normF() / N);
        MatrixD perturbed = A.copy().addInplace(1.0, noise);
        assertTrue(Nystroem.decompose(perturbed, sketch(perturbed, 14, SEED)).size() >= 1);
    }

    @Test(expected = NullPointerException.class)
    public void testNullMatrixRejected() {
        Nystroem.decompose(null, Matrices.randomNormalD(10, 3, 1L));
    }

    @Test(expected = NullPointerException.class)
    public void testNullBasisRejected() {
        Nystroem.decompose(lowRankPsd(20, 3, 1L), null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNonSquareMatrixRejected() {
        Nystroem.decompose(Matrices.randomNormalD(30, 20, 1L), Matrices.randomNormalD(30, 5, 2L));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBasisWithWrongRowCountRejected() {
        Nystroem.decompose(lowRankPsd(30, 4, 1L), Matrices.randomNormalD(20, 5, 2L));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBasisWiderThanTheMatrixRejected() {
        Nystroem.decompose(lowRankPsd(20, 4, 1L), Matrices.randomNormalD(20, 25, 2L));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testZeroMatrixRejected() {
        Nystroem.decompose(Matrices.createD(20, 20), Matrices.randomNormalD(20, 5, 2L));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNaNEntryRejected() {
        MatrixD A = lowRankPsd(20, 4, 1L);
        A.set(3, 3, Double.NaN);
        Nystroem.decompose(A, Matrices.randomNormalD(20, 5, 2L));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testAsymmetricMatrixRejected() {
        Nystroem.decompose(Matrices.randomNormalD(40, 40, 111L), Matrices.randomNormalD(40, 8, 112L));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testIndefiniteMatrixRejected() {
        // symmetric, but with two negative eigenvalues among the leading ones,
        // so that they cannot hide outside the range of the basis
        double[] eigenvalues = new double[40];
        for (int i = 0; i < 10; ++i) {
            eigenvalues[i] = 100.0 / (i + 1);
        }
        eigenvalues[10] = -50.0;
        eigenvalues[11] = -25.0;
        MatrixD A = psd(40, eigenvalues, 121L);
        Nystroem.decompose(A, sketch(A, 20, SEED));
    }
}
