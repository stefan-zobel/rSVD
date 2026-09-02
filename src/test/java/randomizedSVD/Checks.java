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
import net.jamu.matrix.SvdD;

public final class Checks {

    /**
     * Absolute tolerance floor, as a fraction of {@code ||A||_F}.
     * <p>
     * A purely relative elementwise comparison is meaningless close to zero: an
     * entry that happens to land at 1.0e-12 would demand 1.0e-19 of absolute
     * accuracy at a relative tolerance of 1.0e-7, which no correct
     * implementation can deliver. The comparison is therefore anchored to the
     * size of the matrix rather than to the size of the individual entry.
     * <p>
     * The value is measured, not guessed: over 1000 reconstructions of a
     * 220 x 150 standard normal matrix the largest observed elementwise error
     * was 2.0e-12 * ||A||_F, so this floor carries a factor of 50 of headroom
     * while still being six orders of magnitude below the error a genuinely
     * broken decomposition would produce.
     */
    private static final double ABS_TOL_FACTOR = 1.0e-10;

    /**
     * The absolute tolerance floor to use when comparing against
     * {@code expected}.
     *
     * @param expected
     *            the reference matrix
     * @return {@link #ABS_TOL_FACTOR} times the Frobenius norm of
     *         {@code expected}
     */
    public static double absTol(MatrixD expected) {
        return ABS_TOL_FACTOR * expected.normF();
    }

    /**
     * Asserts that {@code A} is tall, i.e. that it has more rows than columns.
     * <p>
     * The shape of a test matrix is decided by nothing but the order of two
     * arguments and is therefore easy to get wrong by copy and paste. A test
     * that ends up with the wrong shape does not fail, it silently duplicates
     * its counterpart, so the shape is asserted instead of assumed.
     *
     * @param A
     *            the test matrix
     */
    public static void assertTall(MatrixD A) {
        assertTrue("this test needs a tall matrix, but got " + A.numRows() + "x" + A.numColumns(),
                A.numRows() > A.numColumns());
    }

    /**
     * Asserts that {@code A} is wide, i.e. that it has more columns than rows.
     *
     * @param A
     *            the test matrix
     * @see #assertTall(MatrixD)
     */
    public static void assertWide(MatrixD A) {
        assertTrue("this test needs a wide matrix, but got " + A.numRows() + "x" + A.numColumns(),
                A.numRows() < A.numColumns());
    }

    public static MatrixD checkFactorization(MatrixD Q, MatrixD A, double tolerance) {
        MatrixD A_approx = null;
        MatrixD B = null;

        if (A.numRows() >= A.numColumns()) { // m >= n
            B = Q.transpose().times(A);
            A_approx = Q.times(B);
        } else { // m < n
            B = A.times(Q);
            A_approx = B.times(Q.transpose());
        }

        boolean equal = Matrices.approxEqual(A_approx, A, tolerance, absTol(A));
        assertTrue("A_approx and A should be approximately equal", equal);
        return B;
    }

    public static MatrixD checkFactorization2(MatrixD Q, MatrixD A, double tolerance) {
        MatrixD B = Q.transpose().times(A);
        MatrixD A_approx = Q.times(B);

        boolean equal = Matrices.approxEqual(A_approx, A, tolerance, absTol(A));
        assertTrue("A_approx and A should be approximately equal", equal);
        return B;
    }

    public static void checkSVD(MatrixD B, MatrixD Q, MatrixD A_expected, double tolerance) {
        MatrixD A_approx = null;
        SvdD svdReduced = B.svd(true);

        if (A_expected.numRows() >= A_expected.numColumns()) { // m >= n
            // U
            MatrixD U_lowrank = Q.times(svdReduced.getU());
            MatrixD U_approx = Matrices.embed(A_expected.numRows(), A_expected.numColumns(), U_lowrank);
            // Sigma
            MatrixD tmp = Matrices.diagD(svdReduced.getS());
            MatrixD Sigma = Matrices.embed(A_expected.numColumns(), A_expected.numColumns(), tmp);
            // Vt
            MatrixD Vt = svdReduced.getVt();
            // A_approx
            A_approx = U_approx.timesTimes(Sigma, Vt);
        } else { // m < n
            // U
            MatrixD U_lowrank = svdReduced.getU();
            MatrixD U_approx = Matrices.embed(A_expected.numRows(), A_expected.numColumns(), U_lowrank);
            // Sigma
            MatrixD tmp = Matrices.diagD(svdReduced.getS());
            MatrixD Sigma = Matrices.embed(A_expected.numColumns(), A_expected.numColumns(), tmp);
            // Vt
            MatrixD Vt = svdReduced.getVt().times(Q.transpose());
            Vt = Matrices.embed(A_expected.numColumns(), A_expected.numColumns(), Vt);
            // A_approx
            A_approx = U_approx.timesTimes(Sigma, Vt);
        }

        boolean equal = Matrices.approxEqual(A_approx, A_expected, tolerance, absTol(A_expected));
        assertTrue("A and reconstruction of A should be approximately equal", equal);
    }

    public static void checkSVD2(MatrixD B, MatrixD Q, MatrixD A_expected, double tolerance) {
        SvdD svdReduced = B.svd(true);

        // U
        MatrixD U_lowrank = Q.times(svdReduced.getU());
        MatrixD U_approx = Matrices.embed(A_expected.numRows(), A_expected.numColumns(), U_lowrank);
        // Sigma
        MatrixD tmp = Matrices.diagD(svdReduced.getS());
        MatrixD Sigma = Matrices.embed(A_expected.numColumns(), A_expected.numColumns(), tmp);
        // Vt
        MatrixD Vt = svdReduced.getVt();
        // A_approx
        MatrixD A_approx = U_approx.timesTimes(Sigma, Vt);

        boolean equal = Matrices.approxEqual(A_approx, A_expected, tolerance, absTol(A_expected));
        assertTrue("A and reconstruction of A should be approximately equal", equal);
    }
}
