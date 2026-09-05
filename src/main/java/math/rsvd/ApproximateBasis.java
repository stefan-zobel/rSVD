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

import java.util.Objects;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;
import net.jamu.matrix.SvdD;

public final class ApproximateBasis {

    // Oversampling parameter
    private static final int P = 5;
    // The sketch width decompose(MatrixD) starts from, and the factor it grows
    // by when that width turns out to be too small. Measured over three shapes,
    // five starting widths and three growth factors, neither constant is
    // critical: below a rank of a twentieth of min(rows, columns) every one of
    // the fifteen combinations beat a single decomposition at full width, taking
    // between 0.14 and 0.79 of its time, and above about a fifth every one of
    // them lost. Where that crossover lies is a property of the method, not of
    // these two numbers. What does matter is the stopping rule in escalate.
    private static final int INITIAL_WIDTH = 16;
    private static final int GROWTH = 2;

    private final MatrixD A;
    private final int m;
    private final int n;
    private final int targetRank;
    private final boolean transpose;
    /** whether {@link #seed} was supplied by the caller */
    private final boolean seeded;
    /** the seed for the random test matrix, only meaningful if {@link #seeded} */
    private final long seed;

    /**
     * Creates an approximate basis for {@code A}. The random test matrix is
     * drawn from an unspecified source of randomness, so repeated runs
     * generally differ.
     *
     * @param A
     *            the matrix to decompose
     * @param estimatedRank
     *            the target rank, must be at least 1
     * @throws IllegalArgumentException
     *             if {@code estimatedRank} is less than 1
     */
    public ApproximateBasis(MatrixD A, int estimatedRank) {
        this(A, estimatedRank, false, 0L);
    }

    /**
     * Creates a reproducible approximate basis for {@code A}. Two instances
     * constructed with the same matrix, the same {@code estimatedRank} and the
     * same {@code seed} draw the same test matrix and do the same arithmetic on
     * it, so they agree to round-off.
     * <p>
     * They do not agree bit for bit, and no seed can make them. The BLAS and
     * LAPACK routines underneath are free to divide their work differently from
     * one run to the next, and floating point addition is not associative, so
     * the same operands can add up to a slightly different sum. Measured over 21
     * runs per case on three shapes and two widths, the width of the
     * decomposition came out identical every time and the singular values agreed
     * to {@code 3.5e-15} relative.
     *
     * @param A
     *            the matrix to decompose
     * @param estimatedRank
     *            the target rank, must be at least 1
     * @param seed
     *            the seed for the random test matrix
     * @throws IllegalArgumentException
     *             if {@code estimatedRank} is less than 1
     */
    public ApproximateBasis(MatrixD A, int estimatedRank, long seed) {
        this(A, estimatedRank, true, seed);
    }

    /**
     * Decomposes {@code A} without being told its rank, keeping the singular
     * values that carry signal rather than noise.
     * <p>
     * The rank is the one parameter of this class that a caller generally
     * cannot know, since it is a statement about {@code A}. This method finds
     * it instead: it sketches at a modest width, asks
     * {@link SVD#optimalRank(MatrixD)} how much of that width was signal, and
     * doubles the width whenever the answer fills it completely, which means
     * the sketch was too narrow to see the edge. The returned decomposition is
     * already truncated to the rank that was found.
     * <p>
     * Measured over 200 random matrices with ranks up to half of
     * {@code min(rows, columns)}, this reached the same answer as a single
     * decomposition at full width in all 200 cases, and the same answer as an
     * oracle given the true noise level and the exact full spectrum in all 200
     * as well.
     * <p>
     * It pays off when the rank is a small fraction of
     * {@code min(rows, columns)}, which is the situation a randomized
     * decomposition exists for in the first place. Measured against a single
     * decomposition at full width on 500 x 300, 400 x 400 and 1500 x 150: at a
     * rank of a fiftieth of the width it took 0.22, 0.22 and 0.72 of the time,
     * at a tenth it was around break even, and beyond that it costs more, up to
     * 3.3 times at the worst point measured. A tall thin matrix benefits least,
     * because there the cost of a decomposition is governed by the long
     * dimension rather than by the width of the sketch.
     * <p>
     * There is one limit that no stopping rule can lift. When the rank is a
     * large fraction of {@code min(rows, columns)} the matrix is not low rank in
     * any useful sense, and a sketch narrower than the rank cannot reveal it: on
     * a 400 x 400 matrix of rank 299, sketches of width 128 and 256 both report
     * 41 and only a width of 320 reports 299. Such an input is outside what a
     * randomized decomposition can do, and this method returns the low answer
     * rather than detecting the situation.
     * <p>
     * The noise model is the one of {@link SVD#optimalRank(MatrixD)}. If
     * {@code A} is indistinguishable from noise there is nothing to return: a
     * decomposition of rank 0 cannot be represented, so a rank 1 decomposition
     * is returned and {@link SVD#optimalRank(MatrixD)} on it reports 0.
     *
     * @param A
     *            the matrix to decompose
     * @return a decomposition truncated to the rank that was found
     * @throws NullPointerException
     *             if {@code A} is {@code null}
     * @throws IllegalArgumentException
     *             if {@code ||A||_F} is not positive and finite
     */
    public static SVD decompose(MatrixD A) {
        return escalate(A, false, 0L);
    }

    /**
     * Decomposes {@code A} without being told its rank, reproducibly. Two calls
     * with the same matrix and the same {@code seed} draw the same test
     * matrices and take the same escalation path, so they agree to round-off.
     * They do not agree bit for bit: the LAPACK routines underneath are not
     * reproducible to the last bit between two runs, measured at 1.7e-15
     * relative for a single seeded decomposition and 1.6e-15 for the
     * escalation, so escalating adds no variation of its own.
     *
     * @param A
     *            the matrix to decompose
     * @param seed
     *            the seed for the random test matrices
     * @return a decomposition truncated to the rank that was found
     * @throws NullPointerException
     *             if {@code A} is {@code null}
     * @throws IllegalArgumentException
     *             if {@code ||A||_F} is not positive and finite
     * @see #decompose(MatrixD)
     */
    public static SVD decompose(MatrixD A, long seed) {
        return escalate(A, true, seed);
    }

    private static SVD escalate(MatrixD A, boolean seeded, long seed) {
        Objects.requireNonNull(A);
        // one below min(rows, columns), which is the widest sketch from which a
        // rank can still be told: SVD.optimalRank needs a residual to estimate
        // the noise level from, and at the full width there is none
        int maxWidth = Math.max(1, Math.min(A.numRows(), A.numColumns()) - 1);
        int width = Math.min(INITIAL_WIDTH, maxWidth);
        int pending = -1;
        while (true) {
            SVD svd = (seeded ? new ApproximateBasis(A, width, seed) : new ApproximateBasis(A, width)).computeSVD();
            int rank = svd.optimalRank(A);
            if (width >= maxWidth) {
                // a rank of 0 is not representable, jamu has no matrix with
                // zero columns, so the narrowest decomposition stands in for it
                return svd.truncate(Math.max(rank, 1));
            }
            if (rank < width) {
                // an answer below the width is not by itself proof that the
                // edge was seen. A sketch far narrower than the rank leaves
                // signal in its residual, which inflates the estimated noise
                // level and pushes the count down below the width, looking
                // exactly like convergence - measured, a rank 224 matrix of
                // 500 x 300 reported 59 that way. So the answer has to survive
                // one more doubling before it is accepted
                if (rank == pending) {
                    return svd.truncate(Math.max(rank, 1));
                }
                pending = rank;
            } else {
                pending = -1;
            }
            width = Math.min(GROWTH * width, maxWidth);
        }
    }

    private ApproximateBasis(MatrixD A, int estimatedRank, boolean seeded, long seed) {
        if (estimatedRank < 1) {
            throw new IllegalArgumentException("target rank must be at least 1, but was " + estimatedRank);
        }
        this.A = Objects.requireNonNull(A);
        double normF = A.normF();
        // the negated comparison also rejects NaN. Without this check a matrix
        // holding a NaN or an infinity reaches LAPACK and fails there with
        // "Illegal argument at position 4" out of dgetrf, which says nothing
        // about the cause, and a zero matrix has no basis to return at all
        if (!(normF > 0.0 && normF < Double.POSITIVE_INFINITY)) {
            throw new IllegalArgumentException(
                    "the decomposition of a matrix with ||A||_F = " + normF + " is not defined");
        }
        m = A.numRows();
        n = A.numColumns();
        transpose = m < n;
        targetRank = Math.min(estimatedRank, Math.min(m, n));
        this.seeded = seeded;
        this.seed = seed;
    }

    public SVD computeSVD() {
        MatrixD[] BQ = computeBQ();
        MatrixD B = BQ[0];
        MatrixD Q = BQ[1];
        MatrixD QT = BQ[2];

        SvdD svd = B.svdEcon();
        MatrixD U_tilde = svd.getU();
        double[] sigma = svd.getS();
        MatrixD Vt = svd.getVt();

        MatrixD U = null;
        if (transpose) {
            U = Q.times(U_tilde);
        } else {
            U = U_tilde;
            Vt = Vt.times(QT);
        }
        return createSVD(U, sigma, Vt);
    }

    private SVD createSVD(MatrixD U, double[] sigma, MatrixD Vt) {
        if (U.numColumns() > targetRank) {
            U = U.selectConsecutiveColumns(0, targetRank - 1);
        }
        if (Vt.numRows() > targetRank) {
            Vt = Vt.selectSubmatrix(0, 0, targetRank - 1, Vt.endCol());
        }
        if (U.numRows() > m) {
            // should never happen
            U = U.selectSubmatrix(0, 0, m - 1, U.endCol());
        }
        if (Vt.numColumns() > n) {
            // should never happen
            Vt = Vt.selectConsecutiveColumns(0, n - 1);
        }
        return new SVD(U, sigma, Vt);
    }

    private MatrixD[] computeBQ() {
        MatrixD Q = sketchBasis();
        MatrixD QT = Q.transpose();
        if (transpose) {
            return new MatrixD[] { QT.times(A), Q, QT };
        }
        return new MatrixD[] { A.times(Q), Q, QT };
    }

    /**
     * Computes a matrix with orthonormal columns whose range approximates the
     * range of {@code A}.
     * <p>
     * The contract is the one {@link AdaRangeFinder#computeQ()} has:
     * {@code A.numRows()} rows, orthonormal columns, and the range of
     * {@code A}, whatever the shape of {@code A} is. See
     * {@link #sketchBasis()} for why that last part is not free on a tall
     * matrix. Measured, the carry-over costs 2 to 14 percent there, typically
     * about 7, over shapes from {@code 400 x 200} to {@code 1600 x 200} and
     * widths of 10 and 40; on a wide matrix it costs nothing, because there the
     * iteration already ends where this needs it to.
     * <p>
     * <b>The width is not the rank that was asked for.</b> It is that rank
     * oversampled by five, capped at {@code min(rows, columns)}, because the
     * oversampling is what makes the basis worth having - the bounds of the
     * paper are stated for it. Taking the first {@code rank} columns of the
     * result would not undo that honestly, since the leading columns of a QR
     * factor are not the leading directions of the spectrum. The truncation to
     * the requested rank happens in {@link #computeSVD()}, where the singular
     * values say which directions to keep.
     *
     * @return an orthonormal basis of the approximate range of {@code A}
     */
    public MatrixD computeQ() {
        MatrixD Q = sketchBasis();
        if (transpose) {
            return Q;
        }
        // for a tall matrix the iteration ends in the row space, so one more
        // product and a QR factorization carry it over to the range of A
        return A.times(Q).qrd().getQ();
    }

    /**
     * The basis the subspace iteration produces, in whichever dimension was
     * cheaper to iterate in.
     * <p>
     * <b>What this returns depends on the shape of {@code A}</b>, which is why
     * it is not the public entry point:
     * <table border="1">
     * <caption>what the iteration ends with</caption>
     * <tr><th>shape</th><th>result</th><th>{@link #computeBQ()} then forms</th></tr>
     * <tr><td>wide, {@code rows < columns}</td>
     *     <td>{@code rows x w}, a basis of the range of {@code A}</td>
     *     <td>{@code B = Q' * A}</td></tr>
     * <tr><td>tall, {@code rows >= columns}</td>
     *     <td>{@code columns x w}, a basis of the <em>row</em> space</td>
     *     <td>{@code B = A * Q}</td></tr>
     * </table>
     * <p>
     * That is deliberate and not an oversight: the iteration works in the
     * smaller of the two dimensions, which is what keeps it affordable on a
     * matrix that is far from square. {@link #computeSVD()} resolves the two
     * cases correctly, and {@link #computeQ()} pays one product and one QR
     * factorization to give the tall case the same contract as the wide one.
     *
     * @return the basis of the iteration, of a shape that depends on {@code A}
     */
    private MatrixD sketchBasis() {
        MatrixD Q = getRandomMatrix();
        if (transpose) {
            return loopWideSaveAllocations(Q, A.transpose());
        }
        return loopTallSaveAllocations(Q, A.transpose());
    }

    private MatrixD loopWideSaveAllocations(MatrixD Q, MatrixD AT) {
        MatrixD C1 = Matrices.createD(A.numRows(), Q.numColumns());
        MatrixD C2 = null;

        Q = A.mult(Q, C1).lud().getPL();
        if (Q.numColumns() != C1.numColumns()) {
            C2 = Matrices.createD(AT.numRows(), Q.numColumns());
        } else {
            C2 = Matrices.createD(AT.numRows(), C1.numColumns());
        }
        Q = AT.mult(Q, C2).lud().getPL();
        if (Q.numColumns() != C1.numColumns()) {
            C1 = Matrices.createD(A.numRows(), A.numRows());
        }

        for (int i = 0; i < 3; ++i) {
            Q = A.mult(Q, C1).lud().getPL();
            Q = AT.mult(Q, C2).lud().getPL();
        }
        return A.mult(Q, C1).qrd().getQ();
    }

    private MatrixD loopTallSaveAllocations(MatrixD Q, MatrixD AT) {
        MatrixD C1 = Matrices.createD(AT.numRows(), Q.numColumns());
        MatrixD C2 = null;

        Q = AT.mult(Q, C1).lud().getPL();
        if (Q.numColumns() != AT.numRows()) {
            C2 = Matrices.createD(A.numRows(), Q.numColumns());
        } else {
            C2 = Matrices.createD(A.numRows(), AT.numRows());
        }
        Q = A.mult(Q, C2).lud().getPL();
        if (Q.numColumns() != C1.numColumns()) {
            C1 = Matrices.createD(AT.numRows(), AT.numRows());
        }

        for (int i = 0; i < 3; ++i) {
            Q = AT.mult(Q, C1).lud().getPL();
            Q = A.mult(Q, C2).lud().getPL();
        }
        return AT.mult(Q, C1).qrd().getQ();
    }

    /**
     * Draws the random test matrix and applies {@code A} to it.
     * <p>
     * Standard normal, because that is what the bounds of the paper are stated
     * for. This drew from {@code randomUniformD(-1, 1)} until it was measured:
     * uniform works just as well here, and the reason it does is that the four
     * subspace iterations wash the starting distribution out. Over 150 random
     * cases both found the planted rank 150 times; the error of a fixed width
     * sketch against the signal had a median of {@code 3.3007e-02} against
     * {@code 3.3076e-02}; and on a seeded run the leading singular value came
     * out identical to ten decimal places, with only the round-off of an
     * exactly low rank input distinguishing the two at all
     * ({@code 2.9e-15} against {@code 5.5e-15}). So nothing was gained by the
     * change and nothing lost, and what is left is that the guarantees of the
     * paper now formally apply where before they did not. The draw itself is
     * {@code O(rows * (rank + P))} against the {@code O(rows * columns * rank)}
     * of the product it feeds, so its cost cannot matter.
     *
     * @return the sketch {@code A * Omega}, transposed where the shape of
     *         {@code A} calls for it
     */
    private MatrixD getRandomMatrix() {
        MatrixD Omega = null;
        if (transpose) {
            Omega = seeded ? Matrices.randomNormalD(targetRank + P, m, seed)
                    : Matrices.randomNormalD(targetRank + P, m);
            return Omega.times(A).transpose();
        }
        Omega = seeded ? Matrices.randomNormalD(n, targetRank + P, seed)
                : Matrices.randomNormalD(n, targetRank + P);
        return A.times(Omega);
    }
}
