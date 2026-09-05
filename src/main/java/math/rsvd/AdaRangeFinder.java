/*
 * Copyright 2020, 2026 Stefan Zobel
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

import java.util.ArrayList;
import java.util.Objects;
import java.util.Random;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;
import net.jamu.matrix.QrdD;

/**
 * Adaptive Randomized Range Finder.
 * <p>
 * Algorithm 4.2 from Nathan Halko, Per-Gunnar Martinsson, and Joel A Tropp.
 * Finding structure with randomness: Probabilistic algorithms for constructing
 * approximate matrix decompositions. SIAM review, 53(2):217-288, 2011.
 */
public class AdaRangeFinder {

    /**
     * The default relative accuracy target used by
     * {@link #AdaRangeFinder(MatrixD)}.
     */
    public static final double DEFAULT_EPSILON = 1.0e-3;

    /** The IEEE 754 machine epsilon from Cephes: (2^-53) */
    private static final double MACH_EPS_DBL = 1.11022302462515654042e-16;
    private static final int r = 10;
    /** The constant 1 / (10 * sqrt(2 / pi)) from Algorithm 4.2 */
    private static final double BOUND_FACTOR = 1.0 / (10.0 * Math.sqrt(2.0 / Math.PI));

    private final MatrixD A;
    private final int n;
    /** epsilon * ||A||_F / (10 * sqrt(2 / pi)) */
    private final double bound;
    /**
     * MACH_EPS_DBL * ||A||_F, below which a vector computed from {@code A} is
     * indistinguishable from zero at working precision. Like {@link #bound}
     * this is relative to the scale of {@code A}, so that no decision of this
     * class depends on the absolute magnitude of the entries
     */
    private final double zeroTol;
    /** the numerical rank of A cannot exceed min(rows, columns) */
    private final int maxCols;
    /** whether {@link #seed} was supplied by the caller */
    private final boolean seeded;
    /** starting point of the seed sequence, only meaningful if {@link #seeded} */
    private final long seed;

    /**
     * Creates a range finder for {@code A} which uses {@link #DEFAULT_EPSILON}
     * as its relative accuracy target. The test vectors are drawn from an
     * unspecified source of randomness, so repeated runs generally differ.
     *
     * @param A
     *            the matrix whose approximate range is sought
     * @throws IllegalArgumentException
     *             if {@code ||A||_F} is not positive and finite
     */
    public AdaRangeFinder(MatrixD A) {
        this(A, DEFAULT_EPSILON);
    }

    /**
     * Creates a range finder for {@code A} which stops as soon as the estimated
     * residual {@code ||(I - Q * Q') * A||} has dropped below
     * {@code epsilon * ||A||_F}, or as soon as {@code Q} has
     * {@code min(rows, columns)} columns, whichever happens first. The
     * criterion is relative to the Frobenius norm of {@code A} and is therefore
     * invariant under a rescaling of {@code A}. That invariance holds as long
     * as the entries of {@code A} stay in the normal range of a {@code double};
     * once they turn denormal the products {@code A * omega} lose accuracy and
     * fewer columns are found.
     *
     * @param A
     *            the matrix whose approximate range is sought
     * @param epsilon
     *            the relative accuracy target, must be in the range
     *            {@code (0.0, 1.0]}
     * @throws IllegalArgumentException
     *             if {@code epsilon} is not in the range {@code (0.0, 1.0]}, or
     *             if {@code ||A||_F} is not positive and finite
     */
    public AdaRangeFinder(MatrixD A, double epsilon) {
        this(A, epsilon, false, 0L);
    }

    /**
     * Creates a reproducible range finder for {@code A}. Two range finders
     * constructed with the same matrix, the same {@code epsilon} and the same
     * {@code seed} draw the same test vectors and stop after the same number of
     * them, and repeated {@link #computeQ()} calls on one instance do so as
     * well.
     * <p>
     * The bases they return agree to round-off rather than bit for bit, and no
     * seed can change that. The BLAS and LAPACK routines underneath are free to
     * divide their work differently from one run to the next, and floating point
     * addition is not associative, so the same operands can add up to a slightly
     * different sum. Measured over 21 runs per case, the column count was
     * identical in every case - on inputs that stop early as well as on inputs
     * that run to the {@code min(rows, columns)} cap - and the entries of
     * {@code Q} agreed to {@code 1.2e-14} absolute.
     * <p>
     * Note that there is deliberately no {@code (MatrixD, long)} overload: it
     * would be chosen over {@code (MatrixD, double)} for an integer literal, so
     * {@code new AdaRangeFinder(A, 1)} would silently mean a seed rather than
     * an accuracy target. Use {@link #DEFAULT_EPSILON} explicitly instead.
     *
     * @param A
     *            the matrix whose approximate range is sought
     * @param epsilon
     *            the relative accuracy target, must be in the range
     *            {@code (0.0, 1.0]}
     * @param seed
     *            the starting point of the sequence of test vectors
     * @throws IllegalArgumentException
     *             if {@code epsilon} is not in the range {@code (0.0, 1.0]}, or
     *             if {@code ||A||_F} is not positive and finite
     */
    public AdaRangeFinder(MatrixD A, double epsilon, long seed) {
        this(A, epsilon, true, seed);
    }

    private AdaRangeFinder(MatrixD A, double epsilon, boolean seeded, long seed) {
        this.A = Objects.requireNonNull(A);
        // the negated comparison also rejects NaN
        if (!(epsilon > 0.0 && epsilon <= 1.0)) {
            throw new IllegalArgumentException("epsilon: " + epsilon);
        }
        double normF = A.normF();
        // the negated comparison also rejects NaN. A zero matrix has to be
        // rejected here because its range is the zero subspace, and a matrix
        // with zero columns cannot be constructed
        if (!(normF > 0.0 && normF < Double.POSITIVE_INFINITY)) {
            throw new IllegalArgumentException(
                    "the range of a matrix with ||A||_F = " + normF + " is not defined");
        }
        this.n = A.numColumns();
        this.bound = epsilon * normF * BOUND_FACTOR;
        this.zeroTol = MACH_EPS_DBL * normF;
        this.maxCols = Math.min(A.numRows(), A.numColumns());
        this.seeded = seeded;
        this.seed = seed;
    }

    /**
     * Draws the next {@code n x 1} test vector.
     * <p>
     * Note that the seeded factory methods of {@code Matrices} construct a new
     * {@code Random} from the seed on every call, so passing one and the same
     * seed to all of them would yield identical vectors and would silently
     * degenerate the algorithm. Each draw therefore consumes a fresh seed from
     * {@code seeds}.
     *
     * @param seeds
     *            the seed sequence, or {@code null} for an unseeded run
     * @return a new {@code n x 1} standard normal random vector
     */
    private MatrixD nextTestVector(Random seeds) {
        return (seeds == null) ? Matrices.randomNormalD(n, 1) : Matrices.randomNormalD(n, 1, seeds.nextLong());
    }

    private static double norm(MatrixD y) {
        // y is always a column vector, so normF() is its euclidean length
        // (norm2() would yield the same value but computes a full SVD)
        return y.normF();
    }

    private static double getMax(ArrayList<MatrixD> vectors) {
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < vectors.size(); ++i) {
            double norm = norm(vectors.get(i));
            if (norm > max) {
                max = norm;
            }
        }
        return max;
    }

    /**
     * Applies the orthogonal projector {@code I - Q * Q'} to every column of
     * {@code Y}, overwriting {@code Y} with {@code Y - Q * (Q' * Y)}.
     * <p>
     * The projector is never formed explicitly. Doing so would need a
     * {@code rows x rows} matrix and {@code O(rows * rows)} work per column,
     * whereas the two products used here need {@code O(rows * k)} work per
     * column and no storage beyond the {@code k x b} matrix of coefficients,
     * where {@code k} is the number of columns of {@code Q} and {@code b} the
     * number of columns of {@code Y}.
     * <p>
     * A single column vector is the case {@code b == 1}. Passing a whole block
     * instead turns the two matrix-vector products into two matrix products,
     * which is the point of {@link #computeQ(int)}.
     *
     * @param Q
     *            a matrix with orthonormal columns
     * @param Y
     *            the columns to project, overwritten with the result
     */
    private static void project(MatrixD Q, MatrixD Y) {
        // C = Q' * Y
        MatrixD C = Matrices.createD(Q.numColumns(), Y.numColumns());
        Q.transAmult(Y, C);
        // Y = Y - Q * C (multAdd accumulates into Y, it does not clear it)
        Q.multAdd(-1.0, C, Y);
    }

    /**
     * Computes a matrix {@code Q} with orthonormal columns whose range
     * approximates the range of {@code A} to the accuracy target given at
     * construction time. {@code Q} has at least one and at most
     * {@code min(rows, columns)} columns and is never {@code null}.
     *
     * @return an orthonormal basis of the approximate range of {@code A}
     * @throws ArithmeticException
     *             if the products {@code A * omega} underflow to zero, which
     *             can only happen for a matrix whose nonzero entries are all
     *             denormal
     */
    public MatrixD computeQ() {

        // a local seed sequence keeps computeQ() idempotent and reentrant
        Random seeds = seeded ? new Random(seed) : null;

        ArrayList<MatrixD> vectors = new ArrayList<>(r);
        for (int k = 0; k < r; ++k) {
            vectors.add(A.times(nextTestVector(seeds)));
        }

        MatrixD y = vectors.get(0);
        double norm = norm(y);
        if (norm <= zeroTol) {
            // A itself is nonzero, the constructor rejects a zero matrix, so
            // the product A * omega must have underflowed. There is no basis
            // to return in that case: a matrix with zero columns cannot be
            // constructed, and normalizing y would divide by zero
            throw new ArithmeticException(
                    "A * omega underflowed to " + norm + " for ||A||_F = " + A.normF());
        }

        MatrixD q = Matrices.sameDimD(y);
        q = y.scale(1.0 / norm, q);

        MatrixD Q = q.copy();

        shift(vectors, q, Q, seeds);

        // the loop test has to see the test vectors in the state they are in
        // now, that is projected onto the orthogonal complement of the range of
        // the Q that has just been built. Taking the maximum before the shift
        // above would compare unprojected norms against the bound and would
        // always spend one iteration too many
        double max = getMax(vectors);

        // stop as soon as the residual estimate has dropped below
        // epsilon * ||A||_F / (10 * sqrt(2 / pi)), and never use more columns
        // than the numerical rank of A can possibly have
        while (max > bound && Q.numColumns() < maxCols) {

            // project the oldest test vector onto the orthogonal complement of
            // the range of Q. This overwrites vectors.get(0) in place, which is
            // safe because the shift() call below discards that element anyway
            y = vectors.get(0);
            project(Q, y);

            norm = norm(y);
            if (norm <= zeroTol) {
                break;
            }
            q = y.scale(1.0 / norm, q);
            Q = Q.appendColumn(q);

            shift(vectors, q, Q, seeds);

            max = getMax(vectors);
        }

        return Q;
    }

    /**
     * Computes a matrix {@code Q} with orthonormal columns whose range
     * approximates the range of {@code A} to the accuracy target given at
     * construction time, drawing and orthogonalizing {@code blockSize} test
     * vectors at a time instead of one at a time.
     * <p>
     * This is the same Algorithm 4.2 that {@link #computeQ()} implements,
     * organized the way Remark 4.2 of the paper describes: "The calculations in
     * Algorithm 4.2 can be organized so that each iteration processes a block
     * of samples simultaneously. This revision can lead to dramatic
     * improvements in speed because it allows us to exploit higher-level linear
     * algebra subroutines (e.g., BLAS3) or parallel processors." Two matrix
     * products replace {@code 2 * blockSize} matrix-vector products, one QR
     * decomposition replaces {@code blockSize} normalizations, and - what costs
     * the most in the column-by-column path - {@code Q} grows once per block
     * rather than once per column. Growing it copies the entire basis, so the
     * copying falls by a factor of {@code blockSize}.
     * <p>
     * The price is named in the same remark: "Although blocking can lead to the
     * generation of unnecessary samples, this outcome is generally harmless".
     * The stopping criterion is tested once per block rather than once per
     * column, and the look-ahead samples it is tested on are not the same ones
     * the column-by-column path would have looked at, so the two do not stop in
     * the same place. Measured over 2040 cases - random shapes, ranks, noise
     * levels and accuracy targets, across seven block sizes - this never
     * returned <em>fewer</em> columns than {@link #computeQ()}, and it always
     * met the accuracy target. The excess is <em>not</em> bounded by the block
     * size: the worst seen was {@code 3.5 * blockSize}, and 38 columns in
     * absolute terms. Where the residual decays slowly the stopping point is
     * sensitive to which samples the estimator happens to look at, and blocking
     * shifts that window. Where the number of columns has to be exactly the one
     * that Algorithm 4.2 prescribes, use {@link #computeQ()}.
     * <p>
     * At {@code blockSize == 1} this draws exactly the test vectors that
     * {@link #computeQ()} draws, in the same order, and returns the same number
     * of columns. It spans the same subspace, but not with the same basis: a QR
     * decomposition of a single column may flip its sign, and a basis of a
     * subspace is only determined up to an orthogonal transformation anyway.
     * <p>
     * When {@code blockSize} exceeds the ten look-ahead samples of the paper,
     * the pool of samples that the stopping criterion looks at grows with it.
     * Requiring more samples to be small is stricter than the criterion of the
     * paper, so that can only stop later, never earlier.
     * <p>
     * On what to pass, measured against {@link #computeQ()} on this machine,
     * median of nine runs. A {@code 400 x 401} matrix holding a rank 10 signal
     * in noise, at the default accuracy target, took 74.8 ms column by column
     * and 28.3 ms at {@code blockSize} 16, and an {@code 800 x 801} matrix of
     * full rank went from 576.8 ms to 112.3 ms. Both are cases where the
     * column-by-column path was slower than a dense {@code svdEcon()}; blocked,
     * the larger one is faster than it. Growing {@code blockSize} beyond 16
     * buys little and starts to hurt where the rank is genuinely small: on an
     * exactly rank 10 input, {@code blockSize} 16 took 1.0 ms against 1.6 ms
     * column by column, but 64 took 3.3 ms, because a block that wide
     * overshoots a rank of 10 by more than five times. Sixteen was the only
     * value measured that never lost.
     * <p>
     * Like {@link #computeQ()} this is idempotent and reentrant, and for a
     * seeded range finder two runs agree to round-off rather than bit for bit,
     * for the reasons given in {@link #AdaRangeFinder(MatrixD, double, long)}.
     *
     * @param blockSize
     *            the number of test vectors processed per iteration, at least
     *            {@code 1}. A value above {@code min(rows, columns)} is capped
     *            at that
     * @return an orthonormal basis of the approximate range of {@code A}
     * @throws IllegalArgumentException
     *             if {@code blockSize} is less than {@code 1}
     * @throws ArithmeticException
     *             if the products {@code A * omega} underflow to zero, which
     *             can only happen for a matrix whose nonzero entries are all
     *             denormal
     */
    public MatrixD computeQ(int blockSize) {
        if (blockSize < 1) {
            throw new IllegalArgumentException("blockSize must be at least 1, but was " + blockSize);
        }

        // a local seed sequence keeps computeQ(int) idempotent and reentrant
        Random seeds = seeded ? new Random(seed) : null;

        // never fewer than the r look-ahead samples of the paper, and never
        // fewer than one block, so that a whole block can always be consumed
        int poolSize = Math.max(r, Math.min(blockSize, maxCols));
        MatrixD pool = sampleBlock(poolSize, seeds);

        MatrixD Q = null;
        while (true) {
            int k = (Q == null) ? 0 : Q.numColumns();
            // the first block is accepted before any test, the way the
            // column-by-column path accepts its first vector unconditionally
            if (Q != null && (k >= maxCols || maxOf(columnNorms(pool)) <= bound)) {
                return Q;
            }

            int width = Math.min(blockSize, maxCols - k);
            MatrixD Y = leadingColumns(pool, width);
            if (Q != null) {
                // step 7 of Algorithm 4.2, applied to a whole block. The pool
                // is kept projected, so this is the second pass that makes the
                // orthogonalization a twice-is-enough one
                project(Q, Y);
            }

            QrdD qr = Y.qrd();
            int kept = usableColumns(qr.getR());
            if (kept == 0) {
                if (Q == null) {
                    // A itself is nonzero, the constructor rejects a zero
                    // matrix, so the product A * omega must have underflowed.
                    // There is no basis to return in that case: a matrix with
                    // zero columns cannot be constructed
                    throw new ArithmeticException("A * omega underflowed to "
                            + Math.abs(qr.getR().get(0, 0)) + " for ||A||_F = " + A.normF());
                }
                return Q;
            }
            MatrixD Qb = qr.getQ();
            if (kept < width) {
                Qb = leadingColumns(Qb, kept);
            }
            if (Q != null) {
                // A block wider than the rank that is left over is deficient,
                // and a QR decomposition invents directions for the deficient
                // part. Those are orthonormal among themselves, but nothing
                // ties them to the orthogonal complement of Q, so appending
                // them as they are destroys the orthonormality of Q - measured
                // on an exactly rank 12 input with a block of eight,
                // |Q'Q - I| reached 1.0 and the range finder ran to the full
                // width instead of stopping at twelve columns. The diagonal of
                // R does not separate the two reliably: there the invented
                // directions came out at 4e-13 to 2.5e-11 while zeroTol was
                // 6e-14, so they passed the test that catches an exactly zero
                // column. Projecting the block once more and orthonormalizing
                // it again needs no threshold at all and keeps Q orthonormal
                // whatever the rank of the block was
                project(Q, Qb);
                Qb = Qb.qrd().getQ();
            }
            Q = (Q == null) ? Qb : Q.appendMatrix(Qb);

            if (kept < width) {
                // the rest of the block already lies in the range of Q, which
                // is where the column-by-column path breaks as well
                return Q;
            }
            pool = refill(pool, width, Q, seeds);
        }
    }

    /**
     * Draws {@code count} test vectors and applies {@code A} to all of them in
     * one product.
     * <p>
     * The vectors are drawn one at a time and assembled into a matrix rather
     * than drawn as a single {@code n x count} random matrix. That costs
     * {@code O(n * count)} against the {@code O(rows * n * count)} of the
     * product, so it is free, and it is what makes {@code computeQ(1)} draw
     * exactly the vectors that {@link #computeQ()} draws.
     *
     * @param count
     *            the number of test vectors to draw
     * @param seeds
     *            the seed sequence, or {@code null} for an unseeded run
     * @return the {@code rows x count} matrix {@code A * Omega}
     */
    private MatrixD sampleBlock(int count, Random seeds) {
        MatrixD Omega = Matrices.createD(n, count);
        for (int j = 0; j < count; ++j) {
            Omega.setColumnInplace(j, nextTestVector(seeds));
        }
        return A.times(Omega);
    }

    /**
     * Replaces the {@code width} samples that have just been consumed with
     * fresh ones and brings the whole pool back into the orthogonal complement
     * of the range of {@code Q}.
     *
     * @param pool
     *            the current pool, whose leading {@code width} columns have
     *            been consumed
     * @param width
     *            the number of columns that were consumed
     * @param Q
     *            the basis as it stands after the consumed block was appended
     * @param seeds
     *            the seed sequence, or {@code null} for an unseeded run
     * @return the refilled pool, projected against {@code Q}
     */
    private MatrixD refill(MatrixD pool, int width, MatrixD Q, Random seeds) {
        int rows = pool.numRows();
        int size = pool.numColumns();
        int surviving = size - width;
        MatrixD next = Matrices.createD(rows, size);
        if (surviving > 0) {
            next.setSubmatrixInplace(0, 0, pool, 0, width, rows - 1, size - 1);
        }
        next.setSubmatrixInplace(0, surviving, sampleBlock(width, seeds), 0, 0, rows - 1, width - 1);
        // one projection for the whole pool rather than two. The surviving
        // samples would only need the newest block, but projecting them against
        // all of Q again is the same second pass that step 7 applies to an
        // accepted block, and it is one matrix product instead of two
        project(Q, next);
        return next;
    }

    /**
     * How many leading columns of a block carry a direction that can be
     * trusted, read off the diagonal of its {@code R} factor.
     * <p>
     * {@code |R[j][j]|} is the length of column {@code j} after it has been
     * orthogonalized against the columns before it, which is exactly the
     * quantity that the column-by-column path compares against
     * {@link #zeroTol} before it accepts a vector. Measuring the columns of the
     * block instead would be the wrong quantity: eight samples drawn from a
     * matrix of rank four are each of full length, only their span is
     * deficient.
     * <p>
     * This is the counterpart of the {@code norm <= zeroTol} break in
     * {@link #computeQ()} and it catches the same thing, a direction that has
     * vanished into the range of {@code Q}. It is <em>not</em> a rank revealing
     * test, and no unpivoted QR is: measured on an exactly rank 12 input, the
     * directions a deficient block invents came out between {@code 4e-13} and
     * {@code 2.5e-11} while {@code zeroTol} was {@code 6e-14}, so they pass
     * this test comfortably. What keeps {@code Q} orthonormal in that case is
     * not this test but the second projection in {@link #computeQ(int)}.
     *
     * @param R
     *            the upper triangular factor of the block
     * @return the number of leading columns whose orthogonalized length is
     *         above {@link #zeroTol}
     */
    private int usableColumns(MatrixD R) {
        int usable = 0;
        while (usable < R.numColumns() && Math.abs(R.get(usable, usable)) > zeroTol) {
            ++usable;
        }
        return usable;
    }

    /**
     * The {@code count} leading columns of {@code M}, as a new matrix.
     *
     * @param M
     *            the matrix to take the columns from
     * @param count
     *            the number of leading columns
     * @return a new {@code M.numRows() x count} matrix
     */
    private static MatrixD leadingColumns(MatrixD M, int count) {
        return M.submatrix(0, 0, M.numRows() - 1, count - 1, Matrices.createD(M.numRows(), count), 0, 0);
    }

    /**
     * The euclidean lengths of the columns of {@code M}.
     * <p>
     * The columns are copied out and measured with {@code normF()} instead of
     * being summed in place from {@code getArrayUnsafe()}, which the
     * column-major storage would allow. Accumulating squares by hand underflows
     * to zero for a matrix scaled to around {@code 1e-200}, and every decision
     * of this class is deliberately invariant under a rescaling of {@code A}.
     * The copy costs {@code O(rows)} per column against the
     * {@code O(rows * k)} of the projection that surrounds it.
     *
     * @param M
     *            the matrix whose columns are to be measured
     * @return the euclidean length of each column of {@code M}
     */
    private static double[] columnNorms(MatrixD M) {
        int rows = M.numRows();
        double[] norms = new double[M.numColumns()];
        MatrixD column = Matrices.createD(rows, 1);
        for (int j = 0; j < norms.length; ++j) {
            M.submatrix(0, j, rows - 1, j, column, 0, 0);
            norms[j] = column.normF();
        }
        return norms;
    }

    /**
     * The largest of {@code values}, which is never empty here.
     *
     * @param values
     *            the values to take the maximum of
     * @return the largest value
     */
    private static double maxOf(double[] values) {
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < values.length; ++i) {
            if (values[i] > max) {
                max = values[i];
            }
        }
        return max;
    }

    private void shift(ArrayList<MatrixD> vectors, MatrixD q, MatrixD Q, Random seeds) {
        vectors.remove(0);
        MatrixD omega = nextTestVector(seeds);
        // yr is retained in the vectors list, so it must be a fresh matrix and
        // must not be written into a shared scratch buffer
        MatrixD yr = A.times(omega);
        project(Q, yr);
        vectors.add(yr);
        // re-orthogonalize the remaining test vectors against the newest basis
        // vector. That is the same projection with a single-column Q; the last
        // element is yr, which was already projected above
        for (int i = 0; i < vectors.size() - 1; ++i) {
            project(q, vectors.get(i));
        }
    }
}
