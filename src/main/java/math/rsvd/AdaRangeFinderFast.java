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

import java.util.ArrayList;
import java.util.Objects;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;

/**
 * This should be about twice as fast as my initial (2021-07-29)
 * {@link AdaRangeFinder} implementation (but it also needs more memory).
 */
public class AdaRangeFinderFast {

    /**
     * The default relative accuracy target used by
     * {@link #AdaRangeFinderFast(MatrixD)}.
     */
    public static final double DEFAULT_EPSILON = 1.0e-3;

    /** The IEEE 754 machine epsilon from Cephes: (2^-53) */
    private static final double MACH_EPS_DBL = 1.11022302462515654042e-16;
    private static final int r = 10;
    /** The constant 1 / (10 * sqrt(2 / pi)) from Algorithm 4.2 */
    private static final double BOUND_FACTOR = 1.0 / (10.0 * Math.sqrt(2.0 / Math.PI));

    private final MatrixD A;
    private final MatrixD I;
    private final MatrixD TEMP1;
    private final MatrixD TEMP2;
    private final MatrixD TEMP3;
    private final int n;
    /** epsilon * ||A||_F / (10 * sqrt(2 / pi)) */
    private final double bound;
    /** the numerical rank of A cannot exceed min(rows, columns) */
    private final int maxCols;

    /**
     * Creates a range finder for {@code A} which uses {@link #DEFAULT_EPSILON}
     * as its relative accuracy target.
     *
     * @param A
     *            the matrix whose approximate range is sought
     */
    public AdaRangeFinderFast(MatrixD A) {
        this(A, DEFAULT_EPSILON);
    }

    /**
     * Creates a range finder for {@code A} which stops as soon as the estimated
     * residual {@code ||(I - Q * Q') * A||} has dropped below
     * {@code epsilon * ||A||_F}, or as soon as {@code Q} has
     * {@code min(rows, columns)} columns, whichever happens first. The
     * criterion is relative to the Frobenius norm of {@code A} and is therefore
     * invariant under a rescaling of {@code A}.
     *
     * @param A
     *            the matrix whose approximate range is sought
     * @param epsilon
     *            the relative accuracy target, must be in the range
     *            {@code (0.0, 1.0]}
     * @throws IllegalArgumentException
     *             if {@code epsilon} is not in the range {@code (0.0, 1.0]}
     */
    public AdaRangeFinderFast(MatrixD A, double epsilon) {
        this.A = Objects.requireNonNull(A);
        // the negated comparison also rejects NaN
        if (!(epsilon > 0.0 && epsilon <= 1.0)) {
            throw new IllegalArgumentException("epsilon: " + epsilon);
        }
        this.I = Matrices.identityD(A.numRows());
        this.TEMP1 = Matrices.createD(I.numRows(), I.numRows());
        this.TEMP2 = Matrices.createD(I.numRows(), I.numRows());
        this.TEMP3 = Matrices.createD(I.numRows(), 1);
        this.n = A.numColumns();
        this.bound = epsilon * A.normF() * BOUND_FACTOR;
        this.maxCols = Math.min(A.numRows(), A.numColumns());
    }

    private static double norm(MatrixD y) {
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

    // NOTE: unlike Algorithm 4.2 (and unlike AdaRangeFinder) the test vectors
    // here are drawn from U(-1, 1) rather than from N(0, 1). Their expected
    // squared norm is ||A||_F^2 / 3, so the effective epsilon of this class is
    // about sqrt(3) times the nominal one.
    public MatrixD computeQ() {

        ArrayList<MatrixD> vectors = new ArrayList<>(r);
        for (int k = 0; k < r; ++k) {
            vectors.add(A.times(Matrices.randomUniformD(n, 1, -1.0, 1.0)));
        }

        MatrixD y = vectors.get(0);
        double norm = norm(y);
        if (norm <= MACH_EPS_DBL) {
            // this also covers the case of a zero matrix (where bound == 0.0)
            return null;
        }
        double max = getMax(vectors);

        MatrixD q = Matrices.sameDimD(y);
        q = y.scale(1.0 / norm, q);

        MatrixD Q = q.copy();

        shift(vectors, q, Q);

        // stop as soon as the residual estimate has dropped below
        // epsilon * ||A||_F / (10 * sqrt(2 / pi)), and never use more columns
        // than the numerical rank of A can possibly have
        while (max > bound && Q.numColumns() < maxCols) {

            MatrixD QQT = Q.transBmult(Q, TEMP1);
            MatrixD x = I.add(-1.0, QQT, TEMP2);
            y = vectors.get(0);
            y = x.times(y);

            norm = norm(y);
            if (norm <= MACH_EPS_DBL) {
                break;
            }
            q = y.scale(1.0 / norm, q);
            Q = Q.appendColumn(q);

            shift(vectors, q, Q);

            max = getMax(vectors);
        }

        return Q;
    }

    private void shift(ArrayList<MatrixD> vectors, MatrixD q, MatrixD Q) {
        vectors.remove(0);
        MatrixD omega = Matrices.randomUniformD(n, 1, -1.0, 1.0);
        MatrixD I_minus = I.add(-1.0, Q.transBmult(Q, TEMP1), TEMP1);
        MatrixD A_times_Omega = A.mult(omega, TEMP3);
        MatrixD yr = I_minus.times(A_times_Omega);
        vectors.add(yr);
        MatrixD qt = q.transpose();
        for (int i = 0; i < vectors.size() - 1; ++i) {
            MatrixD y = vectors.get(i);
            MatrixD x = qt.times(y);
            MatrixD z = q.mult(x, TEMP3);
            // MatrixD z = q.scale(x.getUnsafe(0, 0), TEMP3);
            y.addInplace(-1.0, z);
        }
    }
}
