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

import java.util.Arrays;
import java.util.Objects;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;

/**
 * An approximate singular value decomposition {@code A ~ U * S * Vt}, where
 * {@code U} has orthonormal columns, {@code Vt} has orthonormal rows and
 * {@code S} is the diagonal matrix of the singular values in descending order.
 * <p>
 * The singular values are held as an array rather than as a matrix. The
 * diagonal form is available from {@link #getS()}, but it is built on demand
 * and never stored: measured for a {@code 400 x 400} decomposition of rank
 * 400, it occupies 160,000 cells of which 400 are used, and accounts for a
 * third of the memory of the whole decomposition.
 */
public class SVD {

    /** The IEEE 754 machine epsilon from Cephes: (2^-53) */
    private static final double MACH_EPS_DBL = 1.11022302462515654042e-16;

    private final MatrixD U;
    private final MatrixD Vt;
    /** the singular values, exactly as many as U has columns */
    private final double[] sigma;

    /**
     * Creates a decomposition from its three factors.
     *
     * @param U
     *            the left singular vectors, one per column
     * @param singularValues
     *            the singular values in descending order. Only the first
     *            {@code U.numColumns()} of them belong to this decomposition;
     *            a longer array is accepted and truncated, which is what the
     *            economy sized decompositions of the underlying library hand
     *            out
     * @param Vt
     *            the transposed right singular vectors, one per row
     * @throws IllegalArgumentException
     *             if {@code Vt} does not have as many rows as {@code U} has
     *             columns, or if fewer singular values are supplied than
     *             {@code U} has columns
     */
    public SVD(MatrixD U, double[] singularValues, MatrixD Vt) {
        this.U = Objects.requireNonNull(U);
        this.Vt = Objects.requireNonNull(Vt);
        Objects.requireNonNull(singularValues);
        int k = U.numColumns();
        if (Vt.numRows() != k) {
            throw new IllegalArgumentException(
                    "U has " + k + " columns, so Vt must have " + k + " rows, but it has " + Vt.numRows());
        }
        if (singularValues.length < k) {
            throw new IllegalArgumentException("U has " + k + " columns, so at least " + k
                    + " singular values are needed, but only " + singularValues.length + " were supplied");
        }
        // a copy, for two reasons: the array may be longer than this
        // decomposition is wide, and a caller passing the array of a jamu
        // SvdD would otherwise keep a writable reference to our state, since
        // SvdD.getS() hands out its internal array directly
        this.sigma = Arrays.copyOf(singularValues, k);
    }

    /**
     * The left singular vectors, one per column.
     *
     * @return the {@code m x k} matrix {@code U}
     */
    public MatrixD getU() {
        return U;
    }

    /**
     * The transposed right singular vectors, one per row.
     *
     * @return the {@code k x n} matrix {@code Vt}
     */
    public MatrixD getVt() {
        return Vt;
    }

    /**
     * The singular values in descending order.
     *
     * @return a fresh copy of the {@code k} singular values, so that a caller
     *         cannot change this decomposition through the array it is handed
     */
    public double[] getSingularValues() {
        return sigma.clone();
    }

    /**
     * The singular values as a diagonal matrix.
     * <p>
     * This is the escape hatch for code that wants to write the product
     * {@code U * S * Vt} literally. It is built on every call and not cached,
     * because {@code k * k} cells for {@code k} numbers are not worth holding
     * on to; {@link #reconstruct()} computes that product without it.
     *
     * @return a new {@code k x k} diagonal matrix carrying the singular values
     */
    public MatrixD getS() {
        return Matrices.diagD(sigma);
    }

    /**
     * The number of singular values of this decomposition, that is the number
     * of columns of {@code U} and the number of rows of {@code Vt}.
     *
     * @return the number of singular values
     */
    public int size() {
        return sigma.length;
    }

    /**
     * The numerical rank of this decomposition, counting the singular values
     * above {@code max(rows, columns) * eps * sigma_1}, which is the usual
     * threshold.
     * <p>
     * Note what this can and cannot say. It is the rank of the
     * <em>approximation</em>, and the approximation has at most as many
     * singular values as the caller asked for when constructing it. A matrix
     * of rank 100 decomposed with an estimated rank of 10 reports 10 here, not
     * 100. The number is meaningful when the estimated rank was chosen
     * generously and one wants to know how much of it was actually needed.
     *
     * @return the number of singular values above the default threshold
     */
    public int rank() {
        if (sigma.length == 0) {
            return 0;
        }
        return rank(Math.max(U.numRows(), Vt.numColumns()) * MACH_EPS_DBL * sigma[0]);
    }

    /**
     * The number of singular values strictly above {@code tolerance}.
     *
     * @param tolerance
     *            the threshold below which a singular value counts as zero
     * @return the number of singular values above {@code tolerance}
     * @see #rank()
     */
    public int rank(double tolerance) {
        int r = 0;
        for (int i = 0; i < sigma.length; ++i) {
            if (sigma[i] > tolerance) {
                ++r;
            }
        }
        return r;
    }

    /**
     * Computes the product {@code U * S * Vt}, the approximation of the matrix
     * this decomposition was computed from.
     * <p>
     * The diagonal matrix is not formed: row {@code i} of {@code S * Vt} is
     * simply {@code sigma_i} times row {@code i} of {@code Vt}, which saves
     * both the {@code k x k} matrix and one of the two multiplications.
     *
     * @return a new {@code m x n} matrix
     */
    public MatrixD reconstruct() {
        MatrixD SVt = Vt.copy();
        for (int i = 0; i < sigma.length; ++i) {
            double s = sigma[i];
            for (int j = 0; j < SVt.numColumns(); ++j) {
                SVt.set(i, j, s * SVt.get(i, j));
            }
        }
        return U.times(SVt);
    }

    /**
     * The decomposition of rank {@code k} obtained by keeping only the
     * {@code k} largest singular values and their vectors.
     *
     * @param k
     *            the number of singular values to keep, from 1 to
     *            {@link #size()}
     * @return a new decomposition of rank {@code k}, or this one if {@code k}
     *         is already its size
     * @throws IllegalArgumentException
     *             if {@code k} is outside {@code [1, size()]}
     */
    public SVD truncate(int k) {
        if (k < 1 || k > sigma.length) {
            throw new IllegalArgumentException(
                    "k must be in [1, " + sigma.length + "], but was " + k);
        }
        if (k == sigma.length) {
            return this;
        }
        return new SVD(U.selectConsecutiveColumns(0, k - 1), Arrays.copyOf(sigma, k),
                Vt.selectSubmatrix(0, 0, k - 1, Vt.endCol()));
    }
}
