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

import java.util.ArrayList;
import java.util.Objects;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;

/**
 * This should be about twice as fast as my initial (2021-07-29)
 * {@link AdaRangeFinder} implementation (but it also needs more memory).
 */
public class AdaRangeFinderFast {

    /** The IEEE 754 machine epsilon from Cephes: (2^-53) */
    private static final double MACH_EPS_DBL = 1.11022302462515654042e-16;
    private static final int r = 10;
    private static final double BOUND = 1.0 / (10.0 * Math.sqrt(2.0 / Math.PI));

    private final MatrixD A;
    private final MatrixD I;
    private final MatrixD TEMP1;
    private final MatrixD TEMP2;
    private final MatrixD TEMP3;
    private final int n;

    public AdaRangeFinderFast(MatrixD A) {
        this.A = Objects.requireNonNull(A);
        this.I = Matrices.identityD(A.numRows());
        this.TEMP1 = Matrices.createD(I.numRows(), I.numRows());
        this.TEMP2 = Matrices.createD(I.numRows(), I.numRows());
        this.TEMP3 = Matrices.createD(I.numRows(), 1);
        this.n = A.numColumns();
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

    public MatrixD computeQ() {

        ArrayList<MatrixD> vectors = new ArrayList<>(r);
        for (int k = 0; k < r; ++k) {
            vectors.add(A.times(Matrices.randomUniformD(n, 1, -1.0, 1.0)));
        }

        MatrixD y = vectors.get(0);
        double norm = norm(y);
        if (norm <= MACH_EPS_DBL) {
            return null;
        }
        double max = getMax(vectors);

        MatrixD q = Matrices.sameDimD(y);
        q = y.scale(1.0 / norm, q);

        MatrixD Q = q.copy();

        shift(vectors, q, Q);

        // we implicitly set epsilon == 1
        while (max > BOUND) {

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
