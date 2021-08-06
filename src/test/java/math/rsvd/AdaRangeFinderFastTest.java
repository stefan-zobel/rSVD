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

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;
import randomizedSVD.Checks;

import org.junit.Test;

public class AdaRangeFinderFastTest {

    private static final int m = 220;
    private static final int n = 150;
    // the adaptive algorithm has quite good tolerance even for matrices which
    // are not that large
    private static final double TOLERANCE = 1.0e-7;

    @Test
    public void testNaturalNumbersTall() {
        MatrixD A = Matrices.naturalNumbersD(m, n);
        MatrixD Q = getQ(A);
        MatrixD B = Checks.checkFactorization2(Q, A, TOLERANCE);
        Checks.checkSVD2(B, Q, A, TOLERANCE);
    }

    @Test
    public void testNaturalNumbersWide() {
        MatrixD A = Matrices.naturalNumbersD(n, m);
        MatrixD Q = getQ(A);
        MatrixD B = Checks.checkFactorization2(Q, A, TOLERANCE);
        Checks.checkSVD2(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalTall() {
        MatrixD A = Matrices.randomNormalD(m, n);
        MatrixD Q = getQ(A);
        MatrixD B = Checks.checkFactorization2(Q, A, TOLERANCE);
        Checks.checkSVD2(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalWide() {
        MatrixD A = Matrices.randomNormalD(n, m);
        MatrixD Q = getQ(A);
        MatrixD B = Checks.checkFactorization2(Q, A, TOLERANCE);
        Checks.checkSVD2(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformTall() {
        MatrixD A = Matrices.randomUniformD(m, n);
        MatrixD Q = getQ(A);
        MatrixD B = Checks.checkFactorization2(Q, A, TOLERANCE);
        Checks.checkSVD2(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformWide() {
        MatrixD A = Matrices.randomUniformD(m, n);
        MatrixD Q = getQ(A);
        MatrixD B = Checks.checkFactorization2(Q, A, TOLERANCE);
        Checks.checkSVD2(B, Q, A, TOLERANCE);
    }

    private MatrixD getQ(MatrixD A) {
        return new AdaRangeFinderFast(A).computeQ();
    }
}
