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

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;

import org.junit.Test;

/**
 * This is the recommended algorithm for the fixed rank problem.
 */
public class RanSubspaceIterationTest {

    private static final int m = 220;
    private static final int n = 150;
    // tolerance requirements can be more strict here
    private static final double TOLERANCE = 1.0e-8;
    private static final int q = 4;

    @Test
    public void testNaturalNumbersTall() {
        // this is really low rank
        int estimatedRank = 2;
        MatrixD A = Matrices.naturalNumbersD(m, n);
        MatrixD Q = getQ(A, estimatedRank, q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testNaturalNumbersWide() {
        // this is really low rank
        int estimatedRank = 2;
        MatrixD A = Matrices.naturalNumbersD(n, m);
        MatrixD Q = getQ(A, estimatedRank, q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalTall() {
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomNormalD(m, n);
        MatrixD Q = getQ(A, estimatedRank, q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalWide() {
        // high rank random noise
        int estimatedRank = Math.min(n, m);
        MatrixD A = Matrices.randomNormalD(m, n);
        MatrixD Q = getQ(A, estimatedRank, q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformTall() {
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomUniformD(m, n);
        MatrixD Q = getQ(A, estimatedRank, q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformWide() {
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomUniformD(m, n);
        MatrixD Q = getQ(A, estimatedRank, q);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    private MatrixD getQ(MatrixD A, int estimatedRank, int q) {
        return new RanSubspaceIteration(A, estimatedRank, q).computeQ();
    }
}
