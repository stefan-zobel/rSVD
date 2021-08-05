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

public class RanRangeFinderTest {

    private static final int m = 220;
    private static final int n = 150;
    // this will only work if you get the rank estimation right
    private static final double TOLERANCE = 1.0e-7;

    @Test
    public void testNaturalNumbersTall() {
        // this is really low rank
        int estimatedRank = 2;
        MatrixD A = Matrices.naturalNumbersD(m, n);
        MatrixD Q = getQ(A, estimatedRank);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testNaturalNumbersWide() {
        // this is really low rank
        int estimatedRank = 2;
        MatrixD A = Matrices.naturalNumbersD(n, m);
        MatrixD Q = getQ(A, estimatedRank);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalTall() {
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomNormalD(m, n);
        MatrixD Q = getQ(A, estimatedRank);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomNormalWide() {
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomNormalD(n, m);
        MatrixD Q = getQ(A, estimatedRank);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformTall() {
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomUniformD(m, n);
        MatrixD Q = getQ(A, estimatedRank);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    @Test
    public void testRandomUniformWide() {
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomUniformD(n, m);
        MatrixD Q = getQ(A, estimatedRank);
        MatrixD B = Checks.checkFactorization(Q, A, TOLERANCE);
        Checks.checkSVD(B, Q, A, TOLERANCE);
    }

    private MatrixD getQ(MatrixD A, int estimatedRank) {
        return new RanRangeFinder(A, estimatedRank).computeQ();
    }
}
