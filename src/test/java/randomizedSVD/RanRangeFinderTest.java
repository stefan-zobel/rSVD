package randomizedSVD;

import static org.junit.Assert.assertTrue;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;
import net.jamu.matrix.SvdD;

import org.junit.Test;

public class RanRangeFinderTest {

    private static final int m = 220;
    private static final int n = 150;
    // this will only work if you get the rank estimation right
    private static final double TOLERANCE = 1.0e-7;

    @Test
    public void testNaturalNumbers() {
        // this is really low rank
        int estimatedRank = 2;
        MatrixD A = Matrices.naturalNumbersD(m, n);
        MatrixD Q = getQ(A, estimatedRank);
        MatrixD B = checkFactorization(Q, A);
        checkSVD(B, Q, A);
    }

    @Test
    public void testRandomNormal() {
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomNormalD(m, n);
        MatrixD Q = getQ(A, estimatedRank);
        MatrixD B = checkFactorization(Q, A);
        checkSVD(B, Q, A);
    }

    @Test
    public void testRandomUniform() {
        // high rank random noise
        int estimatedRank = Math.min(m, n);
        MatrixD A = Matrices.randomUniformD(m, n);
        MatrixD Q = getQ(A, estimatedRank);
        MatrixD B = checkFactorization(Q, A);
        checkSVD(B, Q, A);
    }

    private MatrixD getQ(MatrixD A, int estimatedRank) {
        return new RanRangeFinder(A, estimatedRank).computeQ();
    }

    private MatrixD checkFactorization(MatrixD Q, MatrixD A) {
        MatrixD B = Q.transpose().times(A);
        MatrixD A_approx = Q.times(B);
        boolean equal = Matrices.approxEqual(A_approx, A, TOLERANCE);
        assertTrue("A_approx and A should be approximately equal", equal);
        return B;
    }

    private void checkSVD(MatrixD B, MatrixD Q, MatrixD A_expected) {
        SvdD svdReduced = B.svd(true);
        MatrixD U_lowrank = Q.times(svdReduced.getU());
        // U
        MatrixD U_approx = Matrices.embed(m, n, U_lowrank);
        // Sigma
        MatrixD tmp = Matrices.diagD(svdReduced.getS());
        MatrixD Sigma = Matrices.embed(n, n, tmp);
        // Vt
        MatrixD Vt = svdReduced.getVt();
        // A_approx
        MatrixD A_approx = U_approx.timesTimes(Sigma, Vt);
        boolean equal = Matrices.approxEqual(A_approx, A_expected, TOLERANCE);
        assertTrue("A and reconstruction of A should be approximately equal", equal);
    }
}
