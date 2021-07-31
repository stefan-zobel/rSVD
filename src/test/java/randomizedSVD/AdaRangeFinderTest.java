package randomizedSVD;

import static org.junit.Assert.assertTrue;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;
import net.jamu.matrix.SvdD;

import org.junit.Test;

public class AdaRangeFinderTest {

    private static final int m = 220;
    private static final int n = 150;

    @Test
    public void testNaturalNumbers() {
        MatrixD A = Matrices.naturalNumbersD(m, n);
        MatrixD Q = getQ(A);
        MatrixD B = checkFactorization(Q, A);
        checkSVD(B, Q, A);
    }

    @Test
    public void testRandomNormal() {
        MatrixD A = Matrices.randomNormalD(m, n);
        MatrixD Q = getQ(A);
        MatrixD B = checkFactorization(Q, A);
        checkSVD(B, Q, A);
    }

    @Test
    public void testRandomUniform() {
        MatrixD A = Matrices.randomUniformD(m, n);
        MatrixD Q = getQ(A);
        MatrixD B = checkFactorization(Q, A);
        checkSVD(B, Q, A);
    }

    private MatrixD getQ(MatrixD A) {
        return new AdaRangeFinder(A).computeQ();
    }

    private MatrixD checkFactorization(MatrixD Q, MatrixD A) {
        MatrixD B = Q.transpose().times(A);
        MatrixD A_approx = Q.times(B);
        boolean equal = Matrices.approxEqual(A_approx, A, 1.0e-7);
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
        boolean equal = Matrices.approxEqual(A_approx, A_expected, 1.0e-7);
        assertTrue("A and reconstruction of A should be approximately equal", equal);
    }
}
