package randomizedSVD;

import java.util.Objects;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;
import net.jamu.matrix.QrdD;

/**
 * Randomized Range Finder. Fixed-rank problem, where the target rank of the
 * input matrix is specified in advance.
 * <p>
 * Algorithm 4.1 from Nathan Halko, Per-Gunnar Martinsson, and Joel A Tropp.
 * Finding structure with randomness: Probabilistic algorithms for constructing
 * approximate matrix decompositions. SIAM review, 53(2):217–288, 2011.
 */
public class RanRangeFinder {

    // Oversampling parameter
    private static final int P = 10;

    private final MatrixD A;
    private final int n;
    private final int targetRank;

    public RanRangeFinder(MatrixD A, int estimatedRank) {
        if (estimatedRank < 0) {
            throw new IllegalArgumentException("negative target rank: " + estimatedRank);
        }
        this.A = Objects.requireNonNull(A);
        this.n = A.numColumns();
        this.targetRank = estimatedRank;
    }

    public MatrixD computeQ() {
        MatrixD Omega = Matrices.randomNormalD(n, targetRank + P);
        MatrixD Y = A.times(Omega);
        QrdD qr = Y.qrd();
        MatrixD Q = qr.getQ();
        return Q;
    }
}
