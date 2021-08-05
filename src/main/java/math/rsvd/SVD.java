package math.rsvd;

import net.jamu.matrix.MatrixD;

public class SVD {

    public final MatrixD U;
    public final MatrixD S;
    public final MatrixD Vt;

    public SVD(MatrixD U, MatrixD S, MatrixD Vt) {
        this.U = U;
        this.S = S;
        this.Vt = Vt;
    }
}
