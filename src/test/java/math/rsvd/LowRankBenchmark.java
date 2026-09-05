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
import java.util.Arrays;
import java.util.List;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;
import net.jamu.matrix.SvdD;

/**
 * How a dense decomposition, an adaptive range finder and a fixed width sketch
 * compare on a {@code 400 x 401} matrix that is <em>not</em> of full rank.
 * <p>
 * A randomized decomposition can only win where the rank is small, so a
 * benchmark on a full rank matrix measures the one case these classes are not
 * for. This one plants a rank of 10 or 20 and keeps the full rank case as a
 * reference line, so that the size of the win can be read off rather than
 * assumed.
 * <p>
 * Three things are needed to make the comparison honest, and all three are easy
 * to get wrong:
 * <ul>
 * <li>Only the decomposition is timed. Forming the approximation and measuring
 * its error costs a multiplication and two norms on a {@code 400 x 401} matrix,
 * which is of the same order as the fast contenders themselves, so it happens
 * outside the timed region.</li>
 * <li>The accuracy is reported next to the time. The three do not compute the
 * same thing: {@code svdEcon()} returns all {@code min(rows, columns)} singular
 * triplets, {@code AdaRangeFinder.computeQ()} returns only a basis of the
 * approximate range and no decomposition at all, and {@code ApproximateBasis}
 * returns a decomposition of the width it was asked for. The range finder
 * therefore appears twice, on its own and completed to a decomposition the way a
 * caller would have to complete it.</li>
 * <li>Where the input is a low rank signal buried in noise, the error is also
 * reported against the <em>signal</em>. Measured against the noisy input, a
 * method that correctly discards the noise looks as though it had failed by
 * exactly the noise level.</li>
 * </ul>
 * <p>
 * This is a {@code main}, not a test. It is far too slow for the suite and its
 * numbers depend on the machine.
 */
public final class LowRankBenchmark {

    private static final int ROWS = 400;
    private static final int COLUMNS = 401;
    private static final int WARMUP = 3;
    private static final int ITERATIONS = 11;
    private static final long SEED = 7L;
    /** the block sizes compared against the column-by-column range finder */
    private static final int[] BLOCK_SIZES = { 8, 16, 32 };
    /**
     * The block size the completed-to-an-SVD row uses. Sixteen because it was
     * the only one measured that never lost against the column-by-column path.
     */
    private static final int COMPLETION_BLOCK_SIZE = 16;

    /** what a contender produced. Building the approximation is not timed. */
    private interface Result {
        int width();

        MatrixD approximation(MatrixD A);
    }

    private static final class Decomposition implements Result {
        private final SVD svd;

        Decomposition(SVD svd) {
            this.svd = svd;
        }

        public int width() {
            return svd.size();
        }

        public MatrixD approximation(MatrixD A) {
            return svd.reconstruct();
        }
    }

    /** a range finder returns a basis, whose promise is the projection Q * Q' * A */
    private static final class Basis implements Result {
        private final MatrixD Q;

        Basis(MatrixD Q) {
            this.Q = Q;
        }

        public int width() {
            return Q.numColumns();
        }

        public MatrixD approximation(MatrixD A) {
            return Q.times(Q.transpose().times(A));
        }
    }

    private interface Contender {
        Result run(MatrixD A);
    }

    private static double relativeError(MatrixD reference, MatrixD approximation) {
        return reference.copy().addInplace(-1.0, approximation).normF() / reference.normF();
    }

    private static double median(double[] values) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    private static void time(String label, MatrixD A, MatrixD signal, Contender contender) {
        for (int i = 0; i < WARMUP; ++i) {
            contender.run(A);
        }
        double[] milliseconds = new double[ITERATIONS];
        Result result = null;
        for (int i = 0; i < ITERATIONS; ++i) {
            long start = System.nanoTime();
            result = contender.run(A);
            milliseconds[i] = (System.nanoTime() - start) / 1.0e6;
        }
        // outside the timed region: forming the approximation costs as much as
        // the fast contenders do
        MatrixD approximation = result.approximation(A);
        String againstSignal = (signal == null) ? "-"
                : String.format("%.2e", relativeError(signal, approximation));
        Arrays.sort(milliseconds);
        System.out.printf("  %-54s %8.1f %8.1f %7d %11.2e %11s%n", label, median(milliseconds), milliseconds[0],
                result.width(), relativeError(A, approximation), againstSignal);
    }

    private static Contender dense() {
        return new Contender() {
            public Result run(MatrixD A) {
                SvdD svd = A.svdEcon();
                return new Decomposition(new SVD(svd.getU(), svd.getS(), svd.getVt()));
            }
        };
    }

    private static Contender rangeFinderOnly(final double epsilon) {
        return new Contender() {
            public Result run(MatrixD A) {
                return new Basis(new AdaRangeFinder(A, epsilon, SEED).computeQ());
            }
        };
    }

    /**
     * The same range finder with its samples processed a block at a time, which
     * is the organization Remark 4.2 of the paper describes. It returns a few
     * more columns than the column-by-column path, so the width column is worth
     * reading next to the time.
     */
    private static Contender blockedRangeFinder(final double epsilon, final int blockSize) {
        return new Contender() {
            public Result run(MatrixD A) {
                return new Basis(new AdaRangeFinder(A, epsilon, SEED).computeQ(blockSize));
            }
        };
    }

    /**
     * The range finder completed to a decomposition: project, decompose the
     * small matrix, lift the left factors back. This is the work a caller has to
     * do to get from {@code computeQ()} to something comparable with the other
     * two, and leaving it out is what makes a bare {@code computeQ()} timing
     * look better than it is.
     */
    private static Contender rangeFinderToSVD(final double epsilon) {
        return new Contender() {
            public Result run(MatrixD A) {
                MatrixD Q = new AdaRangeFinder(A, epsilon, SEED).computeQ();
                SvdD small = Q.transpose().times(A).svdEcon();
                return new Decomposition(new SVD(Q.times(small.getU()), small.getS(), small.getVt()));
            }
        };
    }

    /**
     * The blocked range finder completed the same way. This is the row where
     * the overshoot of the blocked path first costs something: the completion
     * decomposes a {@code width x columns} matrix and lifts the left factors
     * back, so both steps grow with the width of {@code Q}, and the blocked
     * path returns a wider {@code Q} than the column-by-column one.
     */
    private static Contender blockedRangeFinderToSVD(final double epsilon, final int blockSize) {
        return new Contender() {
            public Result run(MatrixD A) {
                MatrixD Q = new AdaRangeFinder(A, epsilon, SEED).computeQ(blockSize);
                SvdD small = Q.transpose().times(A).svdEcon();
                return new Decomposition(new SVD(Q.times(small.getU()), small.getS(), small.getVt()));
            }
        };
    }

    private static Contender fixedWidth(final int width) {
        return new Contender() {
            public Result run(MatrixD A) {
                return new Decomposition(new ApproximateBasis(A, width, SEED).computeSVD());
            }
        };
    }

    private static Contender automatic() {
        return new Contender() {
            public Result run(MatrixD A) {
                return new Decomposition(ApproximateBasis.decompose(A, SEED));
            }
        };
    }

    private static void scenario(String title, MatrixD A, MatrixD signal, int rank, double[] epsilons) {
        double[] spectrum = A.svd(false).getS();
        System.out.printf("%n=== %s ===%n", title);
        System.out.printf("  sigma[0] %.3e   sigma[%d] %.3e   sigma[%d] %.3e   sigma[%d] %.3e%n", spectrum[0],
                rank - 1, spectrum[rank - 1], rank, spectrum[rank], spectrum.length - 1,
                spectrum[spectrum.length - 1]);
        System.out.printf("  %-54s %8s %8s %7s %11s %11s%n", "", "median", "best", "width", "err vs A",
                "err vs sig");

        List<String> labels = new ArrayList<>();
        List<Contender> contenders = new ArrayList<>();
        labels.add("svdEcon()");
        contenders.add(dense());
        labels.add("AdaRangeFinder(1e-03), Q only");
        contenders.add(rangeFinderOnly(1.0e-3));
        for (int i = 0; i < BLOCK_SIZES.length; ++i) {
            labels.add("AdaRangeFinder(1e-03).computeQ(" + BLOCK_SIZES[i] + "), Q only");
            contenders.add(blockedRangeFinder(1.0e-3, BLOCK_SIZES[i]));
        }
        for (int i = 0; i < epsilons.length; ++i) {
            labels.add(String.format("AdaRangeFinder(%.0e) completed to an SVD", epsilons[i]));
            contenders.add(rangeFinderToSVD(epsilons[i]));
            labels.add(String.format("AdaRangeFinder(%.0e).computeQ(%d) completed to an SVD", epsilons[i],
                    COMPLETION_BLOCK_SIZE));
            contenders.add(blockedRangeFinderToSVD(epsilons[i], COMPLETION_BLOCK_SIZE));
        }
        labels.add("ApproximateBasis(" + rank + "), told the rank");
        contenders.add(fixedWidth(rank));
        int overshot = Math.min(2 * rank, Math.min(ROWS, COLUMNS));
        labels.add("ApproximateBasis(" + overshot + "), overshot");
        contenders.add(fixedWidth(overshot));
        labels.add("ApproximateBasis.decompose(), told nothing");
        contenders.add(automatic());

        for (int i = 0; i < contenders.size(); ++i) {
            time(labels.get(i), A, signal, contenders.get(i));
        }
    }

    private static MatrixD lowRankSignal(int rank, long seed) {
        return Matrices.randomNormalD(ROWS, rank, seed).times(Matrices.randomNormalD(rank, COLUMNS, seed + 1000L));
    }

    public static void main(String[] args) {
        System.out.printf("%d x %d, %d warmup and %d timed repetitions per row, times in ms%n", ROWS, COLUMNS,
                WARMUP, ITERATIONS);

        for (int rank : new int[] { 10, 20 }) {
            MatrixD exact = lowRankSignal(rank, 11L + rank);
            scenario("exactly rank " + rank, exact, null, rank, new double[] { 1.0e-9 });

            MatrixD signal = lowRankSignal(rank, 11L + rank);
            MatrixD noise = Matrices.randomNormalD(ROWS, COLUMNS, 31L + rank).scaleInplace(0.5);
            MatrixD noisy = signal.copy().addInplace(1.0, noise);
            // an accuracy target below the noise level asks for the noise to be
            // reproduced as well, which no low rank basis can do
            scenario(String.format("rank %d buried in noise of relative size %.1e", rank,
                    noise.normF() / noisy.normF()), noisy, signal, rank, new double[] { 0.2 });
        }

        // the case these classes are not for. There is no low rank structure to
        // exploit, and decompose() reports a rank of 0 - represented by the
        // narrowest decomposition there is - which is the right answer for a
        // matrix that is all noise
        scenario("full rank, for reference", Matrices.randomNormalD(ROWS, COLUMNS, 99L), null, 20,
                new double[] { 0.2 });

        System.out.println();
        System.out.println("DONE");
    }
}
