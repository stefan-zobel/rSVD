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

import java.util.Objects;

import net.jamu.matrix.Matrices;
import net.jamu.matrix.MatrixD;
import net.jamu.matrix.SvdD;

/**
 * The way into this library: pick what you know about the matrix, and let the
 * choice of algorithm follow from that.
 * <p>
 * There are three things a caller can know, and each leads somewhere else:
 * <ul>
 * <li>{@link Start#toAccuracy(double) toAccuracy} - "approximate it to this
 * relative accuracy". The width follows from the target, and nothing has to be
 * guessed. This is the adaptive range finder, Algorithm 4.2.</li>
 * <li>{@link Start#findingTheRank() findingTheRank} - "there is a signal in
 * here, find out how wide it is". Neither a rank nor an accuracy target is
 * needed; the rank is determined from the spectrum by the optimal hard
 * threshold of Gavish and Donoho.</li>
 * <li>{@link Start#toRank(int) toRank} - "I know the rank, use it". A sketch of
 * exactly that width plus subspace iteration.</li>
 * </ul>
 * <p>
 * Each choice returns a type that carries <em>only</em> the settings that apply
 * to it, so a combination that makes no sense does not compile rather than
 * failing at run time. {@code toRank(20).blockSize(16)} is not a mistake this
 * API can make: a block size belongs to the adaptive path and exists nowhere
 * else.
 *
 * <pre>
 * RandomizedSVD.of(A).findingTheRank().seed(7L).decompose();
 * RandomizedSVD.of(A).toAccuracy(1e-3).blockSize(16).decompose();
 * RandomizedSVD.of(A).toRank(20).seed(7L).decompose();
 *
 * MatrixD Q = RandomizedSVD.of(A).toAccuracy(1e-3).basis();
 * </pre>
 * <p>
 * The facade delegates, it does not reimplement: every chain below ends in
 * {@link AdaRangeFinder} or {@link ApproximateBasis}, which remain public and
 * are still the way to reach anything this does not offer -
 * {@code math.rsvd.reference}, the transcriptions of the paper, and the
 * {@link Nystroem} postprocessing for a positive semidefinite matrix, which is
 * composed rather than configured:
 *
 * <pre>
 * SVD evd = Nystroem.decompose(A, RandomizedSVD.of(A).toAccuracy(1e-3).basis());
 * </pre>
 */
public final class RandomizedSVD {

    private RandomizedSVD() {
        throw new AssertionError("no instances");
    }

    /**
     * Starts a decomposition of {@code A}.
     *
     * @param A
     *            the matrix to decompose
     * @return the choice of what is known about {@code A}
     * @throws NullPointerException
     *             if {@code A} is {@code null}
     */
    public static Start of(MatrixD A) {
        return new Start(Objects.requireNonNull(A));
    }

    /**
     * Completes a basis of the approximate range of {@code A} to a
     * decomposition: project {@code A} onto the basis, decompose the small
     * matrix that results, and lift its left factors back.
     * <p>
     * {@code Q.transAmult(A, ...)} rather than {@code Q.transpose().times(A)},
     * which would allocate a whole transposed copy of {@code Q} on the way.
     *
     * @param A
     *            the matrix that was sketched
     * @param Q
     *            a matrix with orthonormal columns whose range approximates the
     *            range of {@code A}
     * @return the decomposition {@code A ~ U * S * Vt} carried by that basis
     */
    static SVD complete(MatrixD A, MatrixD Q) {
        MatrixD projected = Q.transAmult(A, Matrices.createD(Q.numColumns(), A.numColumns()));
        SvdD small = projected.svdEcon();
        return new SVD(Q.times(small.getU()), small.getS(), small.getVt());
    }

    /**
     * The choice of what is known about the matrix. Every path from here leads
     * to a decomposition; they differ in what the caller has to supply.
     */
    public static final class Start {

        private final MatrixD A;

        Start(MatrixD A) {
            this.A = A;
        }

        /**
         * Approximate {@code A} to a relative accuracy, letting the width
         * follow from the target rather than from a guess.
         *
         * @param epsilon
         *            the relative accuracy target, in the range
         *            {@code (0.0, 1.0]}. Validated when the decomposition runs,
         *            by {@link AdaRangeFinder}
         * @return the settings of the adaptive path
         */
        public Accuracy toAccuracy(double epsilon) {
            return new Accuracy(A, epsilon);
        }

        /**
         * Sketch {@code A} at a width the caller already knows.
         * <p>
         * If the rank is not known, prefer {@link #findingTheRank()}. It exists
         * precisely because this is the one parameter of the library that a
         * caller usually cannot supply: it is a statement about {@code A}, and
         * whoever knows {@code A} that well rarely needs the decomposition.
         *
         * @param rank
         *            the target rank, at least 1. Validated when the
         *            decomposition runs, by {@link ApproximateBasis}
         * @return the settings of the fixed width path
         */
        public FixedWidth toRank(int rank) {
            return new FixedWidth(A, rank);
        }

        /**
         * Let the library determine how many singular values carry signal, and
         * return a decomposition truncated to exactly those.
         * <p>
         * Neither a rank nor an accuracy target is needed. See
         * {@link ApproximateBasis#decompose(MatrixD)} for what this costs and
         * where it pays off.
         *
         * @return the settings of the automatic path
         */
        public Automatic findingTheRank() {
            return new Automatic(A);
        }
    }

    /** The adaptive path: a relative accuracy target, and the width follows. */
    public static final class Accuracy {

        private final MatrixD A;
        private final double epsilon;
        /**
         * Whether {@link #blockSize(int)} was called, kept apart from the value
         * itself. Treating any value that is not positive as "not set" would
         * turn {@code blockSize(0)} into a silent fall back to the
         * column-by-column path instead of the rejection it has to be.
         */
        private boolean blocked;
        private int blockSize;
        private boolean seeded;
        private long seed;

        Accuracy(MatrixD A, double epsilon) {
            this.A = A;
            this.epsilon = epsilon;
        }

        /**
         * Process the test vectors a block at a time instead of one at a time,
         * which is the organization Remark 4.2 of the paper describes.
         * <p>
         * Not set by default, so that without this call the width is exactly
         * the one Algorithm 4.2 prescribes. Blocking is faster - measured, a
         * {@code 400 x 401} matrix at the default accuracy target went from
         * 74.8 ms to 28.3 ms at a block size of 16, and an {@code 800 x 801}
         * one from 576.8 ms to 112.3 ms - but it returns a wider basis, up to
         * {@code 3.5 * blockSize} columns more, and that width is felt again in
         * {@link #decompose()}, which decomposes a matrix of exactly that many
         * rows. Sixteen was the only value measured that never lost on time.
         *
         * @param blockSize
         *            the number of test vectors per iteration, at least 1.
         *            Validated when the decomposition runs, by
         *            {@link AdaRangeFinder#computeQ(int)}
         * @return this
         */
        public Accuracy blockSize(int blockSize) {
            this.blocked = true;
            this.blockSize = blockSize;
            return this;
        }

        /**
         * Draw the test vectors from {@code seed}, so that a run can be
         * repeated. Two seeded runs agree to round-off rather than bit for bit;
         * see {@link AdaRangeFinder#AdaRangeFinder(MatrixD, double, long)} for
         * why no seed can promise more than that.
         *
         * @param seed
         *            the starting point of the sequence of test vectors
         * @return this
         */
        public Accuracy seed(long seed) {
            this.seeded = true;
            this.seed = seed;
            return this;
        }

        /**
         * Computes a matrix with orthonormal columns whose range approximates
         * the range of {@code A} to the accuracy target.
         *
         * @return the basis, never {@code null}
         */
        public MatrixD basis() {
            AdaRangeFinder finder = seeded ? new AdaRangeFinder(A, epsilon, seed)
                    : new AdaRangeFinder(A, epsilon);
            return blocked ? finder.computeQ(blockSize) : finder.computeQ();
        }

        /**
         * Computes the decomposition that the basis carries.
         *
         * @return the decomposition {@code A ~ U * S * Vt}
         */
        public SVD decompose() {
            return complete(A, basis());
        }
    }

    /** The fixed width path: the caller supplies the rank. */
    public static final class FixedWidth {

        private final MatrixD A;
        private final int rank;
        private boolean seeded;
        private long seed;

        FixedWidth(MatrixD A, int rank) {
            this.A = A;
            this.rank = rank;
        }

        /**
         * Draw the test matrix from {@code seed}, so that a run can be
         * repeated. Two seeded runs agree to round-off rather than bit for bit;
         * see {@link ApproximateBasis#ApproximateBasis(MatrixD, int, long)}.
         *
         * @param seed
         *            the seed for the random test matrix
         * @return this
         */
        public FixedWidth seed(long seed) {
            this.seeded = true;
            this.seed = seed;
            return this;
        }

        /**
         * Computes a matrix with orthonormal columns whose range approximates
         * the range of {@code A}.
         * <p>
         * <b>Wider than the rank that was asked for</b>, by the oversampling of
         * the method, so this and {@link #decompose()} do not agree in width.
         * The oversampling is what makes the basis worth having, and dropping
         * it here would not be honest arithmetic - see
         * {@link ApproximateBasis#computeQ()}. The truncation to the requested
         * rank belongs where the singular values are known, which is in the
         * decomposition.
         *
         * @return the basis, never {@code null}
         */
        public MatrixD basis() {
            return (seeded ? new ApproximateBasis(A, rank, seed) : new ApproximateBasis(A, rank))
                    .computeQ();
        }

        /**
         * Computes the decomposition at the width that was asked for.
         *
         * @return the decomposition {@code A ~ U * S * Vt}
         */
        public SVD decompose() {
            return (seeded ? new ApproximateBasis(A, rank, seed) : new ApproximateBasis(A, rank))
                    .computeSVD();
        }
    }

    /** The automatic path: nothing is supplied, the rank is determined. */
    public static final class Automatic {

        private final MatrixD A;
        private boolean seeded;
        private long seed;

        Automatic(MatrixD A) {
            this.A = A;
        }

        /**
         * Draw the test matrices from {@code seed}, so that a run can be
         * repeated. Two seeded runs agree to round-off rather than bit for bit;
         * see {@link ApproximateBasis#decompose(MatrixD, long)}.
         *
         * @param seed
         *            the seed for the random test matrices
         * @return this
         */
        public Automatic seed(long seed) {
            this.seeded = true;
            this.seed = seed;
            return this;
        }

        /**
         * Computes a matrix with orthonormal columns whose range approximates
         * the range of {@code A}, exactly as wide as the rank that was found.
         * <p>
         * This is the left factor of {@link #decompose()}, which is already an
         * orthonormal basis and already truncated to the rank. Taking it costs
         * nothing beyond the decomposition this path performs anyway, and it is
         * the better basis: on this path the rank is the whole point, so a
         * wider one carrying directions the threshold rejected would be worse,
         * not more generous.
         *
         * @return the basis, never {@code null}
         */
        public MatrixD basis() {
            return decompose().getU();
        }

        /**
         * Computes the decomposition, already truncated to the rank that was
         * found.
         *
         * @return the decomposition {@code A ~ U * S * Vt}
         */
        public SVD decompose() {
            return seeded ? ApproximateBasis.decompose(A, seed) : ApproximateBasis.decompose(A);
        }
    }
}
