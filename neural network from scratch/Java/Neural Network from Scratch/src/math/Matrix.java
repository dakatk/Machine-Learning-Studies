package math;

import functions.CallableFn;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.util.Random;

/**
 *
 */
public class Matrix {

    private double[][] elements;

    private int rows;
    private int cols;

    @Contract(pure=true)
    private Matrix(int rows, int cols) {

        this.elements = new double[rows][cols];

        this.rows = rows;
        this.cols = cols;
    }

    public Matrix(Random random, int rows, int cols) {

        this(rows, cols);

        for (int i = 0; i < rows; i ++) {

            for (int j = 0; j < cols; j ++)
                this.elements[i][j] = random.nextFloat();
        }
    }

    public Matrix(@NotNull double[][] elements) {

        this.rows = elements.length;
        this.cols = elements[0].length;

        this.elements = new double[this.rows][this.cols];

        for (int i = 0; i < this.rows; i ++)
            System.arraycopy(elements[i], 0, this.elements[i], 0, this.cols);
    }

    @NotNull
    public static Matrix ZEROS(int rows, int cols) {

        Matrix zeros = new Matrix(rows, cols);

        for (int i = 0; i < rows; i ++) {

            for (int j = 0; j < cols; j ++)
                zeros.elements[i][j] = 0;
        }

        return zeros;
    }

    public double sum() {

        double net = 0.0;

        for (int i = 0; i < this.rows; i ++) {

            for(int j = 0; j < this.cols; j ++)
                net += this.elements[i][j];
        }

        return net;
    }

    public Matrix dot(@NotNull Matrix other) throws AssertionError {

        if (this.cols != other.rows)
            throw new AssertionError("M.rows != N.cols");

        double[][] product = new double[this.rows][other.cols];

        for (int i = 0; i < this.rows; i ++) {

            for (int j = 0; j < other.cols; j ++) {

                for (int k = 0; k < this.cols; k ++)
                    product[i][j] += this.elements[i][k] * other.elements[k][j];
            }
        }

        return new Matrix(product);
    }

    public Matrix applyFunction(CallableFn<Double, Double> fn) {

        for (int i = 0; i < this.rows; i ++) {

            for (int j = 0; j < this.cols; j ++)
                this.elements[i][j] = fn.call(this.elements[i][j]);
        }

        return this;
    }

    public Matrix add(@NotNull Matrix other) throws AssertionError {

        if (this.rows != other.rows && this.cols != other.cols)
            throw new AssertionError("M.dim != N.dim");

        for (int i = 0; i < this.rows; i ++) {

            for (int j = 0; j < this.cols; j ++)
                this.elements[i][j] += other.elements[i][j];
        }

        return this;
    }

    public Matrix add(double value) {

        for (int i = 0; i < this.rows; i ++) {

            for (int j = 0; j < this.cols; j ++)
                this.elements[i][j] += value;
        }

        return this;
    }

    public Matrix rsub(double value) {

        for (int i = 0; i < this.rows; i ++) {

            for (int j = 0; j < this.cols; j ++)
                this.elements[i][j] = value - this.elements[i][j];
        }

        return this;
    }

    public Matrix rdiv(double value) {

        for (int i = 0; i < this.rows; i ++) {

            for (int j = 0; j < this.cols; j ++)
                this.elements[i][j] = value / this.elements[i][j];
        }

        return this;
    }

    public Matrix sub(@NotNull Matrix other) throws AssertionError {

        if (this.rows != other.rows && this.cols != other.cols)
            throw new AssertionError("M.dim != N.dim");

        for (int i = 0; i < this.rows; i ++) {

            for (int j = 0; j < this.cols; j ++)
                this.elements[i][j] -= other.elements[i][j];
        }

        return this;
    }

    public Matrix sub(double value) {

        for (int i = 0; i < this.rows; i ++) {

            for (int j = 0; j < this.cols; j ++)
                this.elements[i][j] -= value;
        }

        return this;
    }

    public Matrix mul(Matrix other) {

        for (int i = 0; i < this.rows; i ++) {

            for (int j = 0; j < this.cols; j ++)
                this.elements[i][j] *= other.elements[i][j];
        }

        return this;
    }

    public Matrix mul(double value) {

        for (int i = 0; i < this.rows; i ++) {

            for (int j = 0; j < this.cols; j ++)
                this.elements[i][j] *= value;
        }

        return this;
    }

    public Matrix transpose() {

        Matrix T = new Matrix(this.cols, this.rows);

        for (int i = 0; i < this.rows; i ++) {

            for (int j = 0; j < this.cols; j ++)
                T.elements[j][i] = this.elements[i][j];
        }

        return T;
    }

    public void shuffle(int count) {

        Random random = new Random();

        for (int i = 0; i < count; i ++) {

            int swapIndex = random.nextInt(this.rows);
            int nextIndex = (swapIndex == this.rows - 1 ? 0 : swapIndex + 1);

            double[] swapRow = new double[this.cols];

            System.arraycopy(this.elements[swapIndex], 0, swapRow, 0, this.cols);
            System.arraycopy(this.elements[swapIndex], 0, this.elements[nextIndex], 0, this.cols);

            this.elements[nextIndex] = swapRow;
        }
    }

    public int getRows() {

        return this.rows;
    }

    public int getCols() {

        return this.cols;
    }

    @SuppressWarnings("MethodDoesntCallSuperMethod")
    @Override
    public Matrix clone() {

        return new Matrix(this.elements);
    }

    @Override
    public String toString() {

        StringBuilder output = new StringBuilder();

        output.append("[\n");

        for (int i = 0; i < this.rows; i ++) {

            for (int j = 0; j < this.cols; j ++) {

                if (j == 0) output.append(' ');

                output.append('[');
                output.append(this.elements[i][j]);
                output.append(']');

                if (j != this.cols - 1) output.append(", ");
            }
            output.append('\n');
        }
        output.append(']');

        return output.toString();
    }
}
