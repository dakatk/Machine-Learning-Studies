package neural.network;

import functions.ActivationFn;
import functions.CostFn;
import math.Matrix;
import math.fns.Abs;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.util.Random;

/**
 *
 */
@SuppressWarnings("WeakerAccess")
public class NN {

    public static final double DEFAULT_LEARNGING_RATE = 0.01;
    public static final double DEFAULT_LAMBDA = 0.002;

    // Weights and biases for each layer
    private Matrix[] weights;
    private Matrix[] biases;

    // Calculated values between layers
    private Matrix[] a;

    // Activation/transfer functions for each layer
    private ActivationFn[] activationFns;

    // Cost/loss function for the entire network
    private CostFn cost;

    // Learning rate to force gradient descent to
    // be incremental rather than immediate
    private double learningRate;

    // Lasso regularization factor (to reduce overfitting)
    private double regression;

    // Total number of layers (excluding input layer)
    private int layers;

    // Number of neurons between each layer
    private int[] neurons;

    @Contract(pure=true)
    private NN() {}

    public NN(ActivationFn[] activationFns, CostFn cost, int[] inputSize, int[] neurons) {

        this(activationFns, cost, inputSize, neurons, DEFAULT_LEARNGING_RATE, DEFAULT_LAMBDA);
    }

    public NN(ActivationFn[] activationFns, CostFn cost, int[] inputSize, int[] neurons, double regression) {

        this(activationFns, cost, inputSize, neurons, DEFAULT_LEARNGING_RATE, regression);
    }

    public NN(@NotNull ActivationFn[] activationFns, CostFn cost, int[] inputSize, int[] neurons, double learningRate, double regression) {

        this.regression = regression;
        this.learningRate = learningRate;

        this.neurons = neurons;
        this.layers = activationFns.length;

        this.cost = cost;
        this.activationFns = activationFns;

        this.weights = new Matrix[this.layers];
        this.biases = new Matrix[this.layers];

        this.a = new Matrix[this.layers + 1];

        Random random = new Random();

        for (int i = 0; i < this.layers; i ++) {

            int rows = (i == 0 ? inputSize[1] : neurons[i - 1]);

            this.weights[i] = new Matrix(random, rows, neurons[i]);
            this.biases[i] = new Matrix(random, inputSize[0], neurons[i]);
        }
    }

    /**
     *
     * @param inputs
     * @return
     */
    private Matrix feedForward(Matrix inputs) {

        this.a[0] = inputs.clone();

        for (int i = 0; i < this.layers; i ++) {

            // z = (a dot W) + b
            Matrix z = this.a[i].clone().dot(this.weights[i]).add(this.biases[i]);

            // a = f(z)
            this.a[i + 1] = this.activationFns[i].call(z);
        }

        // a[L] = predicted outputs
        return this.a[this.layers];
    }

    /**
     *
     * @param actualOutputs
     * @param predOutputs
     */
    private void backPropogate(Matrix actualOutputs, Matrix predOutputs) {

        // Derivative of cost function wrt weights
        Matrix[] dC_dW = new Matrix[this.layers];

        // Derivative of cost function wrt biases
        Matrix[] dC_db = new Matrix[this.layers];

        // Propogated error and corresponding delta (for both weights and biases)
        Matrix error, delta;

        // To reduce compiler warnings:
        delta = Matrix.ZEROS(0, 0);

        for (int i = this.layers - 1; i >= 0; i --) {

            /*
            Descent from outputs to output layer (final layer in the network) is handled
            using the cost function gradient of the actual and predicted outputs. Lasso
            Regularization (L1) is then applied to make sure weights don't minimize too quickly
             */
            if (i == this.layers - 1) {

                error = this.cost.gradient(actualOutputs, predOutputs);

                // TODO apply this to only weights?...
                double lassoReg = this.weights[i].clone().applyFunction(new Abs()).sum();
                error.add(lassoReg * this.regression);
            }

            // From any layer to it's preceding layer, the equation is E = delta dot weights.T
            else error = delta.dot(this.weights[i + 1].transpose());

            // Delta is determined by multiplying the calculated error with the
            // derivative of the layer's activation function (making sure to pass
            // the layer's output, 'a', as the input to the derivative)
            delta = error.mul(this.activationFns[i].derivative(this.a[i + 1]));

            // Multiply the calculated delta with the inputs to the current layer
            // to produce the change in weights due to the propogated error
            dC_dW[i] = this.a[i].transpose().dot(delta);

            // dz_db = 1, therefore dC_db is just the calculated delta
            dC_db[i] = delta.clone();
        }

        for (int i = 0; i < this.layers; i ++) {

            // weights = weights + (dC_dW * learning rate)
            this.weights[i].add(dC_dW[i].mul(this.learningRate));

            // biases = biases + (dC_dW * learning rate)
            this.biases[i].add(dC_db[i].mul(this.learningRate));
        }
    }

    /**
     * Predict output of neural network given a specific set of inputs
     *
     * @param testInputs inputs used to test neural network
     * @return Feed forward result given inputs
     */
    public Matrix prediction(Matrix testInputs) {

        return this.feedForward(testInputs);
    }

    /**
     *
     * @param testInputs
     * @param testOutputs
     * @return
     */
    public double loss(Matrix testInputs, Matrix testOutputs) {

        return this.cost.call(testOutputs, this.feedForward(testInputs));
    }

    /**
     *
     * @param epochs
     */
    public void train(Matrix trainInputs, Matrix trainOutputs, int epochs) {

        for (int i = 0; i < epochs; i ++)
            this.backPropogate(trainOutputs, this.feedForward(trainInputs));
    }

    @SuppressWarnings("MethodDoesntCallSuperMethod")
    @Override
    public NN clone() {

        NN copy = new NN();

        copy.neurons = this.neurons;

        copy.weights = this.weights.clone();
        copy.biases = this.biases.clone();

        copy.a = this.a.clone();

        copy.activationFns = this.activationFns.clone();
        copy.cost = this.cost;

        copy.learningRate = this.learningRate;
        copy.layers = this.layers;

        return copy;
    }
}
