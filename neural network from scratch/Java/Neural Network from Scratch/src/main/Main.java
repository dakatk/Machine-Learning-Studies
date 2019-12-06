package main;

import functions.ActivationFn;
import functions.activation.Relu;
import functions.activation.Sigmoid;
import functions.cost.BinaryCrossentropy;
import math.Matrix;
import neural.network.NN;

public class Main {

    public static void main(String[] args) {

        Matrix trainInputs = new Matrix(new double[][]{
                {0, 0, 0},
                {0, 0, 1},
                {0, 1, 0},
                {0, 1, 1},
                {1, 0, 0},
                {1, 0, 1},
                {1, 1, 0},
                {1, 1, 1}
        });

        Matrix trainOutputs = new Matrix(new double[][]{
                {0, 1},
                {0, 0},
                {1, 0},
                {1, 0},
                {1, 1},
                {1, 0},
                {0, 0},
                {0, 0}
        });

        ActivationFn sigmoid = new Sigmoid(-1, 0);
        ActivationFn relu = new Relu();

        ActivationFn[] activations = new ActivationFn[]{ sigmoid, sigmoid, sigmoid, sigmoid };

        int[] input_size = new int[]{ trainInputs.getRows(), trainInputs.getCols() };

        NN nn = new NN(activations, new BinaryCrossentropy(), input_size, new int[]{ 4, 3, 3, 2 }, 0.001);
        nn.train(trainInputs, trainOutputs, 2000);

        System.out.println("Training prediction: " + nn.prediction(trainInputs));
        System.out.println("Training loss: " + nn.loss(trainInputs, trainOutputs));

        trainInputs.shuffle(10);

        System.out.println("Test inputs: " + trainInputs);
        System.out.println("Test prediction: " + nn.prediction(trainInputs));
    }
}
