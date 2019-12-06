package functions;

import math.Matrix;

/**
 *
 */
public interface CostFn {

    // o = actual, y = expected

    double call(Matrix y, Matrix o);
    Matrix gradient(Matrix y, Matrix o);
}
