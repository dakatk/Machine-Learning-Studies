package functions;

import math.Matrix;

/**
 *
 */
public interface ActivationFn {

    Matrix call(Matrix x);
    Matrix derivative(Matrix x);
}
