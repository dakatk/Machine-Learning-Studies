package functions.cost;

import functions.CostFn;
import math.Matrix;
import math.fns.Square;
import org.jetbrains.annotations.NotNull;

/**
 *
 */
public class MSE implements CostFn {

    @Override
    public double call(@NotNull Matrix y, Matrix o) {

        Matrix diff = y.clone().sub(o);
        diff.applyFunction(new Square()).mul(0.5);

        return diff.sum();
    }

    @Override
    public Matrix gradient(@NotNull Matrix y, Matrix o) {

        return y.clone().sub(o);
    }
}
