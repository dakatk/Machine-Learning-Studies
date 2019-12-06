package functions.activation;

import functions.ActivationFn;
import math.Matrix;
import math.fns.Max;
import math.fns.Pos;
import org.jetbrains.annotations.NotNull;

/**
 *
 */
public class Relu implements ActivationFn {

    @Override
    public Matrix call(@NotNull Matrix x) {

        return x.clone().applyFunction(new Max());
    }

    @Override
    public Matrix derivative(@NotNull Matrix x) {

        return x.clone().applyFunction(new Pos());
    }
}
