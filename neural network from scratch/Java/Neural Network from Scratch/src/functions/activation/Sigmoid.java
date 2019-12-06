package functions.activation;

import functions.ActivationFn;
import math.Matrix;
import math.fns.Exp;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

/**
 *
 */
public class Sigmoid implements ActivationFn {

    private double k;
    private double x0;

    @Contract(pure=true)
    public Sigmoid(double k, double x0) {

        this.k = k;
        this.x0 = x0;
    }

    @Override
    public Matrix call(@NotNull Matrix x) {

        Matrix xHat = x.clone().sub(this.x0).mul(-this.k);

        return xHat.applyFunction(new Exp()).add(1).rdiv(1);
    }

    @Override
    public Matrix derivative(@NotNull Matrix x) {

        Matrix sig = this.call(x);

        return sig.mul(sig.rsub(1));
    }
}
