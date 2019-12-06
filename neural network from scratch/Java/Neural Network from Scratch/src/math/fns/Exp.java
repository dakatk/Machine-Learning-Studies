package math.fns;

import functions.CallableFn;

/**
 *
 */
public class Exp implements CallableFn<Double, Double> {

    @Override
    public Double call(Double x) {

        return Math.exp(x);
    }
}
