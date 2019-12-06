package math.fns;

import functions.CallableFn;

/**
 *
 */
public class Abs implements CallableFn<Double, Double> {

    @Override
    public Double call(Double x) {

        return Math.abs(x);
    }
}