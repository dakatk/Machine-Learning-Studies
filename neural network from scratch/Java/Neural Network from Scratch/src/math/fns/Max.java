package math.fns;

import functions.CallableFn;

/**
 *
 */
public class Max implements CallableFn<Double, Double> {

    @Override
    public Double call(Double x) {

        return Math.max(0, x);
    }
}
