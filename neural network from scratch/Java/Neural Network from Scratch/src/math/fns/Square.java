package math.fns;

import functions.CallableFn;

/**
 *
 */
public class Square implements CallableFn<Double, Double> {

    @Override
    public Double call(Double x) {

        return x * x;
    }
}
