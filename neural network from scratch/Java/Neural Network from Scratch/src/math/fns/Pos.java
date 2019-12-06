package math.fns;

import functions.CallableFn;

/**
 *
 */
public class Pos implements CallableFn<Double, Double> {

    @Override
    public Double call(Double x) {

        return x > 0.0 ? 1.0 : 0.0;
    }
}
