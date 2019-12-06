package math.fns;

import functions.CallableFn;

public class Log implements CallableFn<Double, Double> {

    @Override
    public Double call(Double x) {

        return Math.log(x);
    }
}
