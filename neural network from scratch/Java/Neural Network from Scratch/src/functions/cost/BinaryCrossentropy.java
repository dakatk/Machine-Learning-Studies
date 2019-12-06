package functions.cost;

import functions.CostFn;
import math.Matrix;
import math.fns.Log;
import org.jetbrains.annotations.NotNull;

public class BinaryCrossentropy implements CostFn {

    @Override
    public double call(@NotNull Matrix y, @NotNull Matrix o) {

        int n = y.getRows() * y.getCols();

        Matrix a = y.clone().mul(o.clone().applyFunction(new Log()));
        Matrix b = y.clone().rsub(1).mul(o.clone().rsub(1).applyFunction(new Log()));

        return a.add(b).sum() / (double)(-n);
    }

    @Override
    public Matrix gradient(@NotNull Matrix y, @NotNull Matrix o) {

        return y.clone().mul(o.clone().sub(1)).add(y.clone().rsub(1).mul(o));
    }
}
