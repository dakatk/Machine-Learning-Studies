package functions;

/**
 *
 * @param <T>
 * @param <U>
 */
public interface CallableFn<T, U> {

    T call(U x);
}
