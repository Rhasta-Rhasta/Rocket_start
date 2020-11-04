"""
Instruments for polynomial interpolation of mathematical discrete functions with one independent variable.
Contains linear and cubic spline interpolation methods.
"""

import numpy as np
import numbers
from collections import defaultdict, OrderedDict

__all__ = [
    "isnumeric", "stack", "Piecewise", "Linear", "Spline",
]

DEFAULT_CACHE_SIZE = 16


def isnumeric(first, *args):
    """Check input array elements for numeric using module 'numbers' instruments.

    Return list of bool with same shape as input args.
    """
    res = []
    for arg in (first, *args):
        if np.iterable(arg) and not isinstance(arg, str):
            value = [isnumeric(x) for x in arg]
        else:
            value = isinstance(arg, numbers.Number)
        res.append(value)
    return res[0] if len(res) == 1 else res


def stack(first, *args):
    """Stacking several functions 'Piecewise' in one.

    Return 'Piecewise'.
    """
    res = first
    for fun in args:
        assert res.abscissas[-1] == fun.abscissas[0], f"Stacking domains {res.domain} and {fun.domain}" \
                                                      f"must have an equal inner boundaries"
        if not np.isclose(res.ordinates[-1], fun.ordinates[0]):
            print(f"Function has a break at the stack abscissa {fun.abscissas[0]}")
        res = Piecewise({**res, **fun})

    return res


def _prepare_arguments(xs, ys):
    assert np.iterable(xs), f"argument 'xs'={xs} must be iterable"
    assert np.iterable(ys), f"argument 'ys'={ys} must be iterable"
    assert np.ndim(xs) == 1, f"argument 'xs'={xs} must have one dimension"
    assert np.ndim(ys) == 1, f"argument 'ys'={ys} must have one dimension"
    assert len(xs) > 1, f"argument 'xs'={xs} must have at least two elements"
    assert len(ys) > 1, f"argument 'ys'={ys} must have at least two elements"
    assert all(isnumeric(xs)), f"argument 'xs'={xs} must have only numeric elements"
    assert all(isnumeric(ys)), f"argument 'ys'={ys} must have only numeric elements"

    assert len(xs) == len(ys), f"arguments 'xs'={xs} (len={len(xs)}) and 'ys'={ys} " \
                               f"(len={len(ys)}) must have an equal length"

    non_increase = [xj for xi, xj in zip(xs[:-1], xs[1:]) if xj <= xi]
    assert len(non_increase) == 0, f"element(s) 'xs'={non_increase} on function domain ({xs[0]}, {xs[-1]}) " \
                                   f"must be non decreased and non duplicated"

    return np.array(xs), np.array(ys)


class CacheDict(OrderedDict):
    """Limit size, evicting the least recently looked-up key when full"""

    def __init__(self, maxsize=DEFAULT_CACHE_SIZE, *args, **kwargs):
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self.nearhits = 0
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

    def cache_info(self):
        print(f"CacheInfo(hits={self.hits}, nearhits={self.nearhits}, misses={self.misses}, maxsize={self.maxsize}, "
              f"currsize={len(self)})")


class Piecewise(defaultdict):
    """Class of math discrete functions with one independent
    variable and polynomial value interpolation inside the domain.
    """

    def __init__(self, dictionary, cache_size=None):
        """Dictionary keys is abscissas, values - arrays of polynomial coefficients.

        It's strongly recommended to init that class with subclasses like 'Linear' or other
        for correct polynomial coefficients compute.

        :param dictionary: dict
        :param cache_size: int
        """
        super().__init__(list, dictionary)
        self._left, self._right = 0, len(self)-1
        if cache_size is not None:
            self.cache = CacheDict(cache_size)
        elif hasattr(dictionary, 'cache'):
            self.cache = CacheDict(dictionary.cache.maxsize)
        else:
            self.cache = CacheDict(DEFAULT_CACHE_SIZE)

    @classmethod
    def from_func(cls, func):
        """Return piecewise function basis on existing function points.

        It's useful for changing interpolation type for already existed function.
        """
        assert isinstance(func, Piecewise), f"argument 'func'={func} must be 'Piecewise' type"
        if isinstance(func, cls):
            return func
        else:
            return cls(func.abscissas, func.ordinates, func.cache.maxsize)

    def __call__(self, *args):
        """Return interpolated function ordinates.

        :param args: array_like of float or float.
            Input abscissas.
        :return: array_like of float or float.
            Output function ordinates.
            Result has the same shape as array_like args.
            For more then one args return the list of results.
        """
        res = []
        for arg in args:
            if np.iterable(arg) and not isinstance(arg, str):
                value = [self(x) for x in arg]
            else:
                if arg not in self.cache:
                    self.cache.misses += 1
                    key = self._find_key(arg)
                    self.cache[arg] = sum(c * arg ** i for i, c in enumerate(self[key]))
                else:
                    self.cache.hits += 1
                value = self.cache[arg]
            res.append(value)
        return res[0] if len(res) == 1 else res

    def _find_key(self, xt):
        """Binary search algorithm.

        Return the smaller nearest array element.

        :param xt: float
        :return: float
        """
        if xt in self:
            x = self.abscissas

            if self._right - self._left == 1 and x[self._left] <= xt < x[self._right]:
                self.cache.nearhits += 1
                return x[self._left]

            self._left, self._right = 0, len(x) - 1

            while self._right - self._left > 1:
                mid = (self._right + self._left) // 2
                if x[mid] > xt:
                    self._right = mid
                else:
                    self._left = mid
            return x[self._left]
        else:
            raise ValueError(f"Abscissa '{xt}' is out of function domain {self.domain}")

    def __contains__(self, value):
        """Checking the entry of 'value' abscissa in function domain."""
        assert isnumeric(value), f"function argument '{value}' must be numeric"
        left, right = self.domain
        return True if left <= value <= right else False

    def __pos__(self):
        """Return +self."""
        return self

    def __neg__(self):
        """Return -self."""
        return self * (-1)

    def __add__(self, other):
        """Return self+other."""
        assert isnumeric(other), f"addition argument '{other}' must be numeric"
        res = Piecewise(self)
        for x in res.abscissas[:-1]:
            res[x][0] += other
        return res

    def __radd__(self, other):
        """Return other+self."""
        return self + other

    def __sub__(self, other):
        """Return self-other."""
        return self + (-other)

    def __rsub__(self, other):
        """Return other-self."""
        return other + (-self)

    def __mul__(self, other):
        """Return new 'Piecewise' self*other."""
        assert isnumeric(other), f"multiplication coefficient '{other}' must be numeric"
        res = Piecewise(self)
        for (x, coeff) in res.items():
            res[x] = [c * other for c in coeff]
        return res

    def __rmul__(self, other):
        """Return other*self."""
        return self * other

    def __truediv__(self, other):
        """Return self/other."""
        return self * (1 / other)

    def cache_info(self):
        self.cache.cache_info()

    @property
    def abscissas(self):
        """Return array of abscissas."""
        return tuple(self.keys())

    @property
    def ordinates(self):
        """Return array of ordinates."""
        return self(self.abscissas)

    @property
    def domain(self):
        """Return function domain."""
        x = self.abscissas
        return x[0], x[-1]

    def diff(self, *args):
        """Compute function derivative.

        If 'args' is empty, return derivative as 'Piecewise',
        else return derivative values with '__call__' method.

        :param args: array_like of float or float
        :return: Piecewise or array_like of float
        """
        res = Piecewise(self)
        for (x, c) in self.items():
            res[x] = [c * i for i, c in enumerate(c) if i > 0]
        return res(*args) if args else res

    def integr(self, *args):
        """Compute function integral with zero initial condition.

        If 'args' is empty, return integral as 'Piecewise',
        else return integral values with '__call__' method.

        :param args: array_like of float or float
        :return: Piecewise or array_like of float
        """
        res = defaultdict(list)
        for x, coeff in self.items():
            res[x] = [c / (i + 1) for i, c in enumerate(coeff)]
            if x == self.domain[0]:
                left = 0
            else:
                left = sum(c * pow(x, i) for i, c in enumerate(tmp))
            res[x].insert(0, left - sum(c * pow(x, i + 1) for (i, c) in enumerate(res[x])))
            tmp = res[x]

        res = Piecewise(res)
        return res(*args) if args else res


class Linear(Piecewise):
    """Piecewise function with linear polynomial interpolation of internal values."""

    def __init__(self, xs, ys, cache_size=DEFAULT_CACHE_SIZE):
        """Return function with values linear interpolation.

        xs - array of abscissas.
        ys - array of ordinates.
        max_cache_size - internal cache size

        :param xs: 1-d array-like
        :param ys: 1-d array-like
        :param cache_size: int
        """
        x, y = _prepare_arguments(xs, ys)
        c = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        c = np.column_stack((y[:-1] - c * x[:-1], c))
        assert len(x) == len(c) + 1
        res = dict(zip(x, c))
        res[x[-1]] = []
        super().__init__(res, cache_size)


def _gauss(A, b):
    """Solving linear equation 'A * x = b' with Gauss procedure for three-diagonal matrix."""
    # direct
    A, b = np.array(A), np.array(b)
    for i in range(1, len(A)):
        s = A[i, 0] / A[i - 1, 1]
        A[i, 1] -= A[i - 1, 2] * s
        b[i] -= b[i - 1] * s
    # converse
    for i in reversed(range(len(A) - 1)):
        s = A[i, 2] / A[i + 1, 1]
        b[i] -= b[i + 1] * s
    return b / A[:, 1]


class Spline(Piecewise):
    """Piecewise function with cubic spline interpolation of internal values."""

    def __init__(self, xs, ys, cache_size=DEFAULT_CACHE_SIZE):
        """Return function with values spline interpolation.

        xs - array of abscissas.
        ys - array of ordinates.
        max_cache_size - internal cache size

        :param xs: 1-d array-like
        :param ys: 1-d array-like
        :param cache_size: int
        """
        x, y = _prepare_arguments(xs, ys)
        A = np.column_stack([
            (x[1:-1] - x[:-2]) / 6,
            (x[2:] - x[:-2]) / 3,
            (x[2:] - x[1:-1]) / 6
        ])
        A = np.insert(A, (0, len(A)), [0, 1, 0], axis=0)
        b = (y[2:] - y[1:-1]) / (x[2:] - x[1:-1]) - (y[1:-1] - y[0:-2]) / (x[1:-1] - x[0:-2])
        b = np.insert(b, (0, len(b)), 0)

        q = _gauss(A, b)

        h = x[1:] - x[:-1]
        c = np.column_stack([
            (q[:-1] * x[1:] ** 3 - q[1:] * x[:-1] ** 3) / 6 / h +
            (y[:-1] * x[1:] - y[1:] * x[:-1]) / h +
            (q[1:] * x[:-1] - q[:-1] * x[1:]) * h / 6,
            (q[1:] * x[:-1] ** 2 - q[:-1] * x[1:] ** 2) / h / 2 +
            (q[:-1] - q[1:]) * h / 6 + (y[1:] - y[:-1]) / h,
            (q[:-1] * x[1:] - q[1:] * x[:-1]) / h / 2,
            (q[1:] - q[:-1]) / h / 6
        ])
        assert len(x) == len(c) + 1
        res = dict(zip(x, c))
        res[x[-1]] = []
        super().__init__(res, cache_size)


if __name__ == '__main__':
    f1 = Linear([1, 1.5, 2], [1, 2, 1])
    print(f1(1.2))
    print(f1(1.3))
    print(f1(1.6))
    f1 = Spline.from_func(f1)
    f1(1, 2, 1, 2, )
    f1.cache_info()
    print(2 - f1)

    TESTS = True
    if TESTS:
        from scipy import interpolate
        f1 = Linear(range(5), np.sin(range(5)))
        f2 = interpolate.interp1d(range(5), np.sin(range(5)))
        print(f1(3.5), f2(3.5))
        print(f2)
        print(f2 * 2)
        print(f2(7))
        print(f1.diff(0))
        print(f1.integr(0))
        f2 = Spline(range(5), np.sin(range(5)))
        f1, f2 = Linear.from_func(f2), Spline.from_func(f1)
        print(f1(1.5))
        print(f1([[1.5, 2.5], 1.2], (0.5, 4), 1.5))
        f1.cache_info()
        f2.cache_info()
        f1 = +f1
        f1 = -f1
        f1 = f1 + 1
        f1 = 1 + f1
        f1 = f1 - 2
        f1 = f1 * 2
        f1 = 2 * f1
        f1 = f1 / 4
        for item in f1:
            print(item)
        f2 = f2.diff()
        f2 = f2.integr()
        print(f1.abscissas)
        print(f1.ordinates)
        print(f1.domain)
        f1.cache_info()
        f1 = stack(f1, Spline(range(4, 10), np.sin(range(4, 10))))
        print(f1(5))
