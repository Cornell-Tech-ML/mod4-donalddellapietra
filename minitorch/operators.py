"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable


# Implementation of a prelude of elementary functions.


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Return the identity of a number."""
    return x


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Compare two numbers."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Compare two numbers."""
    return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close within a tolerance of 1e-2."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid of a number according to the formula."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU of a number."""
    return x if x > 0 else 0


EPS = 1e-6


def log(x: float) -> float:
    """Compute the logarithm of a number."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Compute the exponential of a number."""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the derivative of the logarithm of a number, times a second number."""
    return d / (x + EPS)


def inv(x: float) -> float:
    """Compute the inverse of a number."""
    return 1 / x


def inv_back(x: float, d: float) -> float:
    """Compute the derivative of the inverse of a number, times a second number."""
    return -d / x**2


def relu_back(x: float, d: float) -> float:
    """Compute the derivative of the ReLU of a number, times a second number."""
    return d * (x > 0)


def sigmoid_back(x: float, d: float) -> float:
    """Compute the derivative of the sigmoid of a number, times a second number."""
    return d * sigmoid(x) * (1 - sigmoid(x))


def exp_back(x: float, d: float) -> float:
    """Compute the derivative of the exponential of a number, times a second number."""
    return d * exp(x)


# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(f: Callable[[float], float], xs: Iterable[float]) -> Iterable[float]:
    """Apply a function to each element of an iterable."""
    return [f(x) for x in xs]


def zipWith(
    f: Callable[[float, float], float], xs: Iterable[float], ys: Iterable[float]
) -> Iterable[float]:
    """Apply a function to corresponding elements of two iterables."""
    return [f(x, y) for x, y in zip(xs, ys)]


def reduce(
    f: Callable[[float, float], float], xs: Iterable[float], initial: float = 0.0
) -> float:
    """Reduce an iterable to a single value by applying a function to each element, given some initial value."""
    result = initial
    for x in xs:
        result = f(result, x)
    return result


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate every element of a list."""
    return map(neg, xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add two lists element-wise."""
    return zipWith(add, xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Sum every element in a list."""
    return reduce(add, xs, 0.0)


def prod(xs: Iterable[float]) -> float:
    """Take the product of every element in a list."""
    return reduce(mul, xs, 1.0)


# def max(xs: Iterable[float]) -> float:
#     """Return the maximum element in a list."""
#     return reduce(max, xs, float("-inf"))


# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists
