import operator

import numpy as np

ADD = operator.add
SUBTRACT = operator.sub
MULTIPLY = operator.mul
def DIVIDE(x, y):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(x, y)
def MODULO(x, y):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.mod(x, y)

    return np.where(y == 0, np.nan, result)

LT = operator.lt
GT = operator.gt
LTE = operator.le
GTE = operator.ge
EQ = operator.eq
NEQ = operator.ne

AND = operator.and_
OR = operator.or_
INVERT = operator.invert