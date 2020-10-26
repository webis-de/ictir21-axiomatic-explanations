import numpy as np

def strictlygreater(x, y):
    if x > y: return 1
    elif y > x: return -1
    return 0

def strictlygreaterInverse(x, y):
    if x > y: return -1
    elif y > x: return 1
    return 0

def approximatelyEqual(*args, marginFrac=0.1):
    """True if all numeric args are within (100 * marginFrac)% of the largest."""

    if len(args) == 1 and hasattr(args[0], '__len__'):
        # passed a list instead of multiple args
        args = args[0]

    args = np.array(args)
    m = args[np.abs(args).argmax()]
    if m == 0:
        # largest absolute value is zero -- all zeros
        return True
    b = np.array([m*(1+marginFrac), m*(1-marginFrac)])
    return np.all((args > b.min()) & (args < b.max()))
