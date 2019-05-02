from timeit import default_timer as timer
import sympy as sp
import numpy as np
import autograd.numpy as ag_np
from autograd import grad


def f_autograd(y):
    return (y * y * ag_np.exp(y) + 2 * y * ag_np.sin(y))**3


xval = 2.5
autograd_f = grad(f_autograd)
autograd_eval = autograd_f(xval)

print('Autograd: f({}) = {}'.format(xval, autograd_eval))

x = sp.Symbol('x')
f_sympy = (x * x * sp.exp(x) + 2 * x * sp.sin(x))**3

symbolgrad_f = sp.diff(f_sympy, x)
symbol_eval = symbolgrad_f.evalf(16, subs={x: xval})

print('Symbolic: f({}) = {}'.format(xval, symbol_eval))


