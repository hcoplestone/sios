from timeit import default_timer as timer
import sympy as sp
import numpy as np
import autograd.numpy as ag_np
from autograd import grad

n = 100000
xval = 2.5

times_autograd = []
times_symbolic = []

# Autograd

def f_autograd(y):
    return (y * y * ag_np.exp(y) + 2 * y * ag_np.sin(y)) ** 3

for i in range(n):
    start_time = timer()
    autograd_f = grad(f_autograd)
    autograd_eval = autograd_f(xval)
    end_time = timer()
    elapsed_time = end_time - start_time
    times_autograd.append(elapsed_time)


# Symbolic
x = sp.Symbol('x')
f_sympy = (x * x * sp.exp(x) + 2 * x * sp.sin(x)) ** 3

for i in range(n):
    start_time = timer()
    symbolgrad_f = sp.diff(f_sympy, x)
    symbol_eval = symbolgrad_f.evalf(16, subs={x: xval})
    end_time = timer()
    elapsed_time = end_time - start_time
    times_symbolic.append(elapsed_time)

print('Autograd: f({}) = {}'.format(xval, autograd_eval))
print('Symbolic: f({}) = {}'.format(xval, symbol_eval))

print('Writing to files..')

fautograd = open('times-autograd.csv', 'w+')
fsymbolic = open('times-symbolic.csv', 'w+')
for i in range(n):
    fautograd.write("{}\n".format(times_autograd[i]))
    fsymbolic.write("{}\n".format(times_symbolic[i]))
fautograd.close()
fsymbolic.close()
print('Finished writing to files!')
