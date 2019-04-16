from integrators import GalerkinGaussLobattoIntegrator
from flask import abort



def extract_param(params, key):
    param = next((param for param in params if param["key"].upper() == key.upper()), False)
    if param:
        return param['value']
    else:
        print('Cannot find value with key {0}'.format(key))
        abort(422)


def two_dimension_harmonic_oscillator(params):
    n = int(extract_param(params, 'n'))
    mass = int(extract_param(params, 'm'))
    spring_constant = int(extract_param(params, 'k'))
    t_lower = float(extract_param(params, 't-lower'))
    t_upper = float(extract_param(params, 't-upper'))
    order_of_integrator = int(extract_param(params, 'order-of-integrator'))
    initial_x = float(extract_param(params, 'initial-x'))
    initial_y = float(extract_param(params, 'initial-y'))
    initial_x_momentum = float(extract_param(params, 'initial-x-momentum'))
    initial_y_momentum = float(extract_param(params, 'initial-y-momentum'))

    integrator = GalerkinGaussLobattoIntegrator('t', ['x', 'y'], ['vx', 'vy'], order_of_integrator, verbose=True)

    # Define our properties and the Lagrangian for a spring
    m = mass
    k = spring_constant

    # Get symbols for use in Lagrangian
    vx, vy = integrator.symbols['v']
    x, y = integrator.symbols['q']

    # Define the Lagrangian for the system
    L = 0.5 * m * (vx * vx + vy * vy) - k * (x * x + y * y)

    # Define discretization parameters
    integrator.discretise(L, n, t_lower, t_upper)

    # Set the initial conditions for integration
    integrator.set_initial_conditions([initial_x, initial_y], [initial_x_momentum, initial_y_momentum])

    # Integrate the system
    integrator.integrate()

    x_solutions = []
    y_solutions = []
    px_solutions = []
    py_solutions = []

    for sol in integrator.q_solutions:
        x_solutions.append(sol[0])
        y_solutions.append(sol[1])

    for sol in integrator.p_solutions:
        px_solutions.append(sol[0])
        py_solutions.append(sol[1])

    return {
        't_list': integrator.t_list.tolist(),
        'x_solutions': x_solutions,
        'y_solutions': y_solutions,
        'px_solutions': px_solutions,
        'py_solutions': py_solutions
    }
