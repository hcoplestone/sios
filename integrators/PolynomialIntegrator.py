from typing import List
from .Integrator import Integrator
import autograd.numpy as np
from scipy import optimize
from autograd import elementwise_grad as egrad
from assertions import Assertions
import matplotlib.pyplot as plt
import copy

from .quadrature import FirstOrderQuadrature


# np.numpy_boxes.ArrayBox.__repr__ = lambda self: str(self._value)

class PolynomialIntegrator(Integrator):

    def __init__(self, t: str, q_list: List[str], v_list: List[str], order_of_integrator: int = 2,
                 verbose: bool = False) -> None:
        """
        We subclass from a base integrator class.
        This way we can abstract away common functionality between integrators.
        """

        Integrator.__init__(self, t, q_list, v_list, verbose, 'Polynomial Integrator')
        Assertions.assert_integer(order_of_integrator, "Order of integrator")
        self.order_of_integrator = order_of_integrator

    def evaluate_polynomial(self, coeffs, t):
        value = 0
        for index, coeff in enumerate(coeffs):
            value = value + coeff * np.power(t, index)
        return value

    def interpolate_path(self, q_n_i, q_n_plus_1_i, t_lower, t_upper, t):
        """
        Interpolate the path of a single q degree of freedom.
        This function should be evaluated for each DOF in the solution separately.
        """

        if hasattr(q_n_i, '_value'):
            q_n_i_unpacked = q_n_i._value
            # q_n_i_unpacked = copy.copy(q_n_i._value)
        else:
            q_n_i_unpacked = q_n_i
            # q_n_i_unpacked = copy.copy(q_n_i)

        if hasattr(q_n_plus_1_i, '_value'):
            q_n_plus_1_i_unpacked = q_n_plus_1_i._value
            # q_n_plus_1_i_unpacked = copy.copy(q_n_plus_1_i._value)
        else:
            q_n_plus_1_i_unpacked = q_n_plus_1_i

        # print("\nq_n_i: {0}\nq_n_plus_1_i: {1}\nt_lower: {2}\nt_upper: {3}".format(type(q_n_i), type(q_n_plus_1_i), type(t_lower), type(t_upper)))

        # coefficients = np.flip(
        #     np.polyfit([t_lower, t_upper], [q_n_i_unpacked, q_n_plus_1_i_unpacked], self.order_of_integrator))
        # return self.evaluate_polynomial(coefficients, t)


        def sum_of_residuals_squared(coefficients_guess):
            ans = 0
            x = [t_lower, t_upper]
            y = [q_n_i_unpacked, q_n_plus_1_i_unpacked]
            for index, point in enumerate(x):
                print(coefficients_guess)
                residual = y[index] - self.evaluate_polynomial(coefficients_guess, point)
                residual_squared = np.power(residual, 2)
                ans = ans + residual_squared
            return ans

        def func_to_minimise(coefficients_guess):
            derivative_of_sum_of_residuals_squared = egrad(sum_of_residuals_squared)
            return derivative_of_sum_of_residuals_squared(coefficients_guess)

        coefficients_guess = [1.0,1.0]
        coefficients = optimize.root(func_to_minimise, coefficients_guess).x

        #
        ts = np.linspace(t_lower, t_upper, 200)
        plt.plot([t_lower, t_upper], [q_n_i_unpacked, q_n_plus_1_i_unpacked], 'o')
        plt.plot(ts, [self.evaluate_polynomial(coefficients, tt) for tt in ts])

        return self.evaluate_polynomial(coefficients, t)
        path = q_n_i + (q_n_plus_1_i - q_n_i) * (t - t_lower) / (t_upper - t_lower)
        return path

    def time_derivative_of_interpolatated_path(self, q_n_i, q_n_plus_1_i, t_lower, t_upper, t):
        """
        Interpolate the path of a single q degree of freedom.
        This function should be evaluated for each DOF in the solution separately.
        """

        # path = q_n_i + (q_n_plus_1_i - q_n_i) * 1.0 / (t_upper - t_lower)
        # return path

        if hasattr(q_n_i, '_value'):
            q_n_i_unpacked = q_n_i._value
        else:
            q_n_i_unpacked = q_n_i

        if hasattr(q_n_plus_1_i, '_value'):
            q_n_plus_1_i_unpacked = q_n_plus_1_i._value
        else:
            q_n_plus_1_i_unpacked = q_n_plus_1_i

        coefficients = np.flip(
            np.polyfit([t_lower, t_upper], [q_n_i_unpacked, q_n_plus_1_i_unpacked], self.order_of_integrator))
        coefficients_of_differentiated_path = [0 for i in coefficients]
        for index, c in enumerate(coefficients):
            if index < len(coefficients) - 1:
                coefficients_of_differentiated_path[index] = coefficients[index + 1] * (index + 1)
            else:
                coefficients_of_differentiated_path[index] = 0

        return self.evaluate_polynomial(coefficients_of_differentiated_path, t)

    def action(self, q_n, q_n_plus_1, t, time_step):
        t_lower = t
        t_upper = t + time_step

        # Function returned accepts arguments (t, q, q1, q2..., v, v1, v2...)
        lagrangian_evaluator = self.get_expression_evaluator()

        def path(time):
            components = []
            for i in range(len(self.q_list)):
                component = self.interpolate_path(q_n[i], q_n_plus_1[i], t_lower, t_upper, time)
                components.append(component)
            return components

        def time_derivative_of_path(time):
            components = []
            for i in range(len(self.q_list)):
                component = self.time_derivative_of_interpolatated_path(q_n[i], q_n_plus_1[i], t_lower, t_upper, time)
                components.append(component)
            return components

        N = 5
        h = float(t_upper - t_lower) / N
        t_sample = t_lower
        s = 0.0
        s = s + (lagrangian_evaluator(t_sample, *path(t_sample), *time_derivative_of_path(t_sample)) / 2.0)
        for i in range(1, N):
            t_sample = t_sample + h
            s = s + lagrangian_evaluator(t_sample, *path(t_sample), *time_derivative_of_path(t_sample))
        t_sample = t_sample + h
        s = s + (lagrangian_evaluator(t_sample, *path(t_sample), *time_derivative_of_path(t_sample)) / 2.0)

        return s * h

    def integrate(self):
        """
        Numerically integrate the system.
        """

        # Setup solutions with initial values
        self.setup_solutions()

        # Iterate
        if self.verbose:
            print("\nIterating...")

        for i in range(self.n - 1):
            time_step = self.t_list[i + 1] - self.t_list[i]
            t = self.t_list[i]
            if self.verbose:
                print('.', end='', flush=True)

            def new_position_from_nth_solution_equation(q_n_plus_1_trial_solutions):
                S = lambda q_n: self.action(q_n, q_n_plus_1_trial_solutions, t, time_step)
                partial_differential_of_action_wrt_q_n = egrad(S)
                equation = np.add(self.p_solutions[i], partial_differential_of_action_wrt_q_n(self.q_solutions[i]))
                return equation

            def determine_new_momentum_from_q_n_plus_1th_solution():
                S = lambda q_n_plus_1: self.action(self.q_solutions[i], q_n_plus_1, t, time_step)
                partial_differential_of_action_wrt_q_n_plus_1 = egrad(S)
                return partial_differential_of_action_wrt_q_n_plus_1(self.q_solutions[i + 1])

            if (i > 1):
                q_nplus1_guess = self.q_solutions[i] + (self.q_solutions[i] - self.q_solutions[i - 1])
            else:
                q_nplus1_guess = self.q_solutions[i]

            q_nplus1_solution = optimize.root(new_position_from_nth_solution_equation, q_nplus1_guess, method='hybr')
            plt.show()
            self.q_solutions[i + 1] = q_nplus1_solution.x

            self.p_solutions[i + 1] = determine_new_momentum_from_q_n_plus_1th_solution()

        if self.verbose:
            print("\nIntegration complete!")
