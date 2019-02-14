from typing import List
from .Integrator import Integrator
import autograd.numpy as np
import numpy as numpy
from scipy import optimize
from autograd import elementwise_grad as egrad
from assertions import Assertions

from .quadrature import FirstOrderQuadrature


class PolynomialIntegrator(Integrator):

    def __init__(self, t: str, q_list: List[str], v_list: List[str], order_of_integrator: int,
                 verbose: bool = False) -> None:
        """
        We subclass from a base integrator class.
        This way we can abstract away common functionality between integrators.
        """

        Integrator.__init__(self, t, q_list, v_list, verbose, 'Polynomial Integrator')

        Assertions.assert_integer(order_of_integrator, 'Integrator order')
        self.order_of_integrator = order_of_integrator

    def evaluate_polynomial(self, x, coeffs):
        """
        Evaluates a polynomial at x
        :param coefficients: coefficients of powers of x in order of ascending power
        :param x: value of x at which we are evaluating the polynomial
        :return:
        """
        eval_result = 0
        for index, coefficient in enumerate(coeffs):
            eval_result = eval_result + coefficient * np.power(x, index)
        return eval_result

    def fit_polynomial(self, x_list, y_list, degree=2):
        """
        Fits a polynomial to supplied points
        :param x_list: x values
        :param y_list: y values
        :param degree: degree of polynomial
        :return: array of polynomial coefficients, order of ascending power
        """

        # print(y_list)
        # return [0, 0]
        # return numpy.polynomial.polynomial.polyfit(x_list, y_list, degree)
        return [y_list[-1] - x_list[0] * (y_list[-1] - y_list[0]) / (x_list[-1] - x_list[0]),
                (y_list[-1] - y_list[0]) / (x_list[-1] - x_list[0])]

    def evaluate_path(self, q_n, q_n_plus_1, t_lower, t_upper, t, is_evaluating_componentwise=False):
        # print(self.evaluate_path_derivative([0, 0], [1, 1], 0, 1, 1))
        if is_evaluating_componentwise:
            coefficients = self.fit_polynomial([t_lower, t_upper], [q_n, q_n_plus_1],
                                               self.order_of_integrator)
            return self.evaluate_polynomial(t, coefficients)
        else:
            evaled_polynomials = []
            coefficients = []
            for i in range(len(self.q_list)):
                ith_coefficients = self.fit_polynomial([t_lower, t_upper], [q_n[i], q_n_plus_1[i]],
                                                       self.order_of_integrator)
                coefficients.append(ith_coefficients)
            for c in coefficients:
                evaled_polynomials.append(self.evaluate_polynomial(t, c))
            return evaled_polynomials

        # return q_n + (q_n_plus_1 - q_n) * (t - t_lower) / (t_upper - t_lower)

    def evaluate_path_derivative(self, q_n, q_n_plus_1, t_lower, t_upper, t):
        """
        Could be made more efficient as evaluating path twice - but just a proof of
        concept for now.
        """
        # number_of_generalised_coordinates = len(self.q_list)
        # path_derivative = []
        #
        # for i in range(number_of_generalised_coordinates):
        #     path_of_ith_generalised_coordinate_as_func_of_t = lambda time: self.evaluate_path(q_n[i], q_n_plus_1[i],
        #                                                                                       t_lower, t_upper, time,
        #                                                                                       True)
        #     deriv_of_ith_component_of_path_evaluated_at_t = egrad(path_of_ith_generalised_coordinate_as_func_of_t)(t)
        #     path_derivative.append(deriv_of_ith_component_of_path_evaluated_at_t)

        # return path_derivative

        number_of_generalised_coordinates = len(self.q_list)
        path_derivative = []

        for i in range(number_of_generalised_coordinates):
            coefficients = self.fit_polynomial([t_lower, t_upper], [q_n[i], q_n_plus_1[i]],
                                               self.order_of_integrator)
            deriv_coeffs = []
            for index, c in enumerate(coefficients):
                if index == len(coefficients) - 1:
                    deriv_coeffs.append(0)
                else:
                    deriv_coeffs.append(coefficients[index + 1] * (index + 1))
            path_derivative.append(self.evaluate_polynomial(t, deriv_coeffs))

        return path_derivative

    def action(self, q_n, q_n_plus_1, t, time_step):
        t_lower = t
        t_upper = t + time_step

        # Function returned accepts arguments (t, q, q1, q2..., v, v1, v2...)
        lagrangian_evaluator = self.get_expression_evaluator()

        path_at_t_lower = self.evaluate_path(q_n, q_n_plus_1, t_lower, t_upper, t_lower)
        deriv_of_path_at_t_lower = self.evaluate_path_derivative(q_n, q_n_plus_1, t_lower, t_upper, t_lower)

        path_at_t_upper = self.evaluate_path(q_n, q_n_plus_1, t_lower, t_upper, t_upper)
        deriv_of_path_at_t_upper = self.evaluate_path_derivative(q_n, q_n_plus_1, t_lower, t_upper, t_upper)

        return FirstOrderQuadrature.trapezium_rule(
            lagrangian_evaluator(t_lower, *path_at_t_lower, *deriv_of_path_at_t_lower),
            lagrangian_evaluator(t_upper, *path_at_t_upper, *deriv_of_path_at_t_upper),
            time_step
        )

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

            q_nplus1_solution = optimize.root(new_position_from_nth_solution_equation, q_nplus1_guess,
                                              method='hybr')
            self.q_solutions[i + 1] = q_nplus1_solution.x

            self.p_solutions[i + 1] = determine_new_momentum_from_q_n_plus_1th_solution()

        if self.verbose:
            print("\nIntegration complete!")
