from typing import List
from .Integrator import Integrator
import autograd.numpy as np
from scipy import optimize
import scipy.special as sp
from autograd import elementwise_grad as egrad

from .quadrature import GaussLobattoQuadrature, FirstOrderQuadrature


class GalerkinGaussLobattoIntegrator(Integrator):

    def __init__(self, t: str, q_list: List[str], v_list: List[str], order_of_integrator: int = 1,
                 verbose: bool = False) -> None:
        """
        We subclass from a base integrator class.
        This way we can abstract away common functionality between integrators.
        """

        Integrator.__init__(self, t, q_list, v_list, verbose, 'Galerkin Gauss Lobatto Integrator')
        self.order_of_integrator = order_of_integrator
        self.gauss_lobatto_quadrature = GaussLobattoQuadrature(self.order_of_integrator + 2, False)

        self.D = None

    def calculate_derivative_matrix(self, time_step) -> None:
        """
        Calculate the derivative matrix for a given fixed time step.
        This is used to calculate velocities of a Legendre path, given the value of each quadrature point
        in the interval [t_n, t_n+time_step].
        :param time_step:
        """
        # r is the order of the integrator
        r = self.order_of_integrator

        # Our derivative matrix is n * n, where n is the number of quadrature points to use.
        self.D = np.zeros((r + 2, r + 2))

        # Dij = -(r+1)(r+2)/(2*delta_t) for i = j = 0
        self.D[0][0] = -1 * (r + 1) * (r + 2) / (2 * time_step)

        # Dij = (r+1)(r+2)/(2*delta_t) for i = j = r+1
        self.D[r + 1][r + 1] = -1 * self.D[0][0]

        # Dij = 2*P_{r+1}(x_j) / P_{r+1}(x_j)*(x_i - x_j)*(delta_t)
        for i in range(0, r + 2):
            for j in range(0, r + 2):
                P_r_plus_1 = sp.legendre(r + 1)
                x_i = self.gauss_lobatto_quadrature.points[i]
                x_j = self.gauss_lobatto_quadrature.points[j]

                numerator = 2 * P_r_plus_1(x_i)
                denominator = P_r_plus_1(x_j) * (x_i - x_j) * time_step

                if i != j:
                    self.D[i][j] = numerator / denominator

    def action(self, t, time_step, q_n, q_interior_points, q_n_plus_1):
        """
        Here we use Gauss-Lobatto quadrature to approximate the action integral for t \in [t, t+time_step]
        with quadrature points [q_n] + q_interior_points + [q_n_plus_1] and velocities given by application
        of the derivative matrix to these set of points.

        :param t: The value of t at the beginning of the time interval we are integrating over.
        i.e. we integrate over the region [t, t+time_step]
        :param time_step: The fixed time step we use piecewise-integrate the system.
        :param q_n: Vector describing the initial point in phase space: \vec{q_n} = [GC1, GC2, ...] where GCi is
        value of generalised coordinate at this point in phase space.
        :param q_interior_points: array of vector interior points [\vec{q_interior_1}, \vec{q_interior_2}, ...]
        :param q_n_plus_1: Vector describing the final point in phase space
        """
        t_n_plus_1 = t + time_step
        self.gauss_lobatto_quadrature.scale_to_interval(t, t_n_plus_1)

        # Function returned accepts arguments (t, q, q1, q2..., v, v1, v2...)
        lagrangian_evaluator = self.get_expression_evaluator()

        action = 0.0

        points = [q_n] + q_interior_points + [q_n_plus_1]

        velocities = self.determine_velocities(points)

        for index, weight in enumerate(self.gauss_lobatto_quadrature.scaled_weights):
            scaled_t = self.gauss_lobatto_quadrature.scaled_points[index]

            point_in_interval = points[index]

            # action += weight * lagrangian_evaluator(scaled_t, *point_in_interval, *velocities[index])
            action += weight * lagrangian_evaluator(scaled_t, *point_in_interval, *velocities[index])

        return action

        # Function returned accepts arguments (t, q, q1, q2..., v, v1, v2...)
        # lagrangian_evaluator = self.get_expression_evaluator()
        # ddt = 0.5*time_step
        #
        # v_n = (q_interior - q_n) / ddt
        # lagrangian_evaled_at_n = lagrangian_evaluator(t, *q_n, *v_n)
        # lagrangian_evaled_at_midpoint = lagrangian_evaluator(t+ddt, *q_interior, *v_n)
        #
        # v_n_plus_1 = (q_n_plus_1 - q_interior) / ddt
        # t_n_plus_1 = t + time_step
        # lagrangian_evaled_at_n_plus_1 = lagrangian_evaluator(t_n_plus_1, *q_n_plus_1, *v_n_plus_1)
        #
        # action = np.add(FirstOrderQuadrature.trapezium_rule(lagrangian_evaled_at_n, lagrangian_evaled_at_midpoint, ddt),
        #                 FirstOrderQuadrature.trapezium_rule(lagrangian_evaled_at_midpoint, lagrangian_evaled_at_n_plus_1, ddt))
        #
        # print("Action is:")
        # print(action)

        # return action

    def determine_velocities(self, points):
        """
        Determines the velocity of the Legendre trajectory at a point in phase space.
        :param points: Array of all points in interval
        """

        # return self.D * points
        # return [points[0] - points[0]]/time_step

        velocities = []

        for i, point in enumerate(points):
            v = np.zeros(len(self.q_list))
            for j in range(0, self.order_of_integrator + 2):
                v = np.add(v, self.D[i][j] * points[j])
            velocities.append(v)

        return velocities

    def get_list_of_interior_points(self, points):
        interior_points = points[:-len(self.q_list)]
        interior_points_chunked = [interior_points[i:i + len(self.q_list)] for i in
                                   range(0, len(interior_points), len(self.q_list))]
        return interior_points_chunked

    def integrate(self):
        """
        Numerically integrate the system.
        """

        # Setup solutions with initial values
        self.setup_solutions()

        time_step = self.t_list[1] - self.t_list[0]
        self.calculate_derivative_matrix(time_step)

        # Iterate
        if self.verbose:
            print("\nIterating...")

        for i in range(self.n - 1):
            t = self.t_list[i]
            t_next = self.t_list[i + 1]
            # time_step = t_next - t

            if self.verbose:
                print('.', end='', flush=True)

            def new_position_from_nth_solution_equation(points):
                """
                :param points: array of trial vector points [q_interior_1, q_interior_2, ..., q_n_plus_1]
                :return:
                """
                list_of_equations = []

                list_of_interior_points = self.get_list_of_interior_points(points)
                q_n_plus_1_trial_solution = points[-len(self.q_list):]

                for index, interior_point in enumerate(list_of_interior_points):
                    def interior_point_argument_for_action(point_to_differentiate_wrt_to):
                        return list_of_interior_points[0:index] + [point_to_differentiate_wrt_to] \
                               + list_of_interior_points[index + 1:]

                    s_of_interior_point = lambda q_interior: self.action(t, time_step, self.q_solutions[i],
                                                                         interior_point_argument_for_action(q_interior),
                                                                         q_n_plus_1_trial_solution)

                    partial_differential_of_action_wrt_interior_point = egrad(s_of_interior_point)
                    interior_equation = partial_differential_of_action_wrt_interior_point(interior_point)
                    list_of_equations.append(interior_equation)

                s_of_n = lambda q_n: self.action(t, time_step, q_n, list_of_interior_points,
                                                 q_n_plus_1_trial_solution)

                partial_differential_of_action_wrt_q_n = egrad(s_of_n)
                conservation_equation = np.add(self.p_solutions[i],
                                               partial_differential_of_action_wrt_q_n(self.q_solutions[i]))
                list_of_equations.append(conservation_equation)

                return np.concatenate(tuple(list_of_equations))

            def determine_new_momentum_from_q_n_plus_1th_solution(interior_points):
                S = lambda q_n_plus_1: self.action(t, time_step, self.q_solutions[i], interior_points, q_n_plus_1)
                partial_differential_of_action_wrt_q_n_plus_1 = egrad(S)
                return partial_differential_of_action_wrt_q_n_plus_1(self.q_solutions[i + 1])

            if (i > 1):
                q_nplus1_guess = self.q_solutions[i] + (self.q_solutions[i] - self.q_solutions[i - 1])
            else:
                q_nplus1_guess = self.q_solutions[i]

            q_i_guess = q_nplus1_guess
            point_guesses = [q_i_guess for i in range(self.order_of_integrator)]
            point_guesses.append(q_nplus1_guess)

            solutions = optimize.root(new_position_from_nth_solution_equation, point_guesses)

            # q_interior_solution = solutions.x[0:len(self.q_list)]
            q_interior_points = self.get_list_of_interior_points(solutions.x)
            self.q_solutions[i + 1] = solutions.x[-len(self.q_list):]

            self.p_solutions[i + 1] = determine_new_momentum_from_q_n_plus_1th_solution(q_interior_points)

        if self.verbose:
            print("\nIntegration complete!")
