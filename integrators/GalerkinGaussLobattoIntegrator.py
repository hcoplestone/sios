from typing import List
from .Integrator import Integrator
import autograd.numpy as np
import numpy as np_reg
from scipy import optimize
from autograd import elementwise_grad as egrad

from .quadrature import GaussLobattoQuadrature


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

    def calculate_derivative_matrix(self, time_step):
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
                P_r_plus_1 = np_reg.polynomial.legendre.Legendre([0 for i in range(0, r)] + [1])
                x_i = self.gauss_lobatto_quadrature.points[i]
                x_j = self.gauss_lobatto_quadrature.points[j]

                numerator = 2 * P_r_plus_1(x_i)
                denominator = P_r_plus_1(x_j) * (x_i - x_j) * time_step

                if i != j:
                    self.D[i][j] = numerator / denominator

    # def action(self, q_n, q_n_plus_1, t, time_step):
    def action(self, t, time_step, q_n, q_interior, q_n_plus_1):
        t_n_plus_1 = t + time_step
        self.gauss_lobatto_quadrature.scale_to_interval(t, t_n_plus_1)

        # Function returned accepts arguments (t, q, q1, q2..., v, v1, v2...)
        lagrangian_evaluator = self.get_expression_evaluator()

        action = 0.0

        points = [q_n, q_interior, q_n_plus_1]
        velocities = self.determine_velocities(points, time_step)

        for index, weight in enumerate(self.gauss_lobatto_quadrature.scaled_weights):
            scaled_t = self.gauss_lobatto_quadrature.scaled_points[index]

            point_in_interval = points[index]

            action += weight * lagrangian_evaluator(scaled_t, *point_in_interval, *velocities[index])

        return action

    def determine_velocities(self, points, time_step):
        """
        Determines the velocity of the Legendre trajectory at a point in phase space.
        :param points: Array of all points in interval
        :param time_step: The time step used for the piecewise integration
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

            def new_position_from_nth_solution_equation(point_guesses):
                q_n_interior_point_trial_solution = point_guesses[0:len(self.q_list)]
                q_n_plus_1_trial_solution = point_guesses[len(self.q_list):2*len(self.q_list)]

                S_of_n = lambda q_n: self.action(t, time_step, q_n, q_n_interior_point_trial_solution,
                                                 q_n_plus_1_trial_solution)

                S_of_interior = lambda q_interior: self.action(t, time_step, self.q_solutions[i], q_interior,
                                                               q_n_plus_1_trial_solution)

                partial_differential_of_action_wrt_interior_point = egrad(S_of_interior)
                interior_equation = partial_differential_of_action_wrt_interior_point(q_n_interior_point_trial_solution)

                partial_differential_of_action_wrt_q_n = egrad(S_of_n)
                conservation_equation = np.add(self.p_solutions[i],
                                               partial_differential_of_action_wrt_q_n(self.q_solutions[i]))

                return np.concatenate((interior_equation, conservation_equation))

            def determine_new_momentum_from_q_n_plus_1th_solution(interior_point):
                S = lambda q_n_plus_1: self.action(t, time_step, self.q_solutions[i], interior_point, q_n_plus_1)
                partial_differential_of_action_wrt_q_n_plus_1 = egrad(S)
                return partial_differential_of_action_wrt_q_n_plus_1(self.q_solutions[i + 1])

            if (i > 1):
                q_nplus1_guess = self.q_solutions[i] + (self.q_solutions[i] - self.q_solutions[i - 1])
            else:
                q_nplus1_guess = self.q_solutions[i]

            q_i_guess = q_nplus1_guess

            point_guesses = np.concatenate((q_i_guess, q_nplus1_guess), axis=0)

            root_solutions = optimize.root(new_position_from_nth_solution_equation, point_guesses)

            self.q_solutions[i + 1] = root_solutions.x[len(self.q_list):2*len(self.q_list)]
            interior_point = root_solutions.x[0:len(self.q_list)]

            self.p_solutions[i + 1] = determine_new_momentum_from_q_n_plus_1th_solution(interior_point)

        if self.verbose:
            print("\nIntegration complete!")
