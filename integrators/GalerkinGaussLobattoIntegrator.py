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
        :param time_step: The fixed time interval the Gauss-Lobatto quadrature uses.
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
        :return action: Numerical value of S = \int_{t_n}^{t+time_step} L(t, q_n, [q_interiors], q_n_plus_1) dt
        """
        # Scale the Gauss Lobatto quadrature to the interval [t_n, t_n_plus_1]
        # This adjusts the times to sample at and the weightings (which should be invariant interval-to-interval
        # as the time step is fixed).
        # TODO: Only scale weightings once. Maybe add method to GGL quadrature scale_times_to_interval?
        # TODO: And have separate method scale_weights_to_interval which we call at beginning of integrate?
        t_n_plus_1 = t + time_step
        self.gauss_lobatto_quadrature.scale_to_interval(t, t_n_plus_1)

        # Function returned accepts arguments (t, q, q1, q2..., v, v1, v2...)
        lagrangian_evaluator = self.get_expression_evaluator()

        # The points to be enumerated over for the quadrature method
        points = [q_n] + q_interior_points + [q_n_plus_1]

        # Determine the velocities for each point in the interval
        velocities = self.determine_velocities(points)

        action = 0.0

        # Calculate S = \sum w_i * L(t_i, q_i, v_i)
        for index, weight in enumerate(self.gauss_lobatto_quadrature.scaled_weights):
            scaled_t = self.gauss_lobatto_quadrature.scaled_points[index]

            point_in_interval = points[index]

            action += weight * lagrangian_evaluator(scaled_t, *point_in_interval, *velocities[index])

        return action

    def determine_velocities(self, points):
        """
        Determines the velocities for every point in quadrature interval
        :param points: Array of vector points in quadrature interval = [\vec{point_1}, \vec{point_2}, ...]
        :return: Array of vectors describing velocity at each point:
        i.e. returns velocities = [\vec{velocity_at_point_1}, \vec{velocity_at_point_2}, ...]
        """

        velocities = []

        for i, point in enumerate(points):
            v = np.zeros(len(self.q_list))
            for j in range(0, self.order_of_integrator + 2):
                v = np.add(v, self.D[i][j] * points[j])
            velocities.append(v)

        return velocities

    def get_list_of_interior_points(self, points):
        """
        This function is necessary because root finder takes guess of form x0 = (\vec{point_1}, \vec{point_2},...)
        and transforms this input into (point_1_DOF_1, point_1_DOF_2, point_2_DOF_1, point_2_DOF_2) when passed
        into the function which returns the set of equations we would like to solve.

        Thus, this function (re)constructs a list of vector interior points from this transformed representation.
        This is used in two places:
        (a) In the function that defines the set of equations we want to solve.
        (b) After the root finder method has been called. The solution list is defined in this flattened structure.

        :param points: Concatenated lists of generalised coordinates for interior points and right hand endpoint.
        So if we have a quadrature interval with 2 interior points, with a system with 2 degrees of freedom (DOF),
        these points are represented by the following array:
        points = [interior_point_1_DOF_1, interior_point_1_DOF_2, interior_point_2_DOF_1, interior_point_2_DOF_2,
        q_n_plus_1_DOF_1, q_n_plus_1_DOF_2]
        :return: List of vectorised interior points: [\vec{interior_point_1}, \vec{interior_point_2}, ...]
        """
        interior_points = points[:-len(self.q_list)]
        interior_points_chunked = [interior_points[i:i + len(self.q_list)] for i in
                                   range(0, len(interior_points), len(self.q_list))]
        return interior_points_chunked

    def get_right_hand_exterior_point(self, points):
        """
        Given a set of points described by:
        [interior_1_DOF_1, interior_1_DOF_2, interior_2_DOF_1, interior_2_DOF_2, RH_exterior_DOF_1, RH_exterior_DOF_2]
        (where RH="Right Hand")
        we extract the vector describing the right hand exterior point \vec{exterior} === \vec{q_n_plus_1}

        Motivation is identical to that described in docstring of get_list_of_interior_points.

        :param points: Concatenated lists of generalised coordinates for interior points and right hand endpoint.
        So if we have a quadrature interval with 2 interior points, with a system with 2 degrees of freedom (DOF),
        these points are represented by the following array:
        points = [interior_point_1_DOF_1, interior_point_1_DOF_2, interior_point_2_DOF_1, interior_point_2_DOF_2,
        q_n_plus_1_DOF_1, q_n_plus_1_DOF_2]
        :return: Vector of right hand exterior point \vec{q_n_plus_1}
        """
        return points[-len(self.q_list):]

    def integrate(self):
        """
        Numerically integrate the system.
        """

        # Setup solutions with initial values
        self.setup_solutions()

        # Determine the fixed time step interval
        time_step = self.t_list[1] - self.t_list[0]

        # Determine the derivative matrix for this interval
        self.calculate_derivative_matrix(time_step)

        # Visually track progress of integration
        if self.verbose:
            print("\nIterating...")

        # Determine system piecewise
        for i in range(self.n - 1):
            t = self.t_list[i]

            if self.verbose:
                print('.', end='', flush=True)

            def new_position_from_nth_solution_equation(points):
                """
                :param points: concatenated array of generalised coordinates of trial vector points
                [q_interior_1, q_interior_2, ..., q_n_plus_1]
                """
                list_of_equations = []

                list_of_interior_points = self.get_list_of_interior_points(points)
                q_n_plus_1_trial_solution = self.get_right_hand_exterior_point(points)

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
            self.q_solutions[i + 1] = self.get_right_hand_exterior_point(solutions.x)

            self.p_solutions[i + 1] = determine_new_momentum_from_q_n_plus_1th_solution(q_interior_points)

        if self.verbose:
            print("\nIntegration complete!")
