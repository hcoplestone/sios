from typing import List
from .Integrator import Integrator
import autograd.numpy as np
from scipy import optimize
from autograd import elementwise_grad as egrad

from .quadrature import FirstOrderQuadrature
from progress.bar import IncrementalBar


class FirstOrderIntegrator(Integrator):

    def __init__(self, t: str, q_list: List[str], v_list: List[str], verbose: bool = False) -> None:
        """
        We subclass from a base integrator class.
        This way we can abstract away common functionality between integrators.
        """

        Integrator.__init__(self, t, q_list, v_list, verbose, 'First Order Integrator')

    def action(self, q_n, q_n_plus_1, t, time_step):
        # Function returned accepts arguments (t, q, q1, q2..., v, v1, v2...)
        lagrangian_evaluator = self.get_expression_evaluator()

        v_n = (q_n_plus_1 - q_n) / time_step
        lagrangian_evaled_at_n = lagrangian_evaluator(t, *q_n, *v_n)

        v_n_plus_1 = (q_n_plus_1 - q_n) / time_step
        t_n_plus_1 = t + time_step
        lagrangian_evaled_at_n_plus_1 = lagrangian_evaluator(t_n_plus_1, *q_n_plus_1, *v_n_plus_1)

        action = FirstOrderQuadrature.trapezium_rule(lagrangian_evaled_at_n, lagrangian_evaled_at_n_plus_1, time_step)
        return action

    def integrate(self, should_shoot: bool = True):
        """
        Numerically integrate the system.
        :param should_shoot: Boolean to determine if shooting method should be used to guess q_{n+3} from q_{n+2} and q_{n+1}.
        """

        # Setup solutions with initial values
        self.setup_solutions()

        # Iterate
        if self.verbose:
            bar = IncrementalBar('Iterating', max=self.n)
            bar.next()

        for i in range(self.n - 1):
            time_step = self.t_list[i + 1] - self.t_list[i]
            t = self.t_list[i + 1]

            # Visually track progress of integration
            if self.verbose:
                # print('.', end='', flush=True)
                bar.next()

            def new_position_from_nth_solution_equation(q_n_plus_1_trial_solutions):
                S = lambda q_n: self.action(q_n, q_n_plus_1_trial_solutions, t, time_step)
                partial_differential_of_action_wrt_q_n = egrad(S)
                equation = np.add(self.p_solutions[i], partial_differential_of_action_wrt_q_n(self.q_solutions[i]))
                return equation

            def determine_new_momentum_from_q_n_plus_1th_solution():
                S = lambda q_n_plus_1: self.action(self.q_solutions[i], q_n_plus_1, t, time_step)
                partial_differential_of_action_wrt_q_n_plus_1 = egrad(S)
                return partial_differential_of_action_wrt_q_n_plus_1(self.q_solutions[i + 1])

            if i > 1 and should_shoot:
                q_nplus1_guess = self.q_solutions[i] + (self.q_solutions[i] - self.q_solutions[i - 1])
            else:
                q_nplus1_guess = self.q_solutions[i]

            q_nplus1_solution = optimize.root(new_position_from_nth_solution_equation, q_nplus1_guess, method='hybr')
            self.q_solutions[i + 1] = q_nplus1_solution.x

            self.p_solutions[i + 1] = determine_new_momentum_from_q_n_plus_1th_solution()

        # Let the user know that the integration is finished
        if self.verbose:
            bar.finish()
            print("\nIntegration complete!")
