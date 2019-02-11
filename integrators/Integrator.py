from typing import List
from assertions import Assertions
from prettytable import PrettyTable
from sympy import *
import autograd.numpy as np


class Integrator:

    def __init__(self, t: str, q_list: List[str], v_list: List[str], verbose: bool, integrator_type: str) -> None:
        """
        Implement common logic for all variational integrators we may want to build.

       :param t: What symbol to use as the independent time variable
       :param q_list: List of strings for generating symbols for $q$
       :param v_list: List of strings for generating symbols for $\dot{q}$
       :param verbose: Boolean flag toggling verbose output for debugging or showing off when running the code!
       :param type: The type of integrator that we are running e.g. 'Galerkin Gauss Lobatto'
       """

        # Instance variable declarations
        self.t = t

        # q variables - symbols and initial values
        self.q_list = q_list
        self.q_initial_value_list = None

        # v = $\dot{q}$ variables - symbols and initial values
        self.v_list = v_list
        self.v_initial_value_list = None

        self.t_list = None
        self.q_solutions = []
        self.v_solutions = []

        self.integrator_type = integrator_type
        self.verbose = verbose

        self.symbols = {}

        self.n = None
        # END instance variable declarations

        # Validate the integrator and run all pre-integration prep
        self.validate_integrator()
        self.setup_integrator()

        # Print debug information is requested
        if verbose:
            self.debug()

    def validate_integrator(self) -> None:
        """
        Check that all the integrators parameters are as expected.
        """

        Assertions.assert_string(self.t, 'Symbol for time variable')
        Assertions.assert_list_of_strings(self.q_list, 'Generalised coordinates')
        Assertions.assert_list_of_strings(self.v_list, 'Generalised velocities')
        Assertions.assert_dimensions_match(self.q_list, 'Generalised coordinates', self.v_list,
                                           'Generalised velocities')

    def setup_integrator(self) -> None:
        """
        Run all pre integration setup.
        """
        self.parse_symbols()

    def parse_symbols(self) -> None:
        """
        Convert list of string symbols to Sympy symbols.
        """
        self.symbols['t'] = Symbol(self.t, real=True)
        self.symbols['q'] = [Symbol(q, real=True) for q in self.q_list]
        self.symbols['v'] = [Symbol(v, real=True) for v in self.v_list]

    def set_expression(self, expression) -> None:
        """
        Set the expression that we are going to be integrating.
        :param expression: The expression that we are going to be integrating.
        """
        self.expression = expression
        if self.verbose:
            print("The expression we are integrating is:")
            print(f"L = {self.expression}\n")

    def get_expression_evaluator(self):
        """
        Return the pythonic function equivalent of the sympy expression we are integrating.
        Function returned accepts arguments (t, q, q1, q2..., v, v1, v2...)
        :return: function(t, q variables..., v variables...)
        """
        return lambdify(tuple([self.symbols['t']] + self.symbols['q'] + self.symbols['v']), self.expression, modules=np)

    def set_time_boundaries(self, t_lim_lower: float, t_lim_upper: float) -> None:
        """
        Set the boundaries in time for the integration.
        :param t_lim_lower: Lower time integration limit.
        :param t_lim_upper: Upper time integration limit.
        """
        Assertions.assert_float(t_lim_lower, 't variable lower limit')
        Assertions.assert_float(t_lim_upper, 't variable upper limit')
        self.t_lim_lower = t_lim_lower
        self.t_lim_upper = t_lim_upper

    # def determine_time_step(self, n: int):
    #     """
    #     :param n: Number of discretisation points to use.
    #     """
    #     Assertions.assert_integer(n, 'Number of discretisation points')
    #     self.n = n
    #     self.dt = (self.t_lim_upper - self.t_lim_lower)/self.n

    def debug(self) -> None:
        """
        Print all necessary data for debugging.
        """
        print()
        print('SIOS: Slimplectic Integrator on Steroids')
        print('-----------------------------------------')
        print()
        self.display_integrator_details()
        print()
        self.display_symbol_table()
        print()

    def display_integrator_details(self):
        """
        Print details about the type of integrator we are running.
        """

        print(f'Integrator type: {self.integrator_type}')
        print(f'Independent integration variable: {self.t}')

    def display_symbol_table(self) -> None:
        """
        Display a table of all the sympy variables in use.
        """

        st = PrettyTable()
        st.title = 'Phase Space Coordinates'

        st.field_names = ["Generalised coordinate symbol", "Corresponding velocity symbol$"]
        for i in range(0, len(self.q_list)):
            st.add_row([self.q_list[i], self.v_list[i]])

        print(st)

    def display_solutions(self) -> None:
        """
        Display a table of all the position and momenta solutions
        """

        st = PrettyTable()
        st.title = 'Solutions'

        st.field_names = ["t", "q solution list", "v solution list"]
        for i in range(0, len(self.t_list)):
            st.add_row([
                round(self.t_list[i], 2),
                str(self.q_solutions[i]),
                str(self.v_solutions[i])
            ])

        print(st)
