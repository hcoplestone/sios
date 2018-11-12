from typing import List
from assertions import Assertions
from prettytable import PrettyTable
from sympy import *


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
        self.q_list = q_list
        self.v_list = v_list
        self.integrator_type = integrator_type

        self.symbols = {}
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
