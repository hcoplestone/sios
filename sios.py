from typing import List
from assertions import Assertions
from prettytable import PrettyTable


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

        self.t = t
        self.q_list = q_list
        self.v_list = v_list
        self.verbose = verbose
        self.integrator_type = integrator_type

        self.validateIntegrator()

        if self.verbose:
            self.debug()

    def validateIntegrator(self) -> None:
        """
        Check that all the integrators parameters are as expected.
        """

        Assertions.assert_string(self.t, 'Symbol for time variable')
        Assertions.assert_list_of_strings(self.q_list, 'Generalised coordinates')
        Assertions.assert_list_of_strings(self.v_list, 'Generalised velocities')
        Assertions.assert_dimensions_match(self.q_list, 'Generalised coordinates', self.v_list, 'Generalised velocities')

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

    def display_integrator_details(self):
        """
        Print details about the type of integrator we are running.
        """

        print(f'Integrator type: {self.integrator_type}')

    def display_symbol_table(self) -> None:
        """
        Display a table of all the sympy variables in use.
        """

        st = PrettyTable()
        st.title = 'Phase space coordinates'

        st.field_names = ["Generalised coordinate symbol", "Corresponding velocity symbol$"]
        for i in range(0, len(self.q_list)):
            st.add_row([self.q_list[i], self.v_list[i]])

        print(st)


class GalerkinGaussLobatto(Integrator):

    def __init__(self, t: str, q_list: List[str], v_list: List[str], verbose: bool = False) -> None:
        """
        We subclass from a base integrator class.
        This way we can abstract away common functionality between integrators.
        """

        Integrator.__init__(self, t, q_list, v_list, verbose, 'Galerkin Gauss Lobatto')


def main():
    """
    Run the integrator :)
    :return:
    """
    i = GalerkinGaussLobatto('t', ['q1', 'q2'], ['v1', 'v2'], True)


if __name__ == "__main__":
    main()
