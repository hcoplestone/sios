from typing import List
from assertions import Assertions
from prettytable import PrettyTable
from sympy import *
import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


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

        # v = $\dot{q}$ variables - symbols
        self.v_list = v_list

        # Momenta
        self.p_initial_value_list = None

        # The expression we are integrating
        self.expression = None

        self.t_list = None
        self.q_solutions = []
        self.p_solutions = []

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

    def set_initial_conditions(self, q_initial_value_list: List[float], p_initial_value_list: List[float]):
        """
        Set the initial conditions for the integrator.
        :param q_initial_value_list:  List of initial q values.
        :param p_initial_value_list: List of initial v = $\dot{q}$ values.
        """

        Assertions.assert_list_of_floats(q_initial_value_list, 'Initial q values')
        Assertions.assert_dimensions_match(self.q_list, 'q variables', q_initial_value_list, 'Initial q values')

        Assertions.assert_list_of_floats(p_initial_value_list, 'Initial p values')
        Assertions.assert_dimensions_match(self.q_list, 'q variables', p_initial_value_list, 'Initial p values')

        self.q_initial_value_list = q_initial_value_list
        self.p_initial_value_list = p_initial_value_list

    def discretise(self, expression, n: int, t_lim_lower: float, t_lim_upper: float) -> None:
        """
        Discretise the function that we provide on an interval [t_lim_lower, t_lim_upper].

        :param expression: Sympy expression for the function we want to discretise.
        :param n: The number of quadrature points to use.
        :param t_lim_lower: Lower time limit to sample our continuous function over.
        :param t_lim_upper: Upper time limit to sample our continuous function over.
        """

        Assertions.assert_integer(n, 'number of quadrature points')

        self.set_time_boundaries(t_lim_lower, t_lim_upper)
        self.n = n

        self.t_list = np.linspace(t_lim_lower, t_lim_upper, n)
        self.set_expression(expression)

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

    def setup_solutions(self) -> None:
        """
        Initialise the solution arrays with zeros.
        :return:
        """

        # TODO: assert q_list and p_list not empty
        self.q_solutions = [np.zeros(len(self.q_list)) for i in range(self.n)]
        self.p_solutions = [np.zeros(len(self.q_list)) for i in range(self.n)]

    def display_solutions(self) -> None:
        """
        Display a table of all the position and momenta solutions
        """

        print()
        st = PrettyTable()
        st.title = 'Solutions'

        # st.field_names = ["t", "q solution list", "p solution list"]
        st.field_names = ["t", "[" + " ".join(self.q_list) + "]", "Momenta of [" + " ".join(self.q_list) + "]"]
        for i in range(0, len(self.t_list)):
            st.add_row([
                round(self.t_list[i], 2),
                str(self.q_solutions[i]),
                str(self.p_solutions[i])
            ])

        print(st)

    def animate_trajectory(self) -> None:
        x_data = [result.item(0) for result in self.q_solutions]
        y_data = [result.item(1) for result in self.q_solutions]

        fig = plt.figure()
        ax = plt.axes(xlim=(min(x_data),max(x_data)), ylim=(min(y_data), max(y_data)))

        ax.set_xlabel(self.q_list[0])
        ax.set_ylabel(self.q_list[1])
        line, = ax.plot([], [], lw=2)
        # line, = ax.plot([], [], 'o')
        # line, = ax.plot([], [], 'o', markersize=1)

        title = ax.text(0.5, 0.85, "".format(self.t_list[0]), bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                        transform=ax.transAxes, ha="center")
        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            line.set_data(x_data[0:i], y_data[0:i])
            # line.set_data([x_data[i]], [y_data[i]])
            title.set_text("t={0:.2f}".format(self.t_list[i]))
            return title,line,

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(self.q_solutions), interval=1, blit=True)

        # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
        plt.show()

    def plot_results(self, cols=1) -> None:
        """
        Plot results
        """
        fig = plt.figure(1)

        number_of_subplots = len(self.q_list)

        # Compute number of rows required
        number_of_rows = number_of_subplots // cols
        number_of_rows += number_of_subplots % cols

        # Position index
        position = range(1, number_of_subplots + 1)

        for k in range(number_of_subplots):
            ax = fig.add_subplot(number_of_rows, cols, position[k])
            ax.plot(self.t_list, [result.item(k) for result in self.q_solutions], 'o', markersize=1)
            ax.set_ylabel(self.q_list[k])
            if k == 0:
                ax.set_title('Evolution of generalised coordinates as a function of ' + self.t)
            if k == number_of_subplots - 1:
                ax.set_xlabel(self.t)

        # plt.subplot(211)
        # plt.title('Evolution of generalised coordinates as a function of time')
        # # plt.plot(self.t_list, list(map(lambda result: result.item(0), self.q_solutions)))
        # plt.plot(self.t_list, [result.item(0) for result in self.q_solutions])
        # plt.ylabel(self.q_list[0])

        # plt.subplot(212)
        # # plt.plot(self.t_list, list(map(lambda result: result.item(1), self.q_solutions)))
        # plt.plot(self.t_list, [result.item(1) for result in self.q_solutions])
        # plt.xlabel(self.t)
        # plt.ylabel(self.q_list[1])

        if len(self.q_list) == 2:
            plt.figure(2)
            plt.plot([result.item(0) for result in self.q_solutions], [result.item(1) for result in self.q_solutions],
                     'o', markersize=1)
            plt.title('Trajectory')
            plt.xlabel(self.q_list[0])
            plt.ylabel(self.q_list[1])

        plt.show()
