from integrators import GalerkinGaussLobattoIntegrator

def main():
    """
    Run the integrator :)
    :return:
    """
    ggl = GalerkinGaussLobattoIntegrator('x', ['q1', 'q2'], ['v1', 'v2'], True)
    ggl.integrate('x^2')


if __name__ == "__main__":
    main()
