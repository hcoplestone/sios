from integrators import GalerkinGaussLobattoIntegrator


class Sios:
    def doit(self):
        ggl = GalerkinGaussLobattoIntegrator('x', ['q1', 'q2'], ['v1', 'v2'], True)
        ggl.discretise('x^2', 4, 1.0, 2.0)

if __name__ == "__main__":
   sios = Sios()
   sios.doit()
