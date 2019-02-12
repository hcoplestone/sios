import numpy as np

class GaussLobattoQuadrature:

    @staticmethod
    def trapezium_rule(y0, y1, time_step):
        return 0.5 * time_step * np.add(y0, y1)
