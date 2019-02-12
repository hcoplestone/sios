class FirstOrderQuadrature:

    @staticmethod
    def trapezium_rule(y0, y1, time_step):
        return 0.5 * time_step * (y0 + y1)
