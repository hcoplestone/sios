class FirstOrderQuadrature:

    @staticmethod
    def trapezium_rule(x0, x1, time_step):
        return 0.5 * time_step * (x0 + x1)
