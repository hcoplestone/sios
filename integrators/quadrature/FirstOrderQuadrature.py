class FirstOrderQuadrature:

    @staticmethod
    def trapezium_rule_nd(q_n_array, q_n_plus_1_array, time_step):
        return 0.5 * time_step * (q_n_array + q_n_plus_1_array)
