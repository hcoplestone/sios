class Taylor:
    @staticmethod
    def exp(x, n=200):
        """
        The exponential function as a Taylor series about x = 0.
        :param x: Variable to expand in.
        :param n: Number of terms in taylor series.
        :return: f(x)
        """
        r = 0
        for i in range(1, n):
            r += np.power(x, i) / math.factorial(i)
        return r
