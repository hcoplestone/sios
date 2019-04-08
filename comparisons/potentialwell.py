import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
G = 6.67408e-11
MEarth = 5.972e24
Mmoon = 7.34767309e22

# Orbital Parameters
mu = G*(MEarth + Mmoon)
J = 2.9e34
j = J/Mmoon
perigeeRadius = 3565e5
apogeeRadius = 3829e5

def main():
    rs = np.linspace(0,apogeeRadius/2, 1000)
    Vs = j*j/(2*mu*rs*rs) -1*G*MEarth*Mmoon/rs
    plt.plot(rs, Vs)
    plt.yscale('symlog')
    plt.xscale('symlog')
    plt.show()

if __name__ == '__main__':
    main()