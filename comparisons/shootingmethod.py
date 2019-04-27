import numpy as np
import matplotlib.pyplot as plt

q0 = 1
q1 = 2

guess_not_shooting = 2
guess_shooting = 3

t = [1, 2]
tguess = [3]

def main():
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.grid()
    ax1.plot(t, [q0, q1], 'x--k', zorder=3)
    ax1.plot([2] + tguess, [2, guess_not_shooting], '.--b', zorder=2)
    ax1.set_xticks([1,2,3])
    ax1.set_xticklabels(['$t_1$', '$t_2$', '$t_3$'])
    ax1.set_yticks([1, 2, 3])
    ax1.set_ylabel('$q_i$')
    ax1.set_xlabel('$t$')
    ax1.set_ylim(0.9, 3.1)
    ax1.set_xlim(0.9, 3.1)
    ax1.set_title('Without shooting')

    ax2.grid()
    ax2.plot(t, [q0, q1], 'x--k', zorder=3)
    ax2.plot([2] + tguess, [2, guess_shooting], '.--b', zorder=2)
    ax2.set_xticks([1,2,3])
    ax2.set_xticklabels(['$t_1$', '$t_2$', '$t_3$'])
    ax2.set_yticks([1, 2, 3])
    ax2.set_ylabel('$q_i$')
    ax2.set_xlabel('$t$')
    ax2.set_ylim(0.9, 3.1)
    ax2.set_xlim(0.9, 3.1)
    ax2.set_title('With shooting')

    plt.show()

if __name__ == '__main__':
    main()