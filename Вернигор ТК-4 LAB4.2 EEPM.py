from scipy.integrate import ode
import matplotlib.pyplot as plt

starts = [30, 180]
t1 = 30


def f(t, N):
    return 1.4 * N - 0.007 * N ** 2


def solve(N0, t0=0, t1=1, h=0.05):
    r = ode(f).set_integrator('', method='bdf')
    r.set_initial_value(N0, t0)

    N = [N0]
    t = [t0]

    while r.successful() and r.t < t1:
        t.append(r.t + h)
        N.append(r.integrate(r.t + h))
    return N, t


for i in range(len(starts)):
    plt.subplot(2, 1, i + 1)
    N, t = solve(starts[i], t0=0, t1=t1, h=0.01)
    plt.title("N(0) = " + str(starts[i]))
    plt.plot(t, N)

plt.tight_layout()
plt.show()