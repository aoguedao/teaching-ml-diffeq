import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Parameters and initial conditions
alpha = 2 / 3
beta = 4 / 3
gamma = 1
delta = 1

t_initial = 0
t_final = 10
x0 = 1.2
y0 = 0.8


# Runge-Kutta solver
def runge_kutta(
    t,
    x0,
    y0,
    alpha,
    beta,
    gamma,
    delta
):

    def func(t, Y):
        x, y = Y
        dx_dt = alpha * x - beta * x * y
        dy_dt = - gamma * y  + delta * x * y
        return dx_dt, dy_dt

    Y0 = [x0, y0]
    t_span = (t[0], t[-1])
    sol = solve_ivp(func, t_span, Y0, t_eval=t)
    x_true, y_true = sol.y
    return x_true, y_true

t_array = np.linspace(t_initial, t_final, 100)
x_rungekutta, y_rungekutta = runge_kutta(t_array, x0, y0, alpha, beta, gamma, delta)

# Plot solution
plt.plot(t_array, x_rungekutta, color="green", label="x_runge_kutta")
plt.plot(t_array, y_rungekutta, color="blue", label="y_runge_kutta")
plt.legend()
plt.show()
