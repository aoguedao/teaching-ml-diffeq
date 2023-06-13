import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

# Right-hand side of ODE
def f(t, y):
    return t / 9 * np.cos(2 * y) + t ** 2

# Time array
t0 = 0
tn = 10
h = 0.5
t_array = np.arange(t0, tn, h)
t_array

# Solve IVP
sol = solve_ivp(f, t_span=[0, 10], y0=[1], t_eval=t_array)

# Plot
y_sol = sol.y.flatten()
plt.plot(t_array, y_sol, linestyle="dashed")
plt.show()