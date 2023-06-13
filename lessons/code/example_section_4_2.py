import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
from deepxde.backend import tf

# Parameters and initial conditions
alpha = 2 / 3
beta = 4 / 3
gamma = 1
delta = 1

t_initial = 0
t_final = 10
x0 = 1.2
y0 = 0.8
t_array = np.linspace(t_initial, t_final, 100)

# ODE residuals loss
def ode(t, Y):
    x = Y[:, 0:1]
    y = Y[:, 1:2]

    dx_dt = dde.grad.jacobian(Y, t, i=0)
    dy_dt = dde.grad.jacobian(Y, t, i=1)
    
    return [
        dx_dt - alpha * x + beta * x * y,
        dy_dt + gamma * y  - delta * x * y
    ]


# Geometry and initical condition loss
geom = dde.geometry.TimeDomain(t_initial, t_final)

def boundary(_, on_initial):
    return on_initial

ic_x = dde.icbc.IC(geom, lambda x: x0, boundary, component=0)
ic_y = dde.icbc.IC(geom, lambda x: y0, boundary, component=1)

# Data, neural network and model
data = dde.data.PDE(
    geom,
    ode,
    [ic_x, ic_y],
    num_domain=512,
    num_boundary=2
)

neurons = 64
layers = 6
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN([1] + [neurons] * layers + [2], activation, initializer)
model = dde.Model(data, net)

# Train
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=50000, display_every=10000)

# Plot loss history
dde.utils.external.plot_loss_history(losshistory)

# Prediction
pinn_pred = model.predict(t_array.reshape(-1, 1))
x_pinn = pinn_pred[:, 0:1]
y_pinn = pinn_pred[:, 1:2]

# Plot PINNs solution
plt.plot(t_array, x_pinn, color="green", label="x_pinn")
plt.plot(t_array, y_pinn, color="blue", label="y_pinn")
plt.legend()
plt.show()