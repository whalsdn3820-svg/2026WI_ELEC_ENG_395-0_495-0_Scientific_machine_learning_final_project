# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import savgol_filter


# ======================================
# Duffing oscillator
# ======================================

def duffing_rhs(t, y, delta, alpha, beta, gamma, omega):

    x, v = y

    dxdt = v
    dvdt = -delta*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t)

    return np.array([dxdt, dvdt])


# ======================================
# RK4 solver
# ======================================

def rk4_step(f, t, y, h, *args):

    k1 = f(t, y, *args)
    k2 = f(t + 0.5*h, y + 0.5*h*k1, *args)
    k3 = f(t + 0.5*h, y + 0.5*h*k2, *args)
    k4 = f(t + h, y + h*k3, *args)

    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


def solve_rk4(f, t0, tf, h, y0, *args):

    n_steps = int((tf - t0)/h)

    t = np.zeros(n_steps+1)
    y = np.zeros((n_steps+1, len(y0)))

    t[0] = t0
    y[0] = y0

    ti = t0
    yi = np.array(y0)

    for i in range(n_steps):

        yi = rk4_step(f, ti, yi, h, *args)
        ti += h

        t[i+1] = ti
        y[i+1] = yi

    return t, y


# ======================================
# Generate Duffing data
# ======================================

delta = 0.2
alpha = -1
beta = 1
gamma = 0.3
omega = 1.2

t0 = 0
tf = 160
dt = 0.01

y0 = [1.0, 0.0]

t, y = solve_rk4(
    duffing_rhs,
    t0,
    tf,
    dt,
    y0,
    delta,
    alpha,
    beta,
    gamma,
    omega
)

x_true = y[:,0]
v_true = y[:,1]


# ======================================
# Add noise
# ======================================

np.random.seed(0)

noise_level = 0.05
x_noisy = x_true + noise_level*np.random.randn(len(x_true))


# ======================================
# Use only first 80 seconds
# ======================================

mask = t <= 80

t = t[mask]
x_true = x_true[mask]
v_true = v_true[mask]
x_noisy = x_noisy[mask]


# ======================================
# Fourier feature input
# ======================================

t_norm = 2*(t - t.min())/(t.max()-t.min()) - 1

features = np.stack([
    t_norm,
    np.sin(omega*t),
    np.cos(omega*t)
], axis=1)

X = torch.tensor(features, dtype=torch.float32)
Y = torch.tensor(x_noisy, dtype=torch.float32).view(-1,1)

device = "cuda" if torch.cuda.is_available() else "cpu"

X = X.to(device)
Y = Y.to(device)


# ======================================
# Deep ResNet model
# ======================================

class ResBlock(nn.Module):

    def __init__(self, width):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU()
        )

    def forward(self, x):

        return x + self.net(x)


class DeepResNet(nn.Module):

    def __init__(self):

        super().__init__()

        width = 256

        self.input = nn.Linear(3, width)

        self.blocks = nn.Sequential(

            ResBlock(width),
            ResBlock(width),
            ResBlock(width),
            ResBlock(width),

            ResBlock(width),
            ResBlock(width),
            ResBlock(width),
            ResBlock(width)

        )

        self.output = nn.Linear(width,1)

    def forward(self,x):

        x = torch.tanh(self.input(x))
        x = self.blocks(x)
        return self.output(x)


model = DeepResNet().to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=1500,
    gamma=0.5
)

loss_fn = nn.MSELoss()


# ======================================
# Training
# ======================================

for epoch in range(6000):

    pred = model(X)

    loss = loss_fn(pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    scheduler.step()

    if epoch % 500 == 0:

        print(epoch, loss.item())


# ======================================
# Smooth trajectory from ResNet
# ======================================

model.eval()

with torch.no_grad():

    x_smooth = model(X).cpu().numpy().flatten()


# ======================================
# Derivative estimation
# ======================================

v_smooth = savgol_filter(x_smooth, 51, 3, deriv=1, delta=dt)

dxdt = savgol_filter(x_smooth, 51, 3, deriv=1, delta=dt)
dvdt = savgol_filter(v_smooth, 51, 3, deriv=1, delta=dt)

Xdot = np.vstack([dxdt, dvdt]).T


# ======================================
# SINDy library
# ======================================

def build_library(x, v, t):

    Theta = np.column_stack([
        np.ones_like(x),
        x,
        v,
        x**2,
        x*v,
        v**2,
        x**3,
        np.cos(omega*t)
    ])

    names = [
        "1",
        "x",
        "v",
        "x^2",
        "xv",
        "v^2",
        "x^3",
        "cos(ωt)"
    ]

    return Theta, names


Theta, names = build_library(x_smooth, v_smooth, t)


# ======================================
# STLSQ SINDy
# ======================================

def STLSQ(Theta, dXdt, threshold):

    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]

    for _ in range(10):

        small = np.abs(Xi) < threshold
        Xi[small] = 0

        for i in range(dXdt.shape[1]):

            big = Xi[:,i] != 0

            Xi[big,i] = np.linalg.lstsq(
                Theta[:,big],
                dXdt[:,i],
                rcond=None
            )[0]

    return Xi


Xi = STLSQ(Theta, Xdot, 0.02)


print("\nRecovered equation")

for i,name in enumerate(["dx/dt","dv/dt"]):

    terms = []

    for coef, term in zip(Xi[:,i], names):

        if abs(coef) > 1e-4:

            terms.append(f"{coef:.3f}*{term}")

    print(name,"=", " + ".join(terms))


# ======================================
# SINDy trajectory reconstruction
# ======================================

def sindy_rhs(t, y, Xi):

    x, v = y

    library = np.array([
        1,
        x,
        v,
        x**2,
        x*v,
        v**2,
        x**3,
        np.cos(omega*t)
    ])

    dxdt = np.dot(library, Xi[:,0])
    dvdt = np.dot(library, Xi[:,1])

    return np.array([dxdt, dvdt])


t_s, y_s = solve_rk4(
    sindy_rhs,
    t[0],
    t[-1],
    dt,
    y0,
    Xi
)

x_s = y_s[:,0]
v_s = y_s[:,1]


# ======================================
# Visualization
# ======================================

plt.figure(figsize=(15,4))


# -------------------------
# trajectory
# -------------------------
plt.subplot(1,3,1)

plt.plot(t, x_true, label="True")
plt.scatter(t, x_noisy, s=3, alpha=0.3, color="red", label="Noisy")
plt.plot(t, x_smooth, label="ResNet")
plt.plot(t_s, x_s, "--", label="SINDy")

plt.legend()
plt.title("Trajectory comparison")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid()


# -------------------------
# phase portrait
# -------------------------
plt.subplot(1,3,2)

# noisy velocity approximation
v_noisy = np.interp(t, t, v_true)

plt.plot(x_true, v_true, label="True")
plt.scatter(x_noisy, v_noisy, s=3, alpha=0.3, color="red", label="Noisy")
plt.plot(x_smooth, v_smooth, label="ResNet")
plt.plot(x_s, v_s, "--", label="SINDy")

plt.legend()
plt.title("Phase portrait")
plt.xlabel("x")
plt.ylabel("v")
plt.grid()


# -------------------------
# error
# -------------------------
plt.subplot(1,3,3)

err = np.sqrt((x_true - x_s)**2 + (v_true - v_s)**2)

plt.plot(t, err)

plt.title("SINDy reconstruction error")
plt.xlabel("t")
plt.ylabel("error")
plt.grid()


plt.tight_layout()
plt.show()
print(np.average(err))



# %%
