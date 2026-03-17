# %%
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Duffing oscillator
# -----------------------------
def duffing_rhs(t, y, delta, alpha, beta, gamma, omega):
    x, v = y
    dxdt = v
    dvdt = -delta * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
    return np.array([dxdt, dvdt], dtype=float)


# -----------------------------
# RK4 solver
# -----------------------------
def rk4_step(f, t, y, h, *args):
    k1 = f(t, y, *args)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1, *args)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2, *args)
    k4 = f(t + h, y + h * k3, *args)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_rk4(f, t0, tf, h, y0, *args):
    n_steps = int(np.floor((tf - t0) / h))
    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))

    t[0] = t0
    y[0] = y0

    ti = t0
    yi = np.array(y0, dtype=float)

    for i in range(n_steps):
        yi = rk4_step(f, ti, yi, h, *args)
        ti += h
        t[i + 1] = ti
        y[i + 1] = yi

    return t, y


# -----------------------------
# Parameters
# -----------------------------
delta = 0.2
alpha = -1.0
beta = 1.0
gamma = 0.3
omega = 1.2

t0 = 0.0
tf = 80.0
dt = 0.01
y0 = [1.0, 0.0]

# -----------------------------
# Solve
# -----------------------------
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

x = y[:, 0]
v = y[:, 1]

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Time trajectory
axes[0].plot(t, x, linewidth=2)
axes[0].set_xlabel("Time $t$")
axes[0].set_ylabel("Displacement $x(t)$")
axes[0].set_title("Duffing oscillator trajectory")
axes[0].grid(True)

# Phase portrait
axes[1].plot(x, v, linewidth=2)
axes[1].set_xlabel("Displacement $x$")
axes[1].set_ylabel("Velocity $v$")
axes[1].set_title("Phase portrait")
axes[1].grid(True)

plt.tight_layout()
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Duffing oscillator
# -----------------------------
def duffing_rhs(t, y, delta, alpha, beta, gamma, omega):
    x, v = y
    dxdt = v
    dvdt = -delta * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
    return np.array([dxdt, dvdt])


# -----------------------------
# RK4 solver
# -----------------------------
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


# -----------------------------
# Parameters
# -----------------------------
delta = 0.2
alpha = -1.0
beta = 1.0
gamma = 0.3
omega = 1.2

t0 = 0
tf = 80
dt = 0.01
y0 = [1.0, 0.0]

# -----------------------------
# Solve clean data
# -----------------------------
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

x_clean = y[:,0]
v_clean = y[:,1]

# -----------------------------
# Add noise
# -----------------------------
np.random.seed(0)
noise_level = 0.05

x_noisy = x_clean + noise_level * np.random.randn(len(x_clean))
v_noisy = v_clean + noise_level * np.random.randn(len(v_clean))

# -----------------------------
# (Optional) remove transient for nicer plot
# -----------------------------
mask = t > 20

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12,5))

# Clean phase portrait
axes[0].plot(x_clean[mask], v_clean[mask], linewidth=2)
axes[0].set_title("Clean data (phase portrait)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("v")
axes[0].grid(True)

# Noisy phase portrait
axes[1].scatter(
    x_noisy[mask],
    v_noisy[mask],
    s=3,
    alpha=0.4,
    color="red"
)
axes[1].set_title("Noisy data (phase portrait)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("v")
axes[1].grid(True)

plt.tight_layout()
plt.show()
# %%
