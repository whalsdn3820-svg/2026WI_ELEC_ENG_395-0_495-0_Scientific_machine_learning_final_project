
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ---------------------------------------------------
# 1) Duffing oscillator
# ---------------------------------------------------
def duffing_rhs(t, y, delta, alpha, beta, gamma, omega):

    x, v = y

    dxdt = v
    dvdt = -delta*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t)

    return np.array([dxdt, dvdt])


# ---------------------------------------------------
# 2) RK4 solver
# ---------------------------------------------------
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


# ---------------------------------------------------
# 3) Generate Duffing data
# ---------------------------------------------------

delta = 0.2
alpha = -1.0
beta = 1.0
gamma = 0.3
omega = 1.2

t0 = 0
tf = 80
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

x = y[:,0]
v = y[:,1]


# ---------------------------------------------------
# 4) Savitzky–Golay derivative
# ---------------------------------------------------

dxdt = savgol_filter(x, 51, 3, deriv=1, delta=dt)
dvdt = savgol_filter(v, 51, 3, deriv=1, delta=dt)

Xdot = np.vstack([dxdt, dvdt]).T


# ---------------------------------------------------
# 5) Build library
# ---------------------------------------------------

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


Theta, names = build_library(x, v, t)


# ---------------------------------------------------
# 6) STLSQ SINDy
# ---------------------------------------------------

def STLSQ(Theta, dXdt, threshold, max_iter=10):

    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]

    for _ in range(max_iter):

        small = np.abs(Xi) < threshold
        Xi[small] = 0

        for i in range(dXdt.shape[1]):

            big_idx = Xi[:,i] != 0

            if np.sum(big_idx) == 0:
                continue

            Xi[big_idx,i] = np.linalg.lstsq(
                Theta[:,big_idx],
                dXdt[:,i],
                rcond=None
            )[0]

    return Xi


Xi = STLSQ(Theta, Xdot, threshold=0.05)


# ---------------------------------------------------
# 7) Print recovered equations
# ---------------------------------------------------

state_names = ["dx/dt", "dv/dt"]

for i in range(2):

    print("\nRecovered equation for", state_names[i])

    terms = []

    for coef, name in zip(Xi[:,i], names):

        if abs(coef) > 1e-5:

            terms.append(f"{coef:.3f}*{name}")

    print(state_names[i], "=", " + ".join(terms))


# ---------------------------------------------------
# 8) SINDy RHS
# ---------------------------------------------------

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


# ---------------------------------------------------
# 9) Simulate SINDy dynamics
# ---------------------------------------------------

t_s, y_s = solve_rk4(sindy_rhs, t0, tf, dt, y0, Xi)

x_s = y_s[:,0]
v_s = y_s[:,1]


# ---------------------------------------------------
# 10) Visualization
# ---------------------------------------------------

fig, axes = plt.subplots(1,3, figsize=(16,4))


# Time series
axes[0].plot(t, x, label="True trajectory", linewidth=2)
axes[0].plot(t_s, x_s, "--", label="SINDy reconstruction", linewidth=2)

axes[0].set_xlabel("t")
axes[0].set_ylabel("x(t)")
axes[0].set_title("Time series")
axes[0].legend()
axes[0].grid(True)


# Phase portrait
axes[1].plot(x, v, label="True trajectory", linewidth=2)
axes[1].plot(x_s, v_s, "--", label="SINDy reconstruction", linewidth=2)

axes[1].set_xlabel("x")
axes[1].set_ylabel("v")
axes[1].set_title("Phase portrait")
axes[1].legend()
axes[1].grid(True)


# Error
error = np.sqrt((x - x_s)**2 + (v - v_s)**2)

axes[2].plot(t, error)

axes[2].set_xlabel("t")
axes[2].set_ylabel("trajectory error")
axes[2].set_title("Reconstruction error")
axes[2].grid(True)


plt.tight_layout()
plt.show()

print(np.average(error))

# %%
