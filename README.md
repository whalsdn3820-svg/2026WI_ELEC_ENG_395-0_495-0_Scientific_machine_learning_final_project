# Data-Driven Equation Discovery of the Duffing Oscillator (SINDy)

This project investigates whether the governing equation of a nonlinear dynamical system can be recovered directly from data using the Sparse Identification of Nonlinear Dynamics (SINDy) framework.

---

## 📌 Overview

We consider the Duffing oscillator, a nonlinear system with cubic stiffness and periodic forcing:

$$
\ddot{x} + \delta \dot{x} + \alpha x + \beta x^3 = \gamma \cos(\omega t)
$$

The main objective is to recover this governing equation from trajectory data using SINDy, both in clean and noisy scenarios.

---

## ⚙️ Methodology

The workflow consists of the following steps:

1. **Data Generation**
   - Numerical simulation using a 4th-order Runge–Kutta (RK4) solver  
   - Gaussian noise added to mimic measurement data  

2. **Derivative Estimation**
   - Savitzky–Golay filtering used for stable numerical differentiation  

3. **SINDy (Sparse Regression)**
   - Candidate function library includes:
   $$
     - Polynomial terms: \( x, v, x^2, x^3, xv, v^2 \)
     - Forcing terms: \( \cos(\omega t), \sin(\omega t) \)
   $$
   - Sequential Thresholded Least Squares (STLSQ) is used  

4. **Neural Denoising**
   - A ResNet model is used to smooth noisy trajectories before applying SINDy  

---

## 📊 Results

### ✔ Clean Data
- SINDy successfully recovers the exact governing equation  

### ✔ Noisy Data
Recovered system:

```
dx/dt = 1.000 v
dv/dt = 1.016 x - 0.201 v - 1.014 x^3 + 0.304 cos(ωt)
```

- Coefficients closely match the true system  
- Nonlinear and forcing terms are correctly identified  

### ✔ With ResNet Denoising
- Trajectories become smoother  
- Equation recovery accuracy is improved  

---

## 🚀 Key Insights

- SINDy can recover nonlinear governing equations directly from data  
- The method is sensitive to noise due to numerical differentiation  
- Neural denoising (ResNet) can further improve robustness  

---

## 🎥 Presentation

You can find the presentation here:

👉 https://youtube.com/YOUR_VIDEO_LINK

---

## 📚 Reference

-Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz. "Discovering governing equations from data by sparse identification of nonlinear dynamical systems." Proceedings of the national academy of sciences 113.15 (2016): 3932-3937.
-Brunton, Steven L., and J. Nathan Kutz. Data-driven science and engineering: Machine learning, dynamical systems, and control. Cambridge University Press, 2022.

---

## 👤 Author

Minwoo Cho  
Theoretical and Applied Mechanics Program  
Northwestern University
