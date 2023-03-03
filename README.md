# Numeric Simulator for differential equation
Consistent interface for integrating linear and nonlinear time-invariant state space models using `scipy.integrate.solve_ivp`.

Time continuous state space equations

for linear models:
$$
\begin{aligned}
\dot{x}(t) & = Ax(t) + Bu(t) \\
y(t) & = Cx(t) + Du(t)
\end{aligned}
$$
for nonlinear models:
$$
\begin{aligned}
\dot{x}(t) & = f(x(t), u(t)) \\
y(t) & = g(x(t), u(t))
\end{aligned}.
$$