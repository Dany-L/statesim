# Numeric Simulator for state space models
Simulating ordinary differential equations that have a state space representation with external input. For integration of continuous systems `scipy.integrate.solve_ivp` is used. Discrete systems are iterated with a fixed step size.

State space description, given an initial condition $x(0)$ and a fixed time horizon $T$, for discrete system with a step size $\eta$

- for linear continuous models ($t = [0, T]$):
$$
\dot{x}(t) = Ax(t) + Bu(t), ~ y(t) = Cx(t) + Du(t)
$$
- for nonlinear continuous models ($t = [0, T]$):
$$
\dot{x}(t) = f(x(t), u(t)),~ y(t) = g(x(t), u(t))
$$
- for linear discrete models ($k = 0, \ldots, (T/\eta)-1$):
$$
x(k+1) = Ax(k) + Bu(k), ~ y(k) = Cx(k) + Du(k)
$$

## Example
In `scripts` some examples on how to use `statesim` are shown in *jupyter notebooks*