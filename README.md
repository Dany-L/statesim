# Numeric Simulator for state space models
Simulating ordinary differential equations that have a state space representation with external input. For integration of continuous systems `scipy.integrate.solve_ivp` is used. Discrete systems are iterated with a fixed step size.

State space description, given an initial condition $x(0)$ and a fixed time horizon $T$, for discrete system with a step size $\eta$

- for linear continuous models ($t = [0, T]$):
$$\begin{pmatrix}\dot{x}(t) \\y(t)\end{pmatrix} = \begin{pmatrix}A & B \\ C & D\end{pmatrix}\begin{pmatrix}x(t) \\ u(t) \end{pmatrix} $$
- for nonlinear continuous models ($t = [0, T]$):
$$\dot{x}(t) = f(x(t), u(t)),\qquad y(t) = g(x(t), u(t))$$
- for linear discrete models ($k = 0, \ldots, (T/\eta)-1$):
$$\begin{pmatrix}x(k+1) \\ y(k) \end{pmatrix} = \begin{pmatrix} A & B \\ C & D\end{pmatrix}\begin{pmatrix} x(k) \\ u(k) \end{pmatrix}$$

## Example
In `scripts` some examples on how to use `statesim` are shown in *jupyter notebooks*