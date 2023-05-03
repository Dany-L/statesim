# Numeric Simulator for state space models
Simulating ordinary differential equations that have a state space representation with external input. For the integration of continuous systems `scipy.integrate.solve_ivp` is used. Discrete systems are iterated with a fixed step size.

State space description, given an initial condition $x(0)$ and a fixed time horizon $T$, for discrete system with a step size $\eta$

- for linear continuous models ($t = [0, T]$):
```math
\begin{pmatrix}
    \dot{x}(t) \\
    y(t)
\end{pmatrix} = 
\begin{pmatrix}
    A & B \\
    C & D
\end{pmatrix}
\begin{pmatrix}
    x(t) \\
    u(t)
\end{pmatrix}
```
- for nonlinear continuous models ($t = [0, T]$):
$$\dot{x}(t) = f(x(t), u(t)),\qquad y(t) = g(x(t), u(t))$$
- for linear discrete models ($k = 0, \ldots, (T/\eta)-1$):
```math
\begin{pmatrix}
    x(k+1) \\
    y(k)
\end{pmatrix} = 
\begin{pmatrix}
    A & B \\
    C & D
\end{pmatrix}
\begin{pmatrix}
    x(k) \\
    u(k)
\end{pmatrix}
```

## Simulation
The continuous simulator will evaluate the function depending on the method chosen, the input sequence is always discrete with a constant distance of `step_size`. The output sequence has the same step size as the input sequence, therefore the result of the continuous integrator is resampled (or evaluated) to the `step_size`.

## Systems
The systems that can be simulated with `statesim` are described by nonlinear differential equations. Each system has a nonlinear symbolic expression that can be used for simulation and is considered to be the ground truth data. From the symbolic nonlinear expressions, linearizations can be derived and evaluated at an equilibrium point with `SymPy`.

Currently, the following systems are implemented
- **CartPole**: Zero input, the initial state is around the upright position of the pole (Barto AG, Sutton RS, Anderson CW. Neuronlike adaptive elements that can solve difficult learning control problems. IEEE transactions on systems, man, and cybernetics. 1983 Sep(5):834-46.)
![Cartpole](/img/state_cartpole.png)

- **Coupled mass spring damper system**: states of 4 coupled masses
![4 link coupled msd](/img/sate_msd.png)

- **Inverted pendulum with torque input**: Zero input, the initial state is around the upright position of the pole
![Pendulum](/img/state_pend.png)


## Input sequences
Random input sequences can be generated to excite the system.

Currently, the following types of input generation are implemented:
- **Random Static**: Static inputs that jump to another static value after a random time from a given interval.
![input sequence](/img/input.png)
## Example
In `scripts/notebooks` some examples of how to use `statesim` are shown in *jupyter notebooks*. To generate `.csv` files from the simulations the script `scripts/run_cartpole_datagen.py` shows this for the cartpole example. The configuration can also be external as a `.json` file with the fields:
```json
    {
        "result_directory": "str, Directory where the .csv files will be stored, must exist",
        "base_name": "str, Folder name within the result directory, will be created",
        "seed": "int, a random seed that will be set to numpy",
        "K": "int, Number of samples",
        "T": "float, Simulation end time start is always 0, e.g. [0, T]",
        "step_size": "float, step between two measurements",
        "input": "InputConfig, configuration for generating random input sequences",
        "system": "SystemConfig, specific configuration for the system to be simulated",
        "simulator": "SimulatorConfig, configuration for the simulator, requires an initial state"
    }
```