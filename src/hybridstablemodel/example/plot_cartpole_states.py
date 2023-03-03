from hybridstablemodel.model.statespace import ContinuousNonlinear
from hybridstablemodel.simulator import Simulator
from hybridstablemodel.system.cartpole import CartPole
from hybridstablemodel.plot.plot_simulation_results import plot_states


import numpy as np

if __name__ == "__main__":
    system = CartPole()
    model = ContinuousNonlinear(
        f=system.state_dynamics, g=system.output_function, nx=4, nu=1, ny=1
    )
    T_end = 10.0
    step_size = 1.0
    N = int(T_end)
    sim = Simulator(T=T_end, step_size=step_size)
    u = [np.array([[u]]) for u in np.random.rand(N)]
    result, t, _ = sim.simulate(
        model=model,
        initial_state=np.array([[0], [0], [np.pi + 0.1], [0]]),
        input=u,
    )
    plot_states(result=result, t=t)
