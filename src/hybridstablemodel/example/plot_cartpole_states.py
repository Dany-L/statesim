from hybridstablemodel.model.statespace import ContinuousNonlinear
from hybridstablemodel.simulator import Simulator
from hybridstablemodel.system.cartpole import CartPole
from hybridstablemodel.plot.plot_simulation_results import plot_states, plot_outputs, plot_inputs
import matplotlib.pyplot as plt

import numpy as np

if __name__ == "__main__":
    system = CartPole(mu_c=0.0, mu_p=0.1)
    model = ContinuousNonlinear(
        f=system.state_dynamics, g=system.output_function, nx=4, nu=1, ny=1
    )
    T_end = 10.0
    step_size = 0.02
    N = int(T_end/step_size)
    sim = Simulator(T=T_end, step_size=step_size)
    us = [np.array([[u]]) for u in np.random.uniform(-2,2,size=(N,))]
    result, t, _ = sim.simulate(
        model=model,
        initial_state=np.array([[0], [0], [np.pi + 0.1], [0]]),
        input=us,
    )
    plot_states(result=result, t=t)
    plot_inputs(result=result, t=t)
    plot_outputs(result=result, t=t)
    plt.show()
    
