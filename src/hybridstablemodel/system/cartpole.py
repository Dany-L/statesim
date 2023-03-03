import numpy as np
import abc
from numpy.typing import NDArray
from sympy import symbols, diff, sin, cos, sign, init_printing

init_printing(use_unicode=True)

class DynamicSystem(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def state_dynamics(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64]
    )-> NDArray[np.float64]:
        pass

    @abc.abstractmethod
    def output_function(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        pass

class CartPole(DynamicSystem):
    def __init__(
        self,
        g: float = 9.81,
        m_c: float = 1.0,
        m_p: float = 0.1,
        length: float = 0.5,
        mu_c: float = 0.0,
        mu_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.g = g
        self.m_c = m_c
        self.m_p = m_p
        self.total_mass = (self.m_p + self.m_c)
        self.length = length # actually half the pole's length
        self.mu_c = mu_c # coefficient of friction of cart on track
        self.mu_p = mu_p # coefficient of friction of pole on cart

    def state_dynamics(self, x, u):
        _, x2, x3, x4 = x
        x_dot = np.zeros_like(x)
        x_dot[3] = (self.g * np.sin(x3) + np.cos(x3) * ((- u - self.m_p * self.length * x4**2 * np.sin(x3) + self.mu_c * np.sign(x2)) / (self.m_c + self.m_p)) - (self.mu_p * x4) / (self.m_p * self.length)) / (self.length * (4 / 3 - (self.m_p * np.cos(x3)**2) / (self.m_c + self.m_p)))
        x_dot[0] = x2
        x_dot[1] = (u + self.m_p * self.length * x4**2 * np.sin(x3) - self.m_p * self.length * x_dot[3] * np.cos(x3) - self.mu_c * np.sign(x2)) / (self.m_c + self.m_p)
        x_dot[2] = x4
        return x_dot
    
    def output_function(self, x, u):
        return np.array([[1, 0, 0, 0]]) @ x


    def symb_lin(self, linearization_point):
        # symbolic linearization evaluated at linearization point
        x1, x2, x3, x4, u = symbols('x1 x2 x3 x4 u')

        eval_dict = {x1: linearization_point[0], x2: linearization_point[1], x3:linearization_point[2], x4:linearization_point[3], u:0}
        
        x4_dot = (self.g * sin(x3) + cos(x3) * ((- u - self.m_p * self.l * x4**2 * sin(x3) + self.mu_c * sign(x2)) / (self.m_c + self.m_p)) - (self.mu_p * x4) / (self.m_p * self.l)) / (self.l * (4 / 3 - (self.m_p * cos(x3)**2) / (self.m_c + self.m_p)))
        x1_dot = x2
        x2_dot = (u + self.m_p * self.l * x4**2 * sin(x3) - self.m_p * self.l * x4_dot * cos(x3) - self.mu_c * sign(x2)) / (self.m_c + self.m_p)
        x3_dot = x4

        A = np.array([
            [diff(x1_dot, x1).evalf(subs=eval_dict), diff(x1_dot, x2).evalf(subs=eval_dict), diff(x1_dot, x3).evalf(subs=eval_dict), diff(x1_dot, x4).evalf(subs=eval_dict)],
            [diff(x2_dot, x1).evalf(subs=eval_dict), diff(x2_dot, x2).evalf(subs=eval_dict), diff(x2_dot, x3).evalf(subs=eval_dict), diff(x2_dot, x4).evalf(subs=eval_dict)],
            [diff(x3_dot, x1).evalf(subs=eval_dict), diff(x3_dot, x2).evalf(subs=eval_dict), diff(x3_dot, x3).evalf(subs=eval_dict), diff(x3_dot, x4).evalf(subs=eval_dict)],
            [diff(x4_dot, x1).evalf(subs=eval_dict), diff(x4_dot, x2).evalf(subs=eval_dict), diff(x4_dot, x3).evalf(subs=eval_dict), diff(x4_dot, x4).evalf(subs=eval_dict)],
        ])

        B = np.array([
            [diff(x1_dot, u).evalf(subs=eval_dict)],
            [diff(x2_dot, u).evalf(subs=eval_dict)],
            [diff(x3_dot, u).evalf(subs=eval_dict)],
            [diff(x4_dot, u).evalf(subs=eval_dict)],
        ])

        return A.astype(np.float32),B.astype(np.float32)
    