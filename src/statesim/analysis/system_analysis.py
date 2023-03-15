import matplotlib.pyplot as plt
from typing import Tuple, Union
from numpy.typing import NDArray
import numpy as np


class SystemAnalysisContinuous:
    def __init__(
        self,
        system: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        self.A = system[0]
        self.B = system[1]
        self.C = system[2]
        self.D = system[3]
        self.nx = self.A.shape[0]
        self.tf = (
            lambda s: self.C
            @ np.linalg.inv((s * np.eye(self.nx)) - self.A)
            @ self.B
            + self.D
        )
        self.real_eig = np.real(np.linalg.eig(self.A)[0])

    def plot_magnitude(
        self, w: NDArray[np.float64] = np.linspace(0.001, 10, 1000)
    ) -> None:
        gain = [np.linalg.norm(self.tf(complex(0, w_i)), ord=2) for w_i in w]
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.semilogx(w, 20 * np.log10(np.abs(gain)))
        ax.grid()
        ax.set_xlabel(r'$iw$')
        ax.set_ylabel(r'$20 \log_{10}(\|G(iw)\|_2)$')
        ax.set_title('Magnitude of $G$')

    def get_peak_gain(
        self, w: NDArray[np.float64] = np.linspace(0.001, 10, 1000)
    ) -> np.float64:
        return np.max(
            [np.linalg.norm(self.tf(complex(0, w_i)), ord=2) for w_i in w]
        )

    def get_h_inf_norm(self) -> Union[np.float64, float]:
        return self.get_peak_gain() if self.is_stable() else np.inf

    def get_real_eigenvalues(self) -> NDArray[np.float64]:
        return np.array(self.real_eig)

    def is_stable(self) -> bool:
        return all(self.real_eig < 0)

    def analysis(self) -> None:
        print(f'Is system stable?: {self.is_stable()}')
        print(f'H inf norm: {self.get_h_inf_norm()}')
        print(f'Peak gain: {self.get_peak_gain()}')
        print(f'Eigenvalues: {self.get_real_eigenvalues()}')
