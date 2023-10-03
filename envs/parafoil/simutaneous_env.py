from typing import Tuple
import numpy as np
from simple_env import Parafoil


class RPM_config:
    def __init__(
            self,
            ns: int,
            nu: int,
            t0: float,
            tf: float,
            x0: np.ndarray,
            method: str) -> None:
        self.ns = ns
        self.nu = nu
        self.t0 = t0
        self.tf = tf
        self.x0 = x0
        self.method = method

    def __time_transforme(self):
        """
        Transform time from [t0, tf] to [-1, 1]
        """
        self.t = (2 * self.t - (self.tf + self.t0)) / (self.tf - self.t0)

    def __get_nodes_and_weights(self) -> Tuple[np.ndarray, np.ndarray]:
            """
            Get nodes and weights for RPM
            """
            # use Legendre-Gauss-Radau nodes and weights
            nodes, weights = np.polynomial.legendre.leggauss(self.ns)




class RPM_Parafoil(Parafoil):
    def __init__(
            self, 
            start_point: Tuple = (-600,-600,900), 
            umax: float = 0.14, 
            gamma0: float = 0.75 * np.pi, 
            vs: float = 15, 
            vz: float = -4.6,
            dt: float = 0.1, 
            nstack: int = 1) -> None:
        super().__init__(start_point, umax, gamma0, vs, vz, dt, nstack)



if __name__ == "__main__":
    env = RPM_Parafoil()

    import matplotlib.pyplot as plt 

    N = 4
    N1 = N + 1 
    x = -np.cos(2*np.pi*np.arange(N1)/(2*N+1))
    P = np.zeros((N1, N1+1))

    xold = 2 

    free = np.arange(1, N1)

    while max(abs(x-xold)) > 1e-15:
        xold = x 
        P[0, :] = (-1)**(np.arange(N1+1))
        P[free, 0] = 1
        P[free, 1] = x[free]
        for k in range(1, N1):
            P[free, k+1] = ((2*k-1)*x[free]*P[free, k] - (k-1)*P[free, k-1])/k
        x[free] = xold[free] - ((1 - xold[free]) / N1) * (P[free, N1-1] + P[free, N1]) / (P[free, N1-1] - P[free, N1])

    print(x)
    plt.scatter(x, 0*x, c='r')
    plt.grid()
    plt.xlim([-1.1, 1.1])
    plt.show() 