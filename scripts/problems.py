import numpy as np

from interval import LP, LPV
from utils import intervals_product, p, n


class Integrator(LP):
    def __init__(self, x0=[5.5]):
        A = [[-1]]
        B = [[1]]
        dAs = [[[0]], [[0.5]]]
        x_i = [[5], [6]]
        super().__init__(x0, A, dAs, B, d_i=[[-0.2], [0.2]], x_i=x_i)


class DoubleIntegrator(LP):
    def __init__(self, x0=[0, 0], center=[4, 0]):
        self.params = np.array([[-14-1.5, -14+1.5], [-10-0.05, -10+0.05]])
        A_theta = lambda params: np.array([[0, 1], [params[0], params[1]]])
        A0, dAs = LPV.polytope(A_theta, self.params)
        super().__init__(x0, A0, dAs, center=center)


class Example(LP):
    def __init__(self, x0=[2, 1]):
        A = [[-1, 1], [0.1, -1]]
        B = [[-2], [1]]
        dAs = [0*np.array(A)]
        x_i = [[-1, -2], [2, 1]]
        d_i = [[-1], [1]]
        d = lambda t: [1*np.sin(7*t)]
        super().__init__(x0, A, dAs, B=B, d=d, d_i=d_i, x_i=x_i)


class UniVehicle(LP):
    def __init__(self, x0=[0, 5, 3, 1]):
        self.params = np.array([[5, 5.1], [3, 3.1]])
        v0 = 2
        d0 = 5
        T = 2
        Kx = 6
        center = [-d0-v0*T, 0, v0, v0]
        A_theta = lambda params: np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-Kx, Kx, -params[0] - params[1] - Kx * T, params[1]],
            [0, 0, 0, -params[0]]
        ])
        B = [v0, v0, 0, 0]
        A0, dAs = LPV.polytope(A_theta, self.params)
        super().__init__(x0, A0, dAs, B, center)


class LoneVehicle(LPV):
    def __init__(self, x0=[1, 0, 0.9], center=[0, 0, 1]):
        self.x0 = x0
        self.gains_i = np.array([[-6, -5],  # -K_y K_psi
                                 [-9, -8],  # -K_psi
                                 [-1.01, -0.99]])  # -K_v
        x_i = np.array([x0, x0])
        params = self.params_from_state_interval(x_i, self.gains_i)
        A0, dAs = LPV.polytope(self.A_theta, params)
        super().__init__(x0, A0, dAs, center)

    def params_from_state(self, x, gains):
        v = x[2]
        return [v, gains[0]/v, gains[1]]

    def params_from_state_interval(self, x_i, gains_i):
        v_i = x_i[:, 2]
        inv_v_i = np.flip(1./v_i, axis=0)
        params = np.array([
            v_i,
            intervals_product(gains_i[0], inv_v_i),
            gains_i[1]
        ])
        return params

    def A_theta(self, params):
        return np.array([
            [0, params[0], 0],
            [params[1], params[2], 0],
            [0, 0, -1]
        ])
