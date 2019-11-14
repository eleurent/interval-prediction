import itertools
import numpy as np

from utils import p, n, intervals_product, mesh_box, is_metzler

dt = 0.01
time = np.arange(0, 6, dt)


class LP(object):
    def __init__(self, x0, A0, dAs, B=None, d_i=None, center=None, x_i=None, d=None):
        self.x0 = np.array(x0, dtype=float)
        self.A0 = np.array(A0, dtype=float)
        self.dAs = [np.array(dAi) for dAi in dAs]
        self.x_i = np.array(x_i) if x_i is not None else None
        self.d_i = np.array(d_i) if d_i is not None else np.zeros((2, *self.x0.shape))
        self.B = np.array(B) if B is not None else np.zeros((*self.x0.shape, 1))
        self.d = d
        self.center = np.array(center) if center is not None else np.zeros(self.x0.shape)
        self.coordinates = None

    def update_coordinates_frame(self, A0):
        self.coordinates = None
        # Rotation
        if not is_metzler(A0):
            v, P = np.linalg.eig(A0)
            if np.isreal(v).all():
                self.coordinates = (P, np.linalg.inv(P))
            else:
                print("Non Metzler with complex eigenvalues: ", v)

    def change_coordinates(self, value, matrix=False, back=False, interval=False, offset=True):
        if self.coordinates is None:
            return value
        P, P_inv = self.coordinates
        if interval:
            for t in range(value.shape[0]):
                value[t, :, :] = intervals_product([self.coordinates[0], self.coordinates[0]],
                                                   value[t, :, :, np.newaxis]).squeeze() + offset * np.array([self.center, self.center])
            return value
        elif matrix:  # Matrix
            if back:
                return P @ value @ P_inv
            else:
                return P_inv @ value @ P
        elif isinstance(value, list):  # List
            return [self.change_coordinates(v, back) for v in value]
        elif len(value.shape) == 2:
                for t in range(value.shape[0]):  # Array of vectors
                    value[t, :] = self.change_coordinates(value[t, :], back=back)
                return value
        elif len(value.shape) == 1:  # Vector
            if back:
                return P @ value + offset * self.center
            else:
                return P_inv @ (value - offset * self.center)

    def step(self, args, x):
        A, B, d = args
        dx = A @ x +  B @ d
        return x + dx * dt

    def trajectory(self, args, random=True):
        if random:
            t = np.random.rand()
            x = np.array(t*self.x_i[0] + (1-t)*self.x_i[1])
        else:
            x = np.array(self.x0)

        # Forward coordinates change
        xx = np.zeros((np.size(time), np.size(x)))
        A = args
        self.update_coordinates_frame(A)
        x = self.change_coordinates(x)
        A = self.change_coordinates(A, matrix=True)
        B = self.change_coordinates(self.B, matrix=False) if self.B is not None else None


        if self.d is not None:
            if random:
                r, phi = 2*np.random.rand()-1, np.random.rand()
                disturbance = lambda t: r*np.array(self.d(t + phi))
            else:
                disturbance = self.d
        else:
            t = np.random.rand()
            disturbance = lambda x: np.array(t*self.d_i[0] + (1-t)*self.d_i[1])

        for i in range(np.size(time)):
            d = disturbance(time[i])
            d = self.change_coordinates(d, offset=False)
            xx[i] = x
            x = self.step((A, B, d), x)

        # Backward coordinates change
        xx = self.change_coordinates(xx, back=True)
        return xx

    def step_interval(self, x_i, args):
        A0, dAs, d_i = args
        A_i = A0 + sum(intervals_product([0, 1], [dA, dA]) for dA in dAs)
        dx_i = intervals_product(A_i, x_i) + d_i
        return x_i + dx_i*dt

    def step_interval_predictor(self, x_i, args):
        A0, dAs, d_i = args
        dAp = sum(p(dAi) for dAi in dAs)
        dAn = sum(n(dAi) for dAi in dAs)
        Bp = p(self.B)
        Bn = n(self.B)
        x_m, x_M = x_i[0, :, np.newaxis], x_i[1, :, np.newaxis]
        d_m, d_M = d_i[0, :, np.newaxis], d_i[1, :, np.newaxis]
        dx_m = A0 @ x_m - dAp @ n(x_m) - dAn @ p(x_M) + Bp @ d_m - Bn @ d_M
        dx_M = A0 @ x_M + dAp @ p(x_M) + dAn @ n(x_m) + Bp @ d_M - Bn @ d_m
        dx_i = np.array([dx_m.squeeze(axis=-1), dx_M.squeeze(axis=-1)])
        return x_i + dx_i*dt

    def interval_trajectory(self, args=None, predictor=False):
        if args is None:
            args = self.A0, self.dAs, self.d_i
        x_i = self.x_i if self.x_i is not None else np.array([self.x0, self.x0])
        xx_i = np.zeros((np.size(time), 2, np.size(self.x0)))

        # Forward coordinates change
        args = [*args]
        self.update_coordinates_frame(args[0])
        args[0] = self.change_coordinates(args[0], matrix=True)
        args[1] = self.change_coordinates(args[1], matrix=True)
        args[2] = self.change_coordinates(args[2], offset=False)
        x_i = self.change_coordinates(x_i)

        for k in range(np.size(time)):
            xx_i[k, :, :] = x_i
            if predictor:
                x_i = self.step_interval_predictor(x_i, args)
            else:
                x_i = self.step_interval(x_i, args)

        # Backward coordinates change
        xx_i = self.change_coordinates(xx_i, back=True, interval=True)
        return xx_i

    def mesh(self, n):
        vertices = [self.A0 + dAi for dAi in self.dAs]
        if len(vertices) == 1:
            return [vertices[0]]*n
        if len(vertices) == 2:
            return [(1-t) * vertices[0] + t*vertices[1] for t in np.linspace(0, 1, n)]
        for _ in range(2):
            new_vertices = []
            for _ in range(n // 2):
                v_indexes = list(range(len(vertices)))
                v1 = np.random.choice(v_indexes)
                v_indexes.remove(v1)
                v2 = np.random.choice(v_indexes)
                p = np.random.random()
                new_vertices.append(p*vertices[v1] + (1-p)*vertices[v2])
            vertices.extend(new_vertices)
        # plt.scatter([v[1,0] for v in vertices], [v[1,1] for v in vertices])
        # plt.show()
        return vertices

    @staticmethod
    def polytope(A_theta, params_intervals):
        """

        :param A_theta: parametrized matrix function
        :param params_intervals: axes: params, [min, max]
        :return: A0, dA polytope that represents the matrix interval
        """
        params_means = params_intervals.mean(axis=1)
        A0 = A_theta(params_means)
        vertices_id = itertools.product([0, 1], repeat=params_intervals.shape[0])
        dAs = []
        for vertex_id in vertices_id:
            params_vertex = params_intervals[np.arange(len(vertex_id)), vertex_id]
            dAs.append(A_theta(params_vertex) - A_theta(params_means))
        dAs = list({dAi.tostring(): dAi for dAi in dAs}.values())
        return A0, dAs

    @property
    def tau(self):
        return 4/np.min(np.abs(np.linalg.eigvals(self.A0)))

    def asymptotic_bound(self, frequency, eps=0.05):
        return np.absolute(np.linalg.inv(1j*frequency*np.eye(self.A0.shape[0]) - self.A0) @ np.squeeze(self.B)) \
               * np.max(self.d_i) + eps


class LPV(LP):
    def trajectory(self, args):
        x = np.array(self.x0)
        xx = np.zeros((np.size(time), np.size(x)))

        # Forward coordinates change
        gains = args
        params = self.params_from_state(x, gains)
        A0 = self.A_theta(params)
        self.update_coordinates_frame(A0)
        A0 = self.change_coordinates(A0, matrix=True)
        x = self.change_coordinates(x)

        for i in range(np.size(time)):
            xx[i] = x
            x = self.step(A0, x)

        # Backward coordinates change
        xx = self.change_coordinates(xx, back=True)
        return xx

    def interval_trajectory(self, args=None, predictor=False):
        x_i = np.array([self.x0, self.x0])
        xx_i = np.zeros((np.size(time), 2, np.size(self.x0)))

        # Forward coordinates change
        if args is None:
            args = self.gains_i
        params = self.params_from_state_interval(x_i, args)
        A0, dAs = LPV.polytope(self.A_theta, params)
        self.update_coordinates_frame(A0)
        A0 = self.change_coordinates(A0, matrix=True)
        dAs = self.change_coordinates(dAs, matrix=True)
        x_i = self.change_coordinates(x_i)

        for k in range(np.size(time)):
            xx_i[k, :, :] = x_i
            if predictor:
                x_i = self.step_interval_predictor(x_i, (A0, dAs))
            else:
                x_i = self.step_interval(x_i, (A0, dAs))

        # Backward coordinates change
        xx_i = self.change_coordinates(xx_i, back=True, interval=True)
        return xx_i

    def params_from_state(self, x, gains):
        raise NotImplementedError()

    def params_from_state_interval(self, x_i, gains_i):
        raise NotImplementedError()

    def A_theta(self, params):
        raise NotImplementedError()

    def mesh(self, n):
        m = mesh_box(self.gains_i.transpose(), n)
        return zip(*[i.flatten() for i in m])
