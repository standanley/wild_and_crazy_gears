from gear import Gear
import numpy as np
import matplotlib.pyplot as plt


class Gear3D(Gear):


    def get_plot_coords(self, center, angle):
        #assert len(center) == 3
        M = 4 * 5 * 7 * 4

        # remember that in the move from 2D to 3D, r gets replaced by a
        sample_thetas, sample_as = self.get_r_vs_theta(M)

        sample_thetas = np.concatenate((sample_thetas, [sample_thetas[0]]))
        sample_as = np.concatenate((sample_as, [sample_as[0]]))
        #sample_thetas_transformed = self.transform_thetas(sample_thetas)

        # center x coord is interpreted as angle from +z to axis in xz plane.
        # a is interpreted as angle from axis to point
        # distance from origin to point is always 1 (for now)

        def transform(theta, a):
            # a is gear shape; contact point's angle from axis
            # theta is also gear shape, location of this point relative to reference tooth
            # angle is the rotation of the gear about hte axis
            # axis_angle is the rotation of the axis itself about the origin
            axis_angle = center[0]
            effective_theta = self.transform_thetas(theta - angle)

            x = a * np.cos(effective_theta) + axis_angle
            y = a * np.sin(effective_theta)
            z = 0
            return x, y, z

            r = 1
            # 1: axis is +z, not rotated
            x1 = r * np.sin(a)
            y1 = 0
            z1 = r * np.cos(a)
            # 2: axis is +z, rotated
            x2 = x1 * np.cos(effective_theta) - y1 * np.sin(effective_theta)
            y2 = x1 * np.sin(effective_theta) + y1 * np.cos(effective_theta)
            z2 = z1
            # 3: axis is according to axis_angle, rotated
            x3 = x2 * np.cos(axis_angle) + z2 * np.sin(axis_angle)
            y3 = y2
            z3 = -x2 * np.sin(axis_angle) + z2 * np.cos(axis_angle)
            return x3, y3, z3

        transform_vec = np.vectorize(transform, signature='(),()->(),(),()')

        xs, ys = self.polar_to_rect(self.transform_thetas(self.thetas - angle), self.rs, center)
        xs_fine, ys_fine = self.polar_to_rect(self.transform_thetas(sample_thetas - angle), sample_as, center)
        zs = [0]*len(xs)
        zs_fine = [0]*len(xs_fine)
        #xs, ys, zs = transform_vec(self.thetas, self.rs)
        #xs_fine, ys_fine, zs_fine = transform_vec(sample_thetas, sample_as)
        return xs, ys, zs, xs_fine, ys_fine, zs_fine

    def plot(self, ax=None):
        SIZE = max(self.rs) * 1.02
        if ax is None:
            # NOTE this block is not used when the plotting starts from Assembly.animate
            fig = plt.figure()
            ax = fig.axes(projection='3d')
            #ax.set_xlim_3d([-SIZE, SIZE])
            #ax.set_ylim_3d([-SIZE, SIZE])
            #ax.set_zlim_3d([-SIZE, SIZE])
            ax.set(xlim3d=(-SIZE, SIZE))
            ax.set(ylim3d=(-SIZE, SIZE))
            ax.set(zlim3d=(-SIZE, SIZE))
            #ax.set_aspect('equal')

        xs, ys, zs, xs_fine, ys_fine, zs_fine = self.get_plot_coords([0, 0], 0)

        fine, = ax.plot(xs_fine, ys_fine, '--')
        #coarse, = ax.plot(xs, ys, '*')
        coarse, = ax.plot([], [], [], '*')
        #point, = ax.plot([0], [0], [1], '+')
        # TODO this is a hack because ax.set(xlim3d=()) is not working
        point, = ax.plot([-SIZE, SIZE], [-SIZE, SIZE], [-SIZE, SIZE], '+')

        return [fine, coarse, point]

    def update_plot(self, center, angle, curves):
        fine, coarse, point = curves
        xs, ys, zs, xs_fine, ys_fine, zs_fine = self.get_plot_coords(center, angle)
        fine.set_data_3d(xs_fine, ys_fine, zs_fine)
        coarse.set_data_3d(xs, ys, zs)
        #point.set_data_3d([center[0]], [center[1]], [1])
        point.set_data_3d([0], [0], [0])
        return [fine, coarse, point]
