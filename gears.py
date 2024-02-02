import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

TAU = 2 * np.pi

def binary_search(fun, minimum, maximum, target, N=20, visualize=False):
    if visualize:
        M = 100
        xs = np.linspace(minimum, maximum, M)
        results = []
        for i in range(M):
            result = fun(xs[i])
            results.append(result)
        plt.figure()
        plt.plot(xs, results, '*')
        plt.show()

    rmin = fun(minimum)
    rmax = fun(maximum)

    if rmin > rmax:
        # this is so ugly
        def temp(x):
            res = fun(x)
            return 2*target - res
        fun2 = temp
        rmin, rmax = rmax, rmin
    else:
        fun2 = fun

    if not (rmin <= target <= rmax):
        raise ValueError('You need to change your bounds on this binary search')


    def binary_search_core(fun, target, a, b, N):
        if N == 0:
            return a

        c = (a+b)/2
        res = fun(c)
        if res < target:
            return binary_search_core(fun, target, c, b, N-1)
        else:
            return binary_search_core(fun, target, a, c, N-1)

    x = binary_search_core(fun2, target, minimum, maximum, N)
    return x

class Gear:
    def __init__(self, radius_vs_theta, rotation_schedule=None, center_schedule=None):
        self.radius_vs_theta = radius_vs_theta

        if rotation_schedule is None:
            self.rotation_schedule = lambda t: 1.0*t
        else:
            self.rotation_schedule = rotation_schedule

        if center_schedule is None:
            self.center_schedule = np.vectorize(lambda t: np.array([0, 0]), signature='()->(2)')
        else:
            self.center_schedule = center_schedule

        self.N = 1024

        thetas = np.linspace(0, 1, self.N+1)
        rs = self.radius_vs_theta(thetas)
        length = 0
        lengths = []
        for i in range(self.N):
            lengths.append(length)
            dl = rs[i] * (thetas[i+1]-thetas[i]) * TAU
            length += dl
        lengths.append(length)
        self.total_length = length
        self.theta_vs_length = lambda length: np.interp(length, lengths, thetas, period=self.total_length)



    def plot(self):
        xs, ys = self.get_curve_points()
        plt.figure()
        plt.plot(xs, ys)
        plt.plot([0], [0], 'x')
        plt.show()

    def get_curve_points(self, time=0):
        thetas = np.arange(0, 1, 1/self.N)
        rs = self.radius_vs_theta(thetas)
        rotation = self.rotation_schedule(time)
        center = self.center_schedule(time)
        xs = rs * np.cos(thetas*TAU + rotation*TAU)
        ys = rs * np.sin(thetas*TAU + rotation*TAU)

        center = self.center_schedule(time)
        temp = np.stack((xs, ys), 1)
        temp += center
        xs, ys = temp.T

        return xs, ys

    def get_spoke_points(self, time=0):
        #lengths = np.linspace(0, self.total_length, 32, endpoint=False)
        lengths = np.arange(0, self.total_length, 0.5)
        thetas = self.theta_vs_length(lengths)
        rs = self.radius_vs_theta(thetas)

        rotation = self.rotation_schedule(time)
        xs_r = rs * np.cos(thetas*TAU + rotation*TAU)
        ys_r = rs * np.sin(thetas*TAU + rotation*TAU)

        xs = np.vstack([xs_r, np.zeros(xs_r.shape)]).T.reshape([-1])
        ys = np.vstack([ys_r, np.zeros(xs_r.shape)]).T.reshape([-1])

        center = self.center_schedule(time)
        temp = np.stack((xs, ys), 1)
        temp += center
        xs, ys = temp.T

        return xs, ys

    def set_up_animation(self, ax):
        curve, = ax.plot([0, 5], [0, 5])
        spokes, = ax.plot([0, 3], [0, 3])
        ax.plot([0], [0], 'x')
        SIZE = 9
        ax.set_xlim([-SIZE, SIZE])
        ax.set_ylim([-SIZE, SIZE])
        def update(frame_time):
            xs, ys = self.get_curve_points(frame_time)
            curve.set_data(xs, ys)
            xs_s, ys_s = self.get_spoke_points(frame_time)
            spokes.set_data(xs_s, ys_s)
            return [curve, spokes]
        return update


    @classmethod
    def animate(cls, gears):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_aspect('equal')

        update_functions = []
        for g in gears:
            u = g.set_up_animation(ax)
            update_functions.append(u)

        def update(frame_time):
            things = []
            for u in update_functions:
                things += u(frame_time)
            return things
        #update = self.set_up_animation(ax)
        ani = FuncAnimation(fig, partial(update), frames=np.arange(0, 1, 1/200),
                            blit=True, interval=33)
        plt.show()

    def get_meshing_gear(self):
        assert self.radius_vs_theta is not None
        ts = np.linspace(0, 1, self.N, endpoint=False)
        rotations = self.rotation_schedule(ts)
        centers = self.center_schedule(ts)


        def function(R):
            new_g_center = np.array([R, 0])
            new_center_schedule = np.vectorize(lambda t: new_g_center, signature='()->(2)')
            #new_centers = np.array([[R, 0]]*self.N)
            new_centers = new_center_schedule(ts)

            return self.get_meshing_gear_attempt(ts, rotations, centers, new_centers)

        def fun(R):
            res, _ =  function(R)
            return res


        R = binary_search(fun, 3.5, 8, 1, visualize=True)

        new_g_center = np.array([R, 0])
        new_center_schedule = np.vectorize(lambda t: new_g_center, signature='()->(2)')
        new_contact_local, (new_radius_vs_theta, new_rotation_schedule) = function(R)
        print(new_contact_local)

        g = Gear(new_radius_vs_theta, new_rotation_schedule, new_center_schedule)
        return g

    def get_meshing_gear_attempt(self, ts, rotations_global, centers, new_centers, outer=False, new_outer=False):
        # imagine a line through both gear centers.
        # The contact point could be in between them: outer=False, new_outer=False,
        # it could be on the other side of new's center far from self: outer=True, new_outer=False
        # it could be on the other side of self's center far from new: outer=False, new_outer=True
        assert not (outer and new_outer)
        N = len(rotations_global)
        assert len(centers) == N
        assert len(new_centers) == N

        # rotation holds the rotation of the self gear in a global context
        # contact_global holds the angle at which the new contacts self in a global context
        # contact_local holds the angle at which new contacts self relative to a reference point on self

        # we don't get to define the reference for new_rotation_global. Rather, we define the reference
        # for new_contact_local as 0, and that ends up being wherever the new gear first contacts
        # the self gear. That same point is used as the reference for new_rotation_global.
        new_rotations_global = []
        # new_contact_local is how we keep track of reference points for new_rs
        new_contact_local = 0
        new_contacts_local = []
        new_rs = []
        for i in range(N+1):
            x0, y0 = centers[i%N]
            x1, y1 = new_centers[i%N]
            rotation_global = rotations_global[i%N]

            # line through centers: [x0 + u*(x1-x0), y0 + u*(y1-y0)]
            # NOTE it would be okay to wiggle contact-global a bit, but I will leave it here
            # contact_global is the angle of the ray from self center to contact point
            contact_global_init = np.arctan2(y1-y0, x1-x0) / TAU
            if outer:
                contact_global = contact_global_init + 0.5
            else:
                contact_global = contact_global_init
            if new_outer:
                new_contact_global = contact_global_init
            else:
                new_contact_global = contact_global_init + 0.5

            contact_local = contact_global - rotation_global
            # NOTE we cannot calculate new_contact local like this. The correct way to do it is
            # to think about gear meshing to set new_contact_local, then use that and new_contact_local
            # to set new_rotation_global.
            # new_contact_local = new_contact_global - new_rotation_global


            # we really just need contact_local_prev in order to do this block
            if i != 0:
                r = self.radius_vs_theta(contact_local)
                dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                if outer:
                    new_r = dist + r

                else:
                    new_r = dist - r

                # NOTE I think technically this mod could hide a bug ... but I think your settings
                #  would have to be super wrong for the gear to try moving more than .5 in one step
                d_contact_local = (contact_local_prev - contact_local + 0.5) % 1.0 - 0.5
                d_new_contact_local = d_contact_local * r / new_r
                new_contact_local += d_new_contact_local
                new_rotation_global = new_contact_global - new_contact_local

                new_rotations_global.append(new_rotation_global)
                new_contacts_local.append(new_contact_local)
                new_rs.append(new_r)

            contact_local_prev = contact_local

        new_radius_vs_theta = lambda theta: np.interp(theta, new_contacts_local, new_rs, period=1)
        new_rotation_schedule = lambda t: np.interp(t, ts, new_rotations_global, period=1)

        return new_contact_local, (new_radius_vs_theta, new_rotation_schedule)





#r_vs_t = lambda t: np.cos(t * TAU) + 2
def r_vs_t(t):
    t = t%1
    if t < 0.2:
        return 1.0
    elif t < 0.6:
        return 1.0 + (3 - 1) / (.6-.2) * (t-.2)
    else:
        return 3.0
r_vs_t = np.vectorize(r_vs_t)
g = Gear(r_vs_t)
#g.animate()
#exit()

match = g.get_meshing_gear()
#match.animate()
Gear.animate([g, match])