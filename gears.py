import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

TAU = 2 * np.pi

# add functionality for period_y for cases where y wraps, such as wheel rotation as a function of time
def interp(xs, ys, period, period_y=None):
    xs = np.array(xs)
    ys = np.array(ys)
    if period_y is None:
        xs_new = xs
        ys_new = ys
    else:
        xs_new = np.append(xs, xs[0]+period)
        ys_new = np.append(ys, ys[0]+period_y)

    diffs = np.diff(xs_new)
    assert all(diffs >= 0) or all(diffs <= 0)

    return lambda x: np.interp(x, xs_new, ys_new, period=period)





def binary_search_core(fun, target, a, b, N):
    # assumes fun is increasing
    if N == 0:
        return a

    c = (a + b) / 2
    res = fun(c)
    if res < target:
        return binary_search_core(fun, target, c, b, N - 1)
    else:
        return binary_search_core(fun, target, a, c, N - 1)


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

    x = binary_search_core(fun2, target, minimum, maximum, N)
    return x



class MeshingInfo:
    def __init__(self, g, new_center_schedule=None, num_rotations=1, new_num_rotations=1, outer=False, new_outer=False):
        self.g = g
        self.new_center_schedule = new_center_schedule
        self.num_rotations = num_rotations
        self.new_num_rotations = new_num_rotations
        self.outer = outer
        self.new_outer = new_outer



class MeshingResult:
    def __init__(self, new_contact_local, new_radius_vs_theta, new_rotation_schedule, new_center_schedule):
        self.new_contact_local = new_contact_local
        self.new_radius_vs_theta = new_radius_vs_theta
        self.new_rotation_schedule = new_rotation_schedule
        self.new_center_schedule = new_center_schedule



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
        self.theta_vs_length = interp(lengths, thetas, period=self.total_length, period_y=1)



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

    def get_meshing_gear_simple(self):

        def get_mi(R):
            new_g_center = np.array([R, 0])
            new_center_schedule = np.vectorize(lambda t: new_g_center, signature='()->(2)')
            return MeshingInfo(self, new_center_schedule)

        res = self.get_meshing_gear(get_mi, 3.5, 8)
        new_g = Gear(res.new_radius_vs_theta, res.new_rotation_schedule, res.new_center_schedule)
        return new_g

    @classmethod
    def get_meshing_gear(cls, get_mi, param_min, param_max):
        mi_min, mi_max = get_mi(param_min), get_mi(param_max)
        target = 1 / mi_min.new_num_rotations

        def fun(param):
            mi = get_mi(param)
            res = cls.get_meshing_gear_attempt(mi)
            return res.new_contact_local

        param_opt = binary_search(fun, param_min, param_max, target)
        return cls.get_meshing_gear_attempt(get_mi(param_opt))


    @classmethod
    def get_meshing_gear_attempt(cls, mi):
        # imagine a line through both gear centers.
        # The contact point could be in between them: outer=False, new_outer=False,
        # it could be on the other side of new's center far from self: outer=True, new_outer=False
        # it could be on the other side of self's center far from new: outer=False, new_outer=True
        assert not (mi.outer and mi.new_outer)

        N = mi.g.N
        # first, get ts such that they run through the right amount of g's rotation schedule
        nr = mi.num_rotations
        if nr == 1:
            t_max = 1
        else:
            # if max guess is too close to 1 (relative to g.N) then I think we might have issues
            MAX_GUESS = 0.9
            t_max = binary_search_core(mi.g.rotation_schedule, 1/nr, 0, 0.9, 50)

        ts = np.linspace(0, t_max, mi.g.N, endpoint=False)

        rotations_global = mi.g.rotation_schedule(ts)
        centers = mi.g.center_schedule(ts)
        new_centers = mi.new_center_schedule(ts)

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
            if mi.outer:
                contact_global = contact_global_init + 0.5
            else:
                contact_global = contact_global_init
            if mi.new_outer:
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
                r = mi.g.radius_vs_theta(contact_local)
                dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                if mi.outer:
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

        new_radius_vs_theta = interp(new_contacts_local, new_rs, 1)
        new_rotation_schedule = interp(ts, new_rotations_global, period=1, period_y=1)

        res = MeshingResult(new_contact_local, new_radius_vs_theta, new_rotation_schedule, mi.new_center_schedule)
        return res





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

match = g.get_meshing_gear_simple()
#match.animate()
Gear.animate([g, match])