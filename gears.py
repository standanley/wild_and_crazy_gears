import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

TAU = 2 * np.pi
DEBUG = False

# add functionality for period_y for cases where y wraps, such as wheel rotation as a function of time
def interp(xs, ys, period, period_y=None):
    expected_period = abs(xs[-1] - xs[0])
    period = abs(period)
    #if expected_period > period:
    #    raise ValueError('something else weird in interp')
    xs = np.array(xs)
    ys = np.array(ys)
    if period_y is None:
        xs_new = xs
        ys_new = ys
    else:
        # I wish I didn't need to bump the period by EPS, but np.interp does weird
        # things when I don't
        EPS = 1e-10
        if xs[0] > xs[-1]:
            period_temp = -1 * (period-EPS)
        else:
            period_temp = period - EPS
        if ys[0] > ys[-1]:
            period_y_temp = -1 * period_y
        else:
            period_y_temp = period_y
        if (xs[0] + period_temp > xs[-1]) != (xs[0] > xs[-1]):
            xs_new = np.append(xs, xs[0]+period_temp)
            ys_new = np.append(ys, ys[0]+period_y_temp)
        else:
            xs_new = xs
            ys_new = ys

    diffs = np.diff(xs_new)
    if not (all(diffs >= 0) or all(diffs <= 0)):
        raise ValueError('something weird in interp')


    if DEBUG:
        #xs_new[-1] -= 1e-6
        temp = lambda x: np.interp(x, xs_new, ys_new, period=period)
        xs_fake = np.linspace(-0.1, 0.4, 30000)
        ys_fake = temp(xs_fake)
        plt.plot(xs_fake, ys_fake, '*')
        plt.plot(xs_new, ys_new, '+')
        plt.show()

    return lambda x: np.interp(x, xs_new, ys_new, period=period)





def binary_search_core(fun, target, a, b, N):
    # assumes fun is increasing
    if N == 0:
        return a

    c = (a + b) / 2
    try:
        res = fun(c)
    except ValueError:
        res = target * float('inf')
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
            try:
                result = fun(xs[i])
            except ValueError:
                result = 0
            results.append(result)
        plt.figure()
        plt.plot(xs, results, '*')
        plt.show()

    try:
        rmin = fun(minimum)
    except ValueError:
        rmin = target * float('inf')
    try:
        rmax = fun(maximum)
    except ValueError:
        rmax = target * float('inf')

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
        raise ValueError('You need to change your bounds on this search')

    x = binary_search_core(fun2, target, minimum, maximum, N)
    print('winning parameter', x)
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
        #self.N = 96

        thetas = np.linspace(0, 1, self.N+1)
        rs = self.radius_vs_theta(thetas)
        length = 0
        lengths = []
        for i in range(self.N):
            lengths.append(length)
            dl = rs[i] * (thetas[i+1]-thetas[i]) * TAU
            length += dl

        # I used to put an extra point here at the end so it matches the beginning exactly,
        # but now I let interp() handle that because of its EPS handling
        #lengths.append(length)
        self.total_length = length
        self.theta_vs_length = interp(lengths, thetas[:-1], period=self.total_length, period_y=1)



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
        SIZE = 4
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
            #new_g_center = np.array([R, 0])
            #new_center_schedule = np.vectorize(lambda t: new_g_center, signature='()->(2)')


            new_center_schedule = lambda t: np.array([R*np.cos(-t*TAU), R*np.sin(-t*TAU)])
            new_center_schedule = np.vectorize(new_center_schedule, signature='()->(2)')

            return MeshingInfo(self, new_center_schedule,
                               new_num_rotations=1, num_rotations=3,
                               new_outer=False,
                               outer=False)

        # binary search parameters are annoying to keep changing
        res = self.get_meshing_gear(get_mi, 3, 10)
        new_g = Gear(res.new_radius_vs_theta, res.new_rotation_schedule, res.new_center_schedule)
        return new_g

    @classmethod
    def get_meshing_gear(cls, get_mi, param_min, param_max):
        mi_min, mi_max = get_mi(param_min), get_mi(param_max)
        target_flip = -1 if mi_min.outer ^ mi_min.new_outer else 1
        target = 1 / mi_min.new_num_rotations * mi_min.num_rotations * target_flip

        def fun(param):
            mi = get_mi(param)
            res = cls.get_meshing_gear_attempt(mi)
            return res.new_contact_local

        param_opt = binary_search(fun, param_min, param_max, target, visualize=False)
        global DEBUG
        DEBUG = True
        res = cls.get_meshing_gear_attempt(get_mi(param_opt))

        # TODO I think I don't need this block anymore because it's taken care of
        #  inside get_meshing_gear_attempt now
        #if mi_min.new_num_rotations != 1:
        #    num = mi_min.new_num_rotations
        #    old_rvt = res.new_radius_vs_theta
        #    def radius_vs_theta(theta):
        #        # TODO I don't think the wrapping is perfect here
        #        return old_rvt(theta % (1/num))
        #    res.new_radius_vs_theta = radius_vs_theta

        return res


    @classmethod
    def get_meshing_gear_attempt(cls, mi):
        # imagine a line through both gear centers.
        # The contact point could be in between them: outer=False, new_outer=False,
        # it could be on the other side of new's center far from self: outer=True, new_outer=False
        # it could be on the other side of self's center far from new: outer=False, new_outer=True
        assert not (mi.outer and mi.new_outer)
        r_finished = False

        N = mi.g.N
        # first, get ts such that they run through the right amount of g's rotation schedule
        nr = mi.num_rotations
        #if nr == 1:
        #    t_max = 1
        #else:
        #    # if max guess is too close to 1 (relative to g.N) then I think we might have issues
        #    MAX_GUESS = 0.9
        #    t_max = binary_search_core(mi.g.rotation_schedule, 1/nr, 0, 0.9, 50)
        # TODO I think we need to search for this end point as we go. Rotation schedule is no good because sometimes
        #  movement is caused by the center_schedule, and I'm also not sure rotation schedule is good if this gear
        #  is the result of other weird stuff. I think this will work for now, though
        #  UPDATE: I think it's no good because we need to finish the whole cycle to get all the schedules
        #t_max = 1/nr
        t_max = 1

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
                new_contact_global = contact_global_init
            else:
                new_contact_global = contact_global_init + 0.5
            if mi.new_outer:
                contact_global = contact_global_init + 0.5
            else:
                contact_global = contact_global_init

            contact_local = contact_global - rotation_global
            # NOTE we cannot calculate new_contact local like this. The correct way to do it is
            # to think about gear meshing to set new_contact_local, then use that and new_contact_local
            # to set new_rotation_global.
            # new_contact_local = new_contact_global - new_rotation_global


            # we really just need contact_local_prev in order to do this block
            if i != 0:
                r = mi.g.radius_vs_theta(contact_local)
                dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                if mi.new_outer:
                    new_r = dist + r
                elif mi.outer:
                    new_r = r - dist
                else:
                    new_r = dist - r


                # NOTE I think technically this mod could hide a bug ... but I think your settings
                #  would have to be super wrong for the gear to try moving more than .5 in one step
                period = 1/mi.num_rotations
                d_contact_local = (contact_local_prev - contact_local + (period/2)) % period - (period/2)
                new_contact_ratio_flip = -1 if mi.new_outer ^ mi.outer else 1
                d_new_contact_local = d_contact_local * r / new_r * new_contact_ratio_flip
                new_contact_local += d_new_contact_local
                new_rotation_global = new_contact_global - new_contact_local

                new_rotations_global.append(new_rotation_global)

                if not r_finished:
                    new_contacts_local.append(new_contact_local)
                    new_rs.append(new_r)

            contact_local_prev = contact_local

        new_radius_vs_theta = interp(new_contacts_local, new_rs, 1/mi.new_num_rotations)

        # TODO the period_y here should be 1/new_num_rotations, or even better, the
        #  period should be new_num_rotations but we have to duplicate the arrays accordingly
        # keep in mind that the time will end at 1/num_rotations, and we should have gotten through
        new_rotation_schedule = interp(ts, new_rotations_global, period=1, period_y=1/mi.new_num_rotations)

        res = MeshingResult(new_contact_local, new_radius_vs_theta, new_rotation_schedule, mi.new_center_schedule)
        return res




#r_vs_t = lambda t: np.cos(t * TAU) + 2
def r_vs_t(t):
    A = 0.2
    B = 0.6
    C = 2.0
    D = 3.0
    t = 3*(t%(1/3))
    if t < A:
        return C
    elif t < B:
        return C + (D - C) / (B-A) * (t-A)
    else:
        return D

r_vs_t = np.vectorize(r_vs_t)

def g_rotation_schedule(t):
    return 0
g_rotation_schedule = np.vectorize(g_rotation_schedule, signature='()->()')

g = Gear(r_vs_t, rotation_schedule=g_rotation_schedule)
#g.animate()
#exit()

match = g.get_meshing_gear_simple()

ts = np.linspace(-1.5, 2.5, 5000)
rotations = match.rotation_schedule(ts)
plt.plot(ts, rotations, '*')
plt.show()

#match.animate()
Gear.animate([g, match])