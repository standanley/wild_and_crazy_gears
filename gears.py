import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

TAU = 2 * np.pi
DEBUG = False
VISUALIZE = False
DO_VISUALIZE = False


def get_inverse(fun, xmin, xmax, ymin, ymax, N):
    ys = np.linspace(ymin, ymax, N, endpoint=False)
    xs = []
    for y in ys:
        x = binary_search_core(fun, y, xmin, xmax, 10)
        xs.append(x)
    return Interp(ys, xs, ymax-ymin, xmax-xmin)


class Interp:
    # EPS is used in cases where period_y is not None; this class will insert another point at
    # [xs[0] + period_x - EPS, ys[0]+period_y]
    EPS = 1e-10

    def __init__(self, xs, ys, period_x, period_y=None):
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        #self.ys = ys
        self.period_x = period_x
        self.period_y = period_y

        self.N = len(xs)
        assert len(ys) == self.N

        diffs = np.diff(xs)
        self.x_increasing = True
        nonmonotonic = False
        if all(diffs >= 0):
            pass
        elif all(diffs <= 0):
            self.x_increasing = False
        else:
            nonmonotonic = True


        self.xs_interp = xs
        self.ys_interp = ys
        if self.period_y:
            sign = (1 if self.x_increasing else -1)
            x_final = self.xs_interp[0] + sign * (abs(self.period_x) - self.EPS)
            self.xs_interp = np.append(self.xs_interp, x_final)
            self.ys_interp = np.append(self.ys_interp, self.ys_interp[0] + self.period_y)

        self.fun = lambda x: np.interp(x, self.xs_interp, self.ys_interp, period=self.period_x)

        if (DO_VISUALIZE and len(self.ys.shape) == 1):
            self.visualize()
        #assert not nonmonotonic, 'interp is not monotonic'
        if nonmonotonic:
            raise ValueError('Nonmonotonic')

    def visualize(self):
        #xs_fake = np.linspace(-.2*self.period_x, self.period_x*2.2, 30000)
        xs_fake = np.linspace(-2.1, 2.1, 30000)
        ys_fake = self.fun(xs_fake)
        plt.plot(xs_fake, ys_fake, '*')
        plt.plot(self.xs_interp, self.ys_interp, '+')
        plt.show()

    @classmethod
    def from_fun_xs(cls, fun, xs, period_x, period_y=None):
        ys = fun(xs)
        temp = cls(xs, ys, period_x, period_y)

        # we override the normal interpolation function
        temp.fun = fun
        return temp

    @classmethod
    def from_fun(cls, fun, N, xmin, xmax, period_x, period_y=None):
        xs = np.linspace(xmin, xmax, N, endpoint=False)
        return cls.from_fun_xs(fun, xs, period_x, period_y)

    def __call__(self, x):
        return self.fun(x)




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


def binary_search(fun, minimum, maximum, target, N=10, visualize=False):
    if visualize:
        print('target is', target)
        M = 10
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

    def get_gear(self):
        return Gear(self.new_radius_vs_theta, self.new_rotation_schedule, self.new_center_schedule)



class Gear:
    def __init__(self, radius_vs_theta, rotation_schedule=None, center_schedule=None):
        self.radius_vs_theta = radius_vs_theta

        if rotation_schedule is None:
            self.rotation_schedule = lambda t: 1.0*t
            self.rotation_schedule = Interp.from_fun_xs(self.rotation_schedule, self.radius_vs_theta.xs, 1, 1)
        else:
            self.rotation_schedule = rotation_schedule

        if center_schedule is None:
            self.center_schedule = np.vectorize(lambda t: np.array([0, 0]), signature='()->(2)')
            self.center_schedule = Interp.from_fun_xs(self.center_schedule, self.rotation_schedule.xs, 1)
        else:
            self.center_schedule = center_schedule

        thetas = self.radius_vs_theta.xs
        rs = self.radius_vs_theta.ys
        length = 0
        lengths = []
        for i in range(len(thetas)):
            lengths.append(length)
            j = (i+1) % len(thetas)
            # TODO I guess it would be better to do the actual distance ... but I'm not sure yet how I want to deal
            #  with discontinuities. Also, so far this is only used for visualization
            dl = abs((rs[i] + rs[j])/2 * ((thetas[j]-thetas[i]+0.5)%1-0.5) * TAU)
            length += dl

        # I used to put an extra point here at the end so it matches the beginning exactly,
        # but now I let interp() handle that because of its EPS handling
        #lengths.append(length)
        self.total_length = length
        period_y = np.round(thetas[-1] - thetas[0])
        self.theta_vs_length = Interp(lengths, thetas, self.total_length, period_y=period_y)
        #print('done doing length')

    def clone(self):
        return Gear(self.radius_vs_theta, self.rotation_schedule, self.center_schedule)

    def time_shift(self, dt):
        rotation_orig = self.rotation_schedule
        center_orig = self.center_schedule
        self.rotation_schedule = lambda t: rotation_orig(t-dt)
        self.center_schedule = lambda t: center_orig(t-dt)

    def time_warp(self, warp_fun):
        rotation_orig = self.rotation_schedule
        center_orig = self.center_schedule
        self.rotation_schedule = lambda t: rotation_orig(warp_fun(t))
        self.center_schedule = lambda t: center_orig(warp_fun(t))

    def plot(self):
        xs, ys = self.get_curve_points()
        plt.figure()
        plt.plot(xs, ys)
        plt.plot([0], [0], 'x')
        plt.show()

    def get_curve_points(self, time=0):
        thetas = self.radius_vs_theta.xs
        if True or len(self.radius_vs_theta.xs) > 100:
            DRAW_N = 100
            thetas = np.linspace(0, 1, DRAW_N, endpoint=False)
        thetas = np.append(thetas, thetas[0])
        rs = self.radius_vs_theta(thetas)
        rotation = self.rotation_schedule(time)
        xs = rs * np.cos(thetas*TAU + rotation*TAU)
        ys = rs * np.sin(thetas*TAU + rotation*TAU)

        center = self.center_schedule(time)
        temp = np.stack((xs, ys), 1)
        temp += center
        xs, ys = temp.T

        return xs, ys

    def get_spoke_points(self, time=0):
        #lengths = np.linspace(0, self.total_length, 32, endpoint=False)
        lengths = np.arange(0, self.total_length, 0.15)
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
        curve, = ax.plot([0, 5], [0, 5], '+')
        #spokes, = ax.plot([0, 3], [0, 3])
        #ax.plot([0], [0], 'x')
        SIZE = 2
        ax.set_xlim([-SIZE, SIZE])
        ax.set_ylim([-SIZE, SIZE])
        def update(frame_time):
            xs, ys = self.get_curve_points(frame_time)
            curve.set_data(xs, ys)
            xs_s, ys_s = self.get_spoke_points(frame_time)
            #spokes.set_data(xs_s, ys_s)
            return [curve]#, spokes]
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
                            blit=True, interval=100)
        plt.show()


    @classmethod
    def get_meshing_gear(cls, get_mi, param_min, param_max, visualize=False):
        mi_min, mi_max = get_mi(param_min), get_mi(param_max)

        # target might be positive or negative, but it's too difficult to determine that
        # from mi settings alone, so I always make it positive and abs() the result in fun
        target = 1 / mi_min.new_num_rotations

        def fun(param):
            mi = get_mi(param)
            res = cls.get_meshing_gear_attempt(mi)
            return abs(res.new_contact_local)

        param_opt = binary_search(fun, param_min, param_max, target, visualize=visualize)
        global DEBUG
        #DEBUG = True
        res = cls.get_meshing_gear_attempt(get_mi(param_opt))
        # TODO this way of noting param_opt is a bit hacky
        res.param_opt = param_opt

        return res

    @classmethod
    def animate_meshing_gear_attempt(cls, mi):

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_aspect('equal')
        SIZE = 4
        ax.set_xlim([-SIZE, SIZE])
        ax.set_ylim([-SIZE, SIZE])
        gear, = ax.plot([], [], '*')
        new_gear, = ax.plot([], [], '*')
        points, = ax.plot([], [], 'x')

        gen = cls.get_meshing_gear_attempt_iterator(mi, animate=True)

        def set_curve(curve, center, thetas, rs, new_rotation_global):
            rs = np.array(rs)
            thetas = np.array(thetas)
            xs = center[0] + rs*np.cos((thetas + new_rotation_global)*TAU)
            ys = center[1] + rs*np.sin((thetas + new_rotation_global)*TAU)
            curve.set_data(xs, ys)

        def update(time):
            try:
                data = gen.__next__()
            except StopIteration as e:
                return [gear, new_gear, points]

            (center, thetas, rs, new_rotation_global,
                new_center, new_thetas, new_rs, new_new_rotation_global,
                pointsx, pointsy) = data
            set_curve(gear, center, thetas, rs, new_rotation_global)
            set_curve(new_gear, new_center, new_thetas, new_rs, new_new_rotation_global)
            points.set_data(pointsx, pointsy)

            return [gear, new_gear, points]

        ani = FuncAnimation(fig, partial(update), frames=np.arange(0, 1, 1/200),
                            blit=True, interval=100)
        plt.show()
        exit()

    @classmethod
    def get_meshing_gear_attempt(cls, mi):
        try:
            next(cls.get_meshing_gear_attempt_iterator(mi))
            assert False, 'should have thrown StopIteration immediately'
        except StopIteration as e:
            return e.value


    @classmethod
    def get_meshing_gear_attempt_iterator(cls, mi, animate=False):
        # imagine a line through both gear centers.
        # The contact point could be in between them: outer=False, new_outer=False,
        # it could be on the other side of new's center far from self: outer=True, new_outer=False
        # it could be on the other side of self's center far from new: outer=False, new_outer=True
        assert not (mi.outer and mi.new_outer)
        r_finished = False
        r_one_more = False

        ts = np.unique(np.concatenate([
            mi.g.rotation_schedule.xs,
            mi.g.center_schedule.xs,
            mi.new_center_schedule.xs
        ]))
        N = len(ts)

        rotations_global = mi.g.rotation_schedule(ts)
        centers = mi.g.center_schedule(ts)
        new_centers = mi.new_center_schedule(ts)

        # only used in animation; we don't yet know the theta sampling we'll net to get r
        #thetas_animation = np.linspace(0, 1, 1001)
        thetas_animation = mi.g.radius_vs_theta.xs
        rs_animation = mi.g.radius_vs_theta(thetas_animation)

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
        first_contact_local = None
        new_rotation_global_prev = None
        rotation_global_prev = None
        new_contact_local_prev = None
        contact_local_prev = None
        contact_global_prev = None
        new_r_prev = None
        r_prev = None
        final_new_contact_local = None
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
            if i == 0:
                first_contact_local = contact_local
            # NOTE we cannot calculate new_contact local like this. The correct way to do it is
            # to think about gear meshing to set new_contact_local, then use that and new_contact_local
            # to set new_rotation_global.
            # new_contact_local = new_contact_global - new_rotation_global

            r = mi.g.radius_vs_theta(contact_local)
            dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            if mi.new_outer:
                new_r = dist + r
            elif mi.outer:
                new_r = r - dist
            else:
                new_r = dist - r

            # we really just need contact_local_prev in order to do this block
            if i != 0:
                # FIRST we add new data to new_rotations_global, new_contacts_local, and new_rs
                new_rotations_global.append(new_rotation_global_prev)
                if not r_finished:
                    new_contacts_local.append(new_contact_local_prev)
                    new_rs.append(new_r_prev)
                    if r_one_more:
                        r_finished = True

                if animate:
                    center = centers[(i-1) % N]
                    new_center = new_centers[(i-1) % N]
                    pointsx = [center[0], new_center[0], center[0] + r_prev * np.cos(contact_global_prev * TAU)]
                    pointsy = [center[1], new_center[1], center[1] + r_prev * np.sin(contact_global_prev * TAU)]
                    data = [center, thetas_animation, rs_animation, rotation_global_prev,
                            new_center, new_contacts_local, new_rs, new_rotation_global_prev,
                            pointsx, pointsy]
                    yield data
                # SECOND we think about how much the new gear needs to spin to get from prev to current
                # The old gear spins from r_prev to r over a rotation of d_contact_local
                # the new gear is going from new_r_prev to new_r at the same time
                # NOTE I think technically this mod could hide a bug ... but I think your settings
                #  would have to be super wrong for the gear to try moving more than .5 in one step
                period = 1/mi.num_rotations
                d_contact_local = (contact_local - contact_local_prev + (period/2)) % period - (period/2)
                new_contact_ratio_flip = 1 if mi.new_outer ^ mi.outer else -1
                # TODO I'm averaging r and r_prev here, but I think the proper math is more complicated
                contact_rate_ratio = (r+r_prev) / (new_r+new_r_prev) * new_contact_ratio_flip
                if abs(contact_rate_ratio) > 100:
                    # this is no good, probably new_radius is hitting zero
                    raise ValueError('Gear ratio got above 100 to', contact_rate_ratio, 'old r', r, 'new r', new_r)
                d_new_contact_local = d_contact_local * contact_rate_ratio
                new_contact_local += d_new_contact_local


                # when new gear is created from an old gear with multiple repetitions we do not want the new
                # gears radius to be formed from copies of each time around the old gear, so we end early
                #if len(new_contacts_local) > 0 and abs(new_contact_local - new_contacts_local[0]) > 1:
                #    r_finished = True
                fraction_contacted = contact_local - first_contact_local
                fraction_contacted_desired = 1/mi.num_rotations
                if not r_finished and abs(fraction_contacted) > fraction_contacted_desired:
                    # At this point we are done defining our new gear's shape, but we will continue
                    #  thinking because we might not be done defining the rotation schedule
                    # NOTE we do not check that additional times around defining this gear match up;
                    # in fact we haven't even checked that we are stopping in the same place we started
                    # because that's often not the case when we are still searching for the right params
                    r_one_more = True
                    # print('finishing contact at center', x1, y1)
                    # print('finishing contact at new contact local (before correction', new_contact_local)
                    # We want to prorate final_new_contact_local to account for partial completion of
                    # the next contact_local chunk. We already added d_new_contact_local which put us
                    # over the line, so let's take some of that out depending on how far over we were
                    contact_overdone = (fraction_contacted - fraction_contacted_desired
                                        if fraction_contacted > 0 else
                                        fraction_contacted + fraction_contacted_desired)
                    final_new_contact_local = new_contact_local - contact_overdone * contact_rate_ratio
                    #print('after correction', final_new_contact_local)
                if r_one_more and r_finished:
                    r_one_more = False

            # this needs to happen after new_contact_local has been updated,
            # but also needs to happen on the first time around
            new_rotation_global = new_contact_global - new_contact_local
            if len(new_rotations_global) > 0:
                diff = new_rotation_global - new_rotations_global[-1]
                diff_shift = (diff + 0.5) % 1.0 - 0.5
                new_rotation_global = new_rotations_global[-1] + diff_shift

            contact_local_prev = contact_local
            contact_global_prev = contact_global
            new_contact_local_prev = new_contact_local
            new_r_prev = new_r
            r_prev = r
            new_rotation_global_prev = new_rotation_global
            rotation_global_prev = rotation_global

        new_radius_vs_theta = Interp(new_contacts_local, new_rs, 1/mi.new_num_rotations)

        # TODO the period_y here should be 1/new_num_rotations, or even better, the
        #  period should be new_num_rotations but we have to duplicate the arrays accordingly
        # keep in mind that the time will end at 1/num_rotations, and we should have gotten through
        #print('period y', 1/mi.new_num_rotations)
        # TODO actually I think calculation this is complicated
        snap = 1/mi.new_num_rotations
        period_y = np.round((new_rotations_global[-1] - new_rotations_global[0])/snap)*snap
        new_rotation_schedule = Interp(ts, new_rotations_global, 1, period_y=period_y)

        #if abs(final_new_contact_local - 0.5) < 0.2:
        #    print('final new contact local', final_new_contact_local)
        if final_new_contact_local is None:
            # we somehow didnt make it to the end of the old gear within the time
            final_new_contact_local = new_contact_local
        res = MeshingResult(final_new_contact_local, new_radius_vs_theta, new_rotation_schedule, mi.new_center_schedule)
        return res



#SIMPLE_N = 50
#rvt1 = lambda t: 1+0.5*np.sin(t*TAU)
#rvt1 = Interp.from_fun(rvt1, SIMPLE_N, 0, 1, 1, 1)
#g1 = Gear(rvt1)
#
#
#def get_mi_g2(R):
#    new_g_center = np.array([R, 0])
#    new_center_schedule = np.vectorize(lambda t: new_g_center, signature='()->(2)')
#
#    #new_center_schedule = lambda t: np.array([R * np.cos(-t * TAU), R * np.sin(-t * TAU)])
#    #new_center_schedule = np.vectorize(new_center_schedule, signature='()->(2)')
#
#    new_center_schedule = Interp.from_fun(new_center_schedule, SIMPLE_N, 0, 1, 1)
#
#    return MeshingInfo(g1, new_center_schedule,
#                       new_num_rotations=2, num_rotations=1,
#                       new_outer=False,
#                       outer=False)
#
#
##res = g1.get_meshing_gear(get_mi_g2, 2, 6, visualize=True)
#res = g1.get_meshing_gear_attempt(get_mi_g2(3.172))
#g2 = res.get_gear()
#Gear.animate([g1, g2])
#exit()



PLANETARY_R = 5
PLANETARY_P = 1
PLANETARY_S = 1
PLANET_STRIDE = 2
assert (PLANETARY_R + PLANETARY_S) % PLANET_STRIDE == 0

PLANET_N = 80
RING_N = PLANETARY_R*PLANET_N

def get_planetary_attempt(param):
    ############## RING GEAR ##############

    #r_vs_t = lambda t: np.cos(t * TAU) + 2
    def r_vs_t_old(t):
        A = 0.2
        B = 0.6
        C = 2.0
        D = 3.0
        N = 4
        t = N*(t%(1/N))
        if t < A:
            return C
        elif t < B:
            return C + (D - C) / (B-A) * (t-A)
        else:
            return D

    def r_vs_t_old2(t):
        N = 4
        A = param # -0.4
        B = 0.1
        t = N*((t+0.125)%(1/N))
        wave = np.sin(t*TAU) + A*np.sin(2*t*TAU-0.7) + B*np.sin(3*t*TAU)
        return wave/2+3

    def get_r_vs_t_smooth(N):
        rotations = PLANETARY_R
        if N%rotations != 0:
            print('WARNING: N is not a multiple of rotations in get_r_vs_t_smooth')
        #t = rotations*((t+0.125)%(1/rotations))

        points = np.array([
            #(0.0, 1.1), (0.1, 1.0), (0.3, param), (0.55, param*0.9), (0.6, 1.2), (0.65, 1.1), (0.9, 1.9), (0.95, 1.8)#, (0.6, 1.2), (0.75, 1.6)
            (0.0, 1.1), (0.15, 1.0), (0.4, param), (0.7, param*0.90)
        ])
        # TODO think about the value of QUANTIZATION. Can we do better then hard-coding?
        temp = Interp(points[:, 0]/rotations, points[:, 1], 1/rotations)
        smoothing = 0.15 / rotations
        QUANTIZATION = 1000
        def fun(t):
            samples_x = np.linspace(t-smoothing/2, t+smoothing/2, QUANTIZATION)
            samples_y = temp(samples_x)
            val = np.sum(samples_y) / len(samples_y)
            return val
        fun_v = np.vectorize(fun)
        return Interp.from_fun(fun_v, N, 0, 1, 1)

    #r_vs_t = np.vectorize(r_vs_t)
    #r_vs_t = Interp.from_fun(r_vs_t, RING_N, 0, 1, 1)
    r_vs_t = get_r_vs_t_smooth(RING_N)

    def g_rotation_schedule(t):
        return 0
    g_rotation_schedule = np.vectorize(g_rotation_schedule, signature='()->()')
    g_rotation_schedule = Interp.from_fun(g_rotation_schedule, RING_N, 0, 1, 1)

    ring = Gear(r_vs_t, rotation_schedule=g_rotation_schedule)
    #Gear.animate([ring])
    #exit()

    ############## PLANET GEAR ######################
    def get_mi_planet(R):
        #new_g_center = np.array([R, 0])
        #new_center_schedule = np.vectorize(lambda t: new_g_center, signature='()->(2)')

        new_center_schedule = lambda t: np.array([R*np.cos(-t*TAU), R*np.sin(-t*TAU)])
        new_center_schedule = np.vectorize(new_center_schedule, signature='()->(2)')
        new_center_schedule = Interp.from_fun(new_center_schedule, RING_N, 0, 1, 1)

        return MeshingInfo(ring, new_center_schedule,
                           new_num_rotations=1, num_rotations=PLANETARY_R,
                           new_outer=False,
                           outer=True)

    global DO_VISUALIZE
    DO_VISUALIZE = False

    # binary search parameters are annoying to keep changing
    res = ring.get_meshing_gear(get_mi_planet, 0.7, 1.3)
    #res = ring.get_meshing_gear_attempt(get_mi_planet(1.0697265625))
    #res = ring.get_meshing_gear_attempt(get_mi_planet(2.0))
    #res = ring.get_meshing_gear_attempt(get_mi_planet(1.985703058540821))
    #res = ring.animate_meshing_gear_attempt(get_mi_planet(2.0))
    #planet_param = res.param_opt
    planet = res.get_gear()


    #Gear.animate([ring, planet])

    ############# SUN GEAR #################

    def get_mi_sun(R):
        #new_g_center = np.array([R, 0])
        #new_center_schedule = np.vectorize(lambda t: new_g_center, signature='()->(2)')

        new_center_schedule = lambda t: np.array([R*np.cos(-t*TAU), R*np.sin(-t*TAU)])
        new_center_schedule = np.vectorize(new_center_schedule, signature='()->(2)')
        new_center_schedule = Interp.from_fun(new_center_schedule, RING_N, 0, 1, 1)

        return MeshingInfo(planet, new_center_schedule,
                           new_num_rotations=PLANETARY_S, num_rotations=1,
                           new_outer=False,
                           outer=False)

    global VISUALIZE
    VISUALIZE = False
    #res_sun = planet.get_meshing_gear(get_mi_sun, -1.0, 0.3)
    res_sun = planet.get_meshing_gear_attempt(get_mi_sun(0))
    #res_sun = planet.animate_meshing_gear_attempt(get_mi_sun(0))
    #res_sun = planet.get_meshing_gear_attempt(get_mi_sun(0))
    sun = res_sun.get_gear()
    opt = res_sun.new_contact_local
    return opt, (ring, planet, sun)
    #return 0, (ring, planet, sun)

def get_planetary_attempt_wrapper(param):
    opt, _ = get_planetary_attempt(param)
    return opt

result = binary_search(get_planetary_attempt_wrapper, 1.5, 2.8, 1/PLANETARY_S, visualize=False)
#result = 1.5417449951171878
print()
print('\thard-won result is', result)
#exit()
# hard-won result is -0.7506996726989748
# the radius of the planet's orbit is 1.999993, which is probably supposed to be exactly 2. I cannot fathom why
_, (ring, planet, sun) = get_planetary_attempt(result)
Gear.animate([ring, planet, sun])

############ PLANET GEAR B #################

#print('visualizing old sun rotation')
#sun.rotation_schedule.visualize()


def temp():
    global DO_VISUALIZE
    DO_VISUALIZE = False
temp()
#DO_VISUALIZE = True
print('Doing rotation inverse')

# conceptually we map new ts to sun rotations, then sun rotations to old ts. That's how we get time warp from ts to ts
NUM_SUN_ROTATIONS = -1 * (PLANETARY_R + PLANETARY_S) / PLANETARY_S
r_start, r_end = sun.rotation_schedule.ys[0], sun.rotation_schedule.ys[-1]
if abs(r_end - r_start - NUM_SUN_ROTATIONS) > 0.1:
    print('WARNING something is wrong with NUM_SUN_ROTATIONS',
          NUM_SUN_ROTATIONS, r_start, r_end)
new_ts = (sun.rotation_schedule.ys - r_start) / NUM_SUN_ROTATIONS
old_ts = sun.rotation_schedule.xs
sun_rotation_inverse = Interp(new_ts, old_ts, 1, period_y=1)



def get_mi_planetB(R):
    #new_g_center = np.array([R, 0])
    #new_center_schedule = np.vectorize(lambda t: new_g_center, signature='()->(2)')


    return MeshingInfo(ring, new_center_schedule_v,
                       new_num_rotations=1, num_rotations=4,
                       new_outer=False,
                       outer=True)

## binary search parameters are annoying to keep changing
##res_planetB = ring.get_meshing_gear(get_mi_planetB, 0.1, 1.99)
## TODO we should be able to get rid of this attempt by reusing the old shape and warping the rotation
#res_planetB = ring.get_meshing_gear(get_mi_planetB, 1.9, 2.1)
##res_planetB = ring.get_meshing_gear_attempt(get_mi_planetB(2.0011800765991206))
##res_planetB = ring.get_meshing_gear_attempt(get_mi_planetB(2.0))
#planetB = res_planetB.get_gear()

#def new_center_schedule(t):
#    t_warp = sun_rotation_inverse(t)
#    return np.array([R * np.cos(-t_warp * TAU), R * np.sin(-t_warp * TAU)])
#
#
#new_center_schedule_v = np.vectorize(new_center_schedule, signature='()->(2)')
#new_center_schedule_v = Interp.from_fun_xs(new_center_schedule_v, sun_rotation_inverse.xs, 1, 1)
##planetB_center_schedule =

print('Cloning to create planetB')
planetB = planet.clone()
print('Time warping planetB')
planetB.time_warp(sun_rotation_inverse)


########### SUN GEAR B ##############
def get_mi_sunB(R):
    #new_g_center = np.array([R, 0])
    #new_center_schedule = np.vectorize(lambda t: new_g_center, signature='()->(2)')

    new_center_schedule = lambda t: np.array([R*np.cos(-t*TAU), R*np.sin(-t*TAU)])
    new_center_schedule_v = np.vectorize(new_center_schedule, signature='()->(2)')
    new_center_schedule_v = Interp.from_fun_xs(new_center_schedule_v, sun_rotation_inverse.xs, 1, 1)

    return MeshingInfo(planetB, new_center_schedule_v,
                       new_num_rotations=PLANETARY_S, num_rotations=1,
                       new_outer=False,
                       outer=False)

#res_sunB = planetB.get_meshing_gear(get_mi_sunB, -0.1, 0.1)
#res_sunB = planetB.get_meshing_gear_attempt(get_mi_sunB(0))
#res_sunB = planetB.animate_meshing_gear_attempt(get_mi_sunB(-1.144409179681095e-06))
#sunB = res_sunB.get_gear()

NUM_PLANETS = PLANETARY_R + PLANETARY_S

DO_VISUALIZE = False
print('Cloning to create sunB')
sunB = sun.clone()
print('Time warping sunB')
# TODO consider whether I should hard-code a normal sun rotation schedule here
sunB.time_warp(sun_rotation_inverse)
#sunB.rotation_schedule = Interp.from_fun_xs(lambda t: -NUM_PLANETS/PLANETARY_S * t, sunB.radius_vs_theta.xs, 1, 1)


print('Creating planet clones')
planet_clones = []
for i in range(PLANET_STRIDE, NUM_PLANETS, PLANET_STRIDE):
    p = planetB.clone()
    p.time_shift(i/NUM_PLANETS)
    planet_clones.append(p)
Gear.animate([ring, planetB, sunB, *planet_clones])
