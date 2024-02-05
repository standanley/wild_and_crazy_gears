import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

TAU = 2 * np.pi
DEBUG = False
VISUALIZE = False

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
        self.period_x = period_x
        self.period_y = period_y

        self.N = len(xs)
        assert len(ys) == self.N

        diffs = np.diff(xs)
        if all(diffs >= 0):
            self.x_increasing = True
        elif all(diffs <= 0):
            self.x_increasing = False
        else:
            assert False, 'interp is not monotonic'

        self.xs_interp = xs
        self.ys_interp = ys
        if self.period_y:
            sign = (1 if self.x_increasing else -1)
            x_final = self.xs_interp[0] + sign * (abs(self.period_x) - self.EPS)
            self.xs_interp = np.append(self.xs_interp, x_final)
            self.ys_interp = np.append(self.ys_interp, self.ys_interp[0] + self.period_y)

        self.fun = lambda x: np.interp(x, self.xs_interp, self.ys_interp, period=self.period_x)

    def visualize(self):
        xs_fake = np.linspace(-.2*self.period_x, self.period_x*2.2, 30000)
        ys_fake = self.fun(xs_fake)
        plt.plot(xs_fake, ys_fake, '*')
        plt.plot(self.xs, self.ys, '+')
        plt.show()

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


def binary_search(fun, minimum, maximum, target, N=20, visualize=True):
    if visualize:
        print('target is', target)
        M = 5
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
        else:
            self.rotation_schedule = rotation_schedule

        if center_schedule is None:
            self.center_schedule = np.vectorize(lambda t: np.array([0, 0]), signature='()->(2)')
        else:
            self.center_schedule = center_schedule

        #self.N = 8000
        self.N = 1000

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
        self.theta_vs_length = Interp(lengths, thetas[:-1], self.total_length, period_y=1)

    def clone(self):
        return Gear(self.radius_vs_theta, self.rotation_schedule, self.center_schedule)

    def timeshift(self, dt):
        rotation_orig = self.rotation_schedule
        center_orig = self.center_schedule
        self.rotation_schedule = lambda t: rotation_orig(t-dt)
        self.center_schedule = lambda t: center_orig(t-dt)


    def plot(self):
        xs, ys = self.get_curve_points()
        plt.figure()
        plt.plot(xs, ys)
        plt.plot([0], [0], 'x')
        plt.show()

    def get_curve_points(self, time=0):
        DRAW_N = 1000
        thetas = np.arange(0, 1, 1/DRAW_N)
        thetas = np.append(thetas, thetas[0])
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
        #spokes, = ax.plot([0, 3], [0, 3])
        #ax.plot([0], [0], 'x')
        SIZE = 4
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
                            blit=True, interval=33)
        plt.show()


    @classmethod
    def get_meshing_gear(cls, get_mi, param_min, param_max):
        mi_min, mi_max = get_mi(param_min), get_mi(param_max)
        target_flip = -1 if mi_min.outer ^ mi_min.new_outer else 1
        target = 1 / mi_min.new_num_rotations * mi_min.num_rotations * target_flip

        def fun(param):
            mi = get_mi(param)
            res = cls.get_meshing_gear_attempt(mi)
            return res.new_contact_local

        param_opt = binary_search(fun, param_min, param_max, target, visualize=VISUALIZE)
        global DEBUG
        #DEBUG = True
        res = cls.get_meshing_gear_attempt(get_mi(param_opt))
        # TODO this way of noting param_opt is a bit hacky
        res.param_opt = param_opt

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

                # when new gear is created from an old gear with multiple repetitions we do not want the new
                # gears radius to be formed from copies of each time around the old gear, so we end early
                if len(new_contacts_local) > 0 and abs(new_contact_local - new_contacts_local[0]) > 1:
                    r_finished = True

                if not r_finished:
                    new_contacts_local.append(new_contact_local)
                    new_rs.append(new_r)

            contact_local_prev = contact_local

        new_radius_vs_theta = Interp(new_contacts_local, new_rs, 1/mi.new_num_rotations)

        # TODO the period_y here should be 1/new_num_rotations, or even better, the
        #  period should be new_num_rotations but we have to duplicate the arrays accordingly
        # keep in mind that the time will end at 1/num_rotations, and we should have gotten through
        new_rotation_schedule = Interp(ts, new_rotations_global, 1, period_y=1/mi.new_num_rotations)

        res = MeshingResult(new_contact_local, new_radius_vs_theta, new_rotation_schedule, mi.new_center_schedule)
        return res



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

    def r_vs_t(t):
        N = 4
        A = param # -0.4
        B = 0.1
        t = N*(t%(1/N))
        wave = np.sin(t*TAU) + A*np.sin(2*t*TAU-0.8) + B*np.sin(3*t*TAU)
        return wave/2+3

    r_vs_t = np.vectorize(r_vs_t)

    def g_rotation_schedule(t):
        return 0
    g_rotation_schedule = np.vectorize(g_rotation_schedule, signature='()->()')

    ring = Gear(r_vs_t, rotation_schedule=g_rotation_schedule)
    #g.animate()
    #exit()

    ############## PLANET GEAR ######################
    def get_mi_planet(R):
        #new_g_center = np.array([R, 0])
        #new_center_schedule = np.vectorize(lambda t: new_g_center, signature='()->(2)')

        new_center_schedule = lambda t: np.array([R*np.cos(-t*TAU), R*np.sin(-t*TAU)])
        new_center_schedule = np.vectorize(new_center_schedule, signature='()->(2)')

        return MeshingInfo(ring, new_center_schedule,
                           new_num_rotations=1, num_rotations=4,
                           new_outer=False,
                           outer=True)

    # binary search parameters are annoying to keep changing
    res = ring.get_meshing_gear(get_mi_planet, 0.1, 2.1)
    #res = ring.get_meshing_gear_attempt(get_mi_planet(1.7409096717834474))
    planet = res.get_gear()


    #Gear.animate([ring, planet])

    ############# SUN GEAR #################

    def get_mi_sun(R):
        #new_g_center = np.array([R, 0])
        #new_center_schedule = np.vectorize(lambda t: new_g_center, signature='()->(2)')

        new_center_schedule = lambda t: np.array([R*np.cos(-t*TAU), R*np.sin(-t*TAU)])
        new_center_schedule = np.vectorize(new_center_schedule, signature='()->(2)')

        return MeshingInfo(planet, new_center_schedule,
                           new_num_rotations=1, num_rotations=4,
                           new_outer=False,
                           outer=False)

    res_sun = planet.get_meshing_gear(get_mi_sun, -1.0, 0.3)
    sun = res_sun.get_gear()
    return res_sun.param_opt, (ring, planet, sun)

def get_planetary_attempt_wrapper(param):
    opt, _ = get_planetary_attempt(param)
    return opt

#result = binary_search(get_planetary_attempt_wrapper, -0.76, -0.7, 0, visualize=False)
result = -0.7506996726989748
print('hard-won result is', result)
# hard-won result is -0.7506996726989748
# the radius of the planet's orbit is 1.999993, which is probably supposed to be exactly 2. I cannot fathom why
_, (ring, planet, sun) = get_planetary_attempt(result)
#Gear.animate([ring, planet, sun])

############ PLANET GEAR B #################


EPS = 0
new_ys = (1-sun.rotation_schedule.ys) / 5
#DEBUG=True
# I have no idea why I need this cutoff ... might be related to imperfection in the parameters?
CUTOFF = 2
sun_rotation_inverse = Interp(new_ys[:-CUTOFF], sun.rotation_schedule.xs[:-CUTOFF], 1, period_y=1)

#ts = np.linspace(-1.5, 2.5, 5000)
#rotations = sun_rotation_inverse(ts)
#plt.plot(ts, rotations, '*')
#plt.show()


def get_mi_planetB(R):
    #new_g_center = np.array([R, 0])
    #new_center_schedule = np.vectorize(lambda t: new_g_center, signature='()->(2)')

    def new_center_schedule(t):
        t_warp = sun_rotation_inverse(t)
        return np.array([R*np.cos(-t_warp*TAU), R*np.sin(-t_warp*TAU)])
    new_center_schedule_v = np.vectorize(new_center_schedule, signature='()->(2)')

    return MeshingInfo(ring, new_center_schedule_v,
                       new_num_rotations=1, num_rotations=4,
                       new_outer=False,
                       outer=True)

# binary search parameters are annoying to keep changing
#res_planetB = ring.get_meshing_gear(get_mi_planetB, 0.1, 1.99)
#res_planetB = ring.get_meshing_gear(get_mi_planetB, 1.9, 2.1)
res_planetB = ring.get_meshing_gear_attempt(get_mi_planetB(2))
planetB = res_planetB.get_gear()


########### SUN GEAR B ##############
def get_mi_sunB(R):
    #new_g_center = np.array([R, 0])
    #new_center_schedule = np.vectorize(lambda t: new_g_center, signature='()->(2)')

    new_center_schedule = lambda t: np.array([R*np.cos(-t*TAU), R*np.sin(-t*TAU)])
    new_center_schedule = np.vectorize(new_center_schedule, signature='()->(2)')

    return MeshingInfo(planetB, new_center_schedule,
                       new_num_rotations=1, num_rotations=4,
                       new_outer=False,
                       outer=False)

#res_sunB = planetB.get_meshing_gear(get_mi_sunB, -0.1, 0.1)
res_sunB = planetB.get_meshing_gear_attempt(get_mi_sunB(0))
sunB = res_sunB.get_gear()



planet_clones = []
for i in range(1, 5):
    p = planetB.clone()
    p.timeshift(i/5)
    planet_clones.append(p)
Gear.animate([ring, planetB, sunB, *planet_clones])
