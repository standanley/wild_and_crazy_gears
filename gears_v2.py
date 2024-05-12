import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
TAU = np.pi*2


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

        def fun(x):
            y = np.interp(x, self.xs_interp, self.ys_interp, period=self.period_x)
            if self.period_y is not None:
                i = np.floor((x - self.xs_interp[0]) / self.period_x)
                y += i * self.period_y
            return y

        self.fun = fun

        #if (DO_VISUALIZE and len(self.ys.shape) == 1):
        #    self.visualize()
        #assert not nonmonotonic, 'interp is not monotonic'
        if nonmonotonic:
            raise ValueError('Nonmonotonic')

    def visualize(self):
        #xs_fake = np.linspace(-.2*self.period_x, self.period_x*2.2, 30000)
        xs_fake = np.linspace(-2.1, 15, 30000)
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

class Gear:
    def __init__(self, repetitions, thetas, rs, is_outer=False, mirror=False, ignore_checks=False):
        self.repetitions = repetitions
        self.thetas = thetas
        if not ignore_checks:
            assert thetas[0] == 0
            assert thetas[-1] < TAU/repetitions
        self.rs = rs
        self.N = len(thetas)
        self.is_outer = is_outer
        self.mirror = mirror


    @classmethod
    def fun(cls, dts, drs, r0s, a, self_is_outer=False, partner_is_outer=False):
        # consider [t0, r0] -> [t1, r1] on self
        # this function returns the corresponding dts for a partner gear
        # NEW we want to be exponential in radial space. curves are a piece of ae^(bt)
        # if we go though (0, r0), (t1, r1) then r1/r0 = e^(b*t1) -> b = log(r1/r0)/t1
        # and a=r0
        # Indefinite integral of ae^(b*t)/(d-ae^(b*t)) dt
        # -log(d - ae^(bt))/b
        # Definite integral from t=0 to t=t1
        # (-log(d-r0*e^(bt) + log(d-r0))/b
        # -1/b * log(1-r0*e^(bt)/(d-r0))
        # put in definition of b
        # -t1/log(r1/r0) * log(d-r1)
        # annoyingly we need a separate equation for the case r0=r1. But this one I can do in my head
        # = (t1-t0)(r0/(a-r0))
        # VARIATION for when self is outer
        # Indefinite integral of ae^(b*t)/(ae^(b*t)-d) dt
        # We can just pull a negative sign out
        # VARIATION for when partner is outer. NOTE d must be negative for this to make sense
        # Indefinite integral of ae^(b*t)/(-d + ae^(b*t)) dt
        # actually the same as the other variation

        flip = -1 if self_is_outer or partner_is_outer else 1
        #assert not ((a < 0) ^ partner_is_outer)
        def fun_exp(dt, dr, r0, a):
            b = np.log((r0+dr) / r0) / dt
            return -1/b * np.log((a-(r0+dr))/(a - r0))

        def fun_const(dt, r0, a):
            return dt * r0 / (a - r0)

        const = fun_const(dts, r0s, a)
        linear = fun_exp(dts, drs, r0s, a)
        zero = np.zeros(dts.shape)
        temp = np.where(drs==0, const, linear)
        result = np.where(dts==0, zero, temp)
        return result * flip

    def get_partner_dts_from_dist(self, a, partner_outer):
        # given distance between centers, get dthetas for partner
        dts = np.diff(self.thetas, append=self.thetas[0:1]+TAU/self.repetitions)
        drs = np.diff(self.rs, append=self.rs[0:1])
        partner_dts = self.fun(dts, drs, self.rs, a, self_is_outer=self.is_outer, partner_is_outer=partner_outer)
        return partner_dts

    def get_partner(self, partner_repetitions, partner_outer=False):

        assert not (self.is_outer and partner_outer)
        if self.is_outer:
            assert self.repetitions > partner_repetitions
        if partner_outer:
            assert partner_repetitions > self.repetitions


        if not partner_outer and not self.is_outer:
            a_min_0 = np.min(self.rs) * (1 + partner_repetitions / self.repetitions)
            a_min = max(np.max(self.rs), a_min_0)
            a_max = np.max(self.rs) * (1 + partner_repetitions / self.repetitions)
        elif self.is_outer:
            a_min = np.min(self.rs) * (1 - partner_repetitions / self.repetitions)
            a_max_0 = np.max(self.rs) * (1 - partner_repetitions / self.repetitions)
            a_max = min(np.min(self.rs), a_max_0)
        elif partner_outer:
            # min and max kinda get swapped because a is always negative
            a_min = np.max(self.rs) * (1 - partner_repetitions / self.repetitions)
            a_max = np.min(self.rs) * (1 - partner_repetitions / self.repetitions)
        else:
            assert False


        def error(xs):
            a = xs[0]
            partner_dts = self.get_partner_dts_from_dist(a, partner_outer)
            return (sum(partner_dts) - TAU/partner_repetitions)**2
        res = scipy.optimize.minimize(error, [(a_min+a_max)/2], bounds=[(a_min, a_max)])#, method='Nelder-Mead')
        assert res.success
        a_opt = res.x[0]
        print('Optimizer result:', a_opt)


        if False:
            xs = np.linspace(a_min, a_max, 100)
            ys = np.array([sum(self.get_partner_dts_from_dist(x, partner_outer)) for x in xs])
            plt.plot([xs[0], xs[-1]], [TAU/partner_repetitions]*2, '--')
            plt.plot(xs, ys, '*')
            plt.show()

        flip = -1 if self.is_outer or partner_outer else 1
        partner_rs = (a_opt - self.rs) * flip
        partner_dthetas = self.get_partner_dts_from_dist(a_opt, partner_outer) # self.fun(dts, drs, self.rs, a_opt, self_is_outer=self.is_outer, partner_is_outer=partner_outer)

        assert (sum(partner_dthetas) - TAU/partner_repetitions < 1e-2)

        partner_thetas = np.concatenate(([0], np.cumsum(partner_dthetas)[:-1]))

        partner = Gear(
            partner_repetitions,
            partner_thetas,
            partner_rs,
            is_outer=partner_outer,
            mirror=(not self.mirror) ^ (self.is_outer or partner_outer)
        )
        return partner

    @classmethod
    def get_planetary_from_sun(cls, sun_fun, param_range, carrier_dist_range,
                               planet_repetitions, ring_repetitions, return_param=False):

        def try_planetary(xs):
            param, carrier_dist = xs
            sun = sun_fun(param)

            planet_dthetas = sun.get_partner_dts_from_dist(carrier_dist, False)
            planet_rs = carrier_dist - sun.rs
            planet_thetas = np.concatenate(([0], np.cumsum(planet_dthetas)[:-1]))
            planet = Gear(planet_repetitions, planet_thetas, planet_rs, mirror=True, ignore_checks=True)

            ring_dthetas = planet.get_partner_dts_from_dist(-carrier_dist, True)
            return sun, planet, planet_dthetas, ring_dthetas

        def error(xs):
            sun, planet, planet_dthetas, ring_dthetas = try_planetary(xs)
            planet_error = np.sum(planet_dthetas) - TAU/planet_repetitions
            ring_error = np.sum(ring_dthetas) - TAU/ring_repetitions
            total_error = planet_error**2 + ring_error**2
            return total_error

        bounds = np.array([param_range, carrier_dist_range])
        init = np.mean(bounds, axis=1)
        res = scipy.optimize.minimize(error, init, bounds=bounds, method='Nelder-Mead')
        assert res.success
        print('optimizer result', res.x, 'bounds', bounds)
        print('error', res.fun)
        param_opt, carrier_dist_opt = res.x
        if return_param:
            return param_opt

        sun, planet, planet_dthetas, ring_dthetas = try_planetary(res.x)
        ring_thetas = np.concatenate(([0], np.cumsum(ring_dthetas)[:-1]))
        ring_rs = planet.rs + carrier_dist_opt
        ring = Gear(ring_repetitions, ring_thetas, ring_rs, is_outer=True, mirror=True)

        return sun, planet, ring


    def polar_to_rect(self, thetas, rs, center):
        return rs*np.cos(thetas) + center[0], rs*np.sin(thetas) + center[1]

    def transform_thetas(self, thetas):
        if self.mirror:
            return TAU/2 - thetas
        else:
            return thetas

    def get_plot_coords(self, center, angle):
        M = 4*5*7 * 4
        sample_thetas = np.zeros(0)
        sample_rs = np.zeros(0)
        for i in range(self.N):
            next_i = (i+1)%self.N
            t0, t1 = self.thetas[i], self.thetas[next_i]
            r0, r1 = self.rs[i], self.rs[next_i]
            if i==self.N-1:
                t1 += TAU/self.repetitions
            section_M = max(1, int(M * (t1-t0)/(TAU/self.repetitions)+1))
            section_ts = np.linspace(t0, t1, section_M, endpoint=False)

            if t0 == t1:
                section_rs = np.array([r0])
            elif r0 == r1:
                section_rs = np.full(section_M, r0)
            else:
                b = np.log(r1/r0)/(t1-t0)
                section_rs = r0 * np.exp(b*(section_ts-t0))

            sample_thetas = np.concatenate((sample_thetas, section_ts))
            sample_rs = np.concatenate((sample_rs, section_rs))

        sample_thetas = np.concatenate([sample_thetas+(TAU/self.repetitions*i) for i in range(self.repetitions)])
        sample_rs = np.concatenate([sample_rs]*self.repetitions)

        sample_thetas = np.concatenate((sample_thetas, [sample_thetas[0]]))
        sample_rs = np.concatenate((sample_rs, [sample_rs[0]]))


        xs, ys = self.polar_to_rect(self.transform_thetas(self.thetas - angle), self.rs, center)
        xs_fine, ys_fine = self.polar_to_rect(self.transform_thetas(sample_thetas - angle), sample_rs, center)
        return xs, ys, xs_fine, ys_fine

    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
            SIZE = max(self.rs)*1.02
            ax.set_xlim([-SIZE, SIZE])
            ax.set_ylim([-SIZE, SIZE])
            ax.set_aspect('equal')

        xs, ys, xs_fine, ys_fine = self.get_plot_coords([0, 0], 0)

        fine, = ax.plot(xs_fine, ys_fine, '--')
        coarse, = ax.plot(xs, ys, '*')
        point, = ax.plot([0], [0], '+')
        return [fine, coarse, point]

    def update_plot(self, center, angle, curves):
        fine, coarse, point = curves
        xs, ys, xs_fine, ys_fine = self.get_plot_coords(center, angle)
        fine.set_data(xs_fine, ys_fine)
        coarse.set_data(xs, ys)
        point.set_data([center[0]], [center[1]])
        return [fine, coarse, point]


class Assembly:
    def __init__(self, ts, gears, angles, centers):
        self.ts = ts
        self.M = len(self.ts)
        self.gears = gears
        self.num_gears = len(self.gears)
        self.angles = angles
        assert len(self.angles) == self.num_gears
        assert all(len(x)==self.M for x in self.angles)
        self.centers = centers
        assert all(len(x)==self.M for x in self.centers)

    @classmethod
    def mesh(cls, g1, g2):
        assert g1.N == g2.N
        M = 4*5*7 * 4
        ts = np.linspace(0, 1, M, endpoint=False)
        angles1 = ts * TAU / g1.repetitions

        flip = -1 if (g1.is_outer or g2.is_outer) else 1
        distance = np.mean(g1.rs + g2.rs*flip)
        assert all(abs(g1.rs + g2.rs*flip - distance) < 1e-4)

        # g1 goes from r1 to r2 as g2 goes from d-r1 to d-r2
        # The goal is to create angles2, the angle gear2 is at
        # for each corresponding g1 angle in angles.
        # segment counts through segments in g.thetas and g.rs
        angles2 = []
        segment = 0
        repetition_count = 0
        for angle in angles1:
            # TODO I could make better use of numpy here by doing a whole segment at a time

            def get_thetas(i):
                # start and end thetas for g1 this segment. Takes into account the fact that
                # we may loop around multiple times (repetition_count > 0) so theta may be >tau
                # NOTE I later changed angles such that it won't go through more than once
                next_i = (i+1)%g1.N
                theta = g1.thetas[i]
                theta += repetition_count * TAU / g1.repetitions

                next_theta1 = TAU / g1.repetitions if next_i == 0 else g1.thetas[next_i]
                next_theta1 += repetition_count * TAU / g1.repetitions
                return theta, next_theta1

            theta, next_theta1 = get_thetas(segment)
            # this while loop usually runs 0 times, and only runs when we cross into the next
            # segment, and can occasionally run >1 times when the segment is short/zero
            # compared to the distance between angles
            while angle >= next_theta1:
                segment = (segment+1) % g1.N
                if segment == 0:
                    repetition_count += 1
                theta, next_theta1 = get_thetas(segment)

            next_segment = (segment+1) % g1.N

            dtheta1_segment = next_theta1 - theta
            dtheta1 = angle - theta

            # note that dtheta1_segment will never be 0 because the above while loop
            # effectively skips over those segments
            b = np.log((g1.rs[next_segment]) / g1.rs[segment]) / dtheta1_segment
            r1 = g1.rs[segment] * np.exp(b*dtheta1)
            dr1 = r1 - g1.rs[segment]

            # note that we are calculating here the additional theta turned by g2
            # from the start of the segment to here, NOT from the last angle2 to here
            dtheta2 = Gear.fun(dtheta1,
                               dr1,
                               g1.rs[segment],
                               distance,
                               self_is_outer=g1.is_outer,
                               partner_is_outer=g2.is_outer)
            angles2.append(g2.thetas[segment] + dtheta2)
        angles2 = np.array(angles2)

        # TODO I'm not sure this next line is right
        if g1.mirror:
            angles1 += TAU/2

        # mirror, not outer -> no
        # no mirror, is outer -> no
        if (not g2.mirror) ^ (g1.is_outer or g2.is_outer):
            angles2 += TAU/2

        assert len(angles2) == M

        center1, center2 = [-distance/2, 0], [distance/2, 0]

        a = Assembly(ts,
                     [g1, g2],
                     [angles1, angles2],
                     [[center1]*M, [center2]*M]
                     )
        return a

    @classmethod
    def mesh_planetary(cls, sun, planet, ring):
        def repeat(xs, R, period):
            return np.concatenate([xs + i*period for i in range(R)])
        sp = cls.mesh(sun, planet)
        #sp.animate()
        pr = cls.mesh(planet, ring)
        #pr.animate()
        M = sp.M
        # TODO I think the lcm is better than the product here
        R = sun.repetitions * planet.repetitions * ring.repetitions

        # If we drive the pr from the planet gear, we already have a mapping from planet thing to gear thing.
        # We can just warp the speed of that and put the sun in the center.
        #ring_angles = np.interp(sp.angles[1], *pr.angles, period=TAU/planet.repetitions)
        temp_interp = Interp(*pr.angles, TAU/planet.repetitions, TAU/ring.repetitions)
        #temp_interp.visualize()
        ring_angles = temp_interp(sp.angles[1])
        planet_centers = np.array(sp.centers[1]) - np.array(sp.centers[0])


        #plt.figure()
        #plt.plot(*pr.angles, '--')
        #plt.plot(sp.angles[1], ring_angles)
        #plt.show()

        # with planet gear center stationary
        ts = repeat(sp.ts, R, 1)
        sun_angles1 = repeat(sp.angles[0], R, TAU/sun.repetitions)
        planet_angles1 = repeat(sp.angles[1], R, TAU/planet.repetitions)
        ring_angles1 = repeat(ring_angles, R, TAU/ring.repetitions)
        zero_centers = np.zeros((M*R, 2))
        planet_centers1 = np.concatenate([planet_centers]*R)
        spr = Assembly(
            ts,
            [sun, planet, ring],
            [sun_angles1, planet_angles1, ring_angles1],
            [zero_centers, planet_centers1, zero_centers]
        )
        #spr.animate()

        # with ring gear stationary
        planet_dist = planet_centers[0][0]
        sun_angles2 = sun_angles1 + ring_angles1
        planet_centers2 = planet_dist * np.stack([np.cos(-ring_angles1), np.sin(-ring_angles1)], axis=1)
        planet_angles2 = planet_angles1 - ring_angles1
        ring_angles2 = np.zeros(M*R)
        spr2 = Assembly(
            ts,
            [sun, planet, ring],
            [sun_angles2, planet_angles2, ring_angles2],
            [zero_centers, planet_centers2, zero_centers]
        )
        #spr2.animate()


        # now warp time to make sun rotation constant speed
        ts_warped = (sun_angles2 - sun_angles2[0]) / (sun_angles2[-1] - sun_angles2[0]) * R
        ts_warped_inverse = Interp(ts_warped, spr2.ts, ts_warped[-1])(spr2.ts)
        spr2.time_warp(ts_warped_inverse)

        num_planets = sun.repetitions + ring.repetitions
        for i in range(1, num_planets):
            offset = (len(spr2.ts)*i)//(num_planets * sun.repetitions)
            print(offset)
            spr2.gears.append(planet)
            spr2.angles.append(np.roll(spr2.angles[1], offset))
            #centers_T = spr2.centers[1].T
            #thetas = np.arctan2(centers_T[1], centers_T[0])
            #rs = np.sqrt(centers_T[0]**2 + centers_T[1]**2)
            #xs = rs * np.cos(thetas + offset)
            #ys = rs * np.sin(thetas + offset)
            #np.stack([xs, ys], axis=1)
            spr2.centers.append(np.roll(spr2.centers[1], offset, axis=0))

        print('final')
        spr2.animate()



    def time_warp(self, new_ts):
        # we can't really use new_ts directly becasue the ts need to be evenly spaced for animatino
        # to work propoerly.

        for i in range(len(self.gears)):
            # TODO find proper period
            old_angles = self.angles[i]
            new_angles = Interp(self.ts, old_angles, self.ts[-1])(new_ts)
            self.angles[i] = new_angles

            xs, ys = self.centers[i].T
            new_xs = Interp(self.ts, xs, self.ts[-1])(new_ts)
            new_ys = Interp(self.ts, ys, self.ts[-1])(new_ts)
            new_centers = np.stack((new_xs, new_ys), axis=1)
            self.centers[i] = new_centers










    def animate(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        SIZE = 4
        ax.set_xlim([-SIZE, SIZE])
        ax.set_ylim([-SIZE, SIZE])
        ax.set_aspect('equal')

        curves_lists = []
        for g in self.gears:
            g_curves = g.plot(ax)
            curves_lists.append(g_curves)

        def update(frame_num):

            all_curves = []
            for g, g_curves, centers, angles in zip(self.gears, curves_lists, self.centers, self.angles):
                all_curves += g.update_plot(centers[frame_num], angles[frame_num], g_curves)
            return all_curves

        # show just the first frame
        #update(0)
        #plt.show()

        #update = self.set_up_animation(ax)
        ani = FuncAnimation(fig, partial(update), frames=range(self.M),
                            blit=True, interval=30)
        plt.show()



def test_simple():
    g1_R = 2
    g2_R = 3
    thetas = np.array([
        0.0,
        0.4,
        0.4,
        0.6,
        0.9,
    ]) * TAU / g1_R
    rs = np.array([
        1,
        4,
        3,
        3,
        1.2,
    ])
    # thetas = np.array([
    #    0.0,
    #    0.0,
    #    0.2,
    #    0.2,
    #    0.7,
    #    0.7,
    # ]) * TAU
    # rs = np.array([
    #    1.0,
    #    5.1,
    #    5.1,
    #    3.2,
    #    3.2,
    #    1.0,
    # ])

    g1 = Gear(g1_R, thetas, rs, is_outer=False, mirror=False)
    g2 = g1.get_partner(g2_R, partner_outer=True)
    print('finished creating gears')

    # g1.plot()
    # g2.plot()
    # plt.show()

    assembly = Assembly.mesh(g1, g2)
    assembly.animate()

    exit()


if __name__ == '__main__':

    #test_simple()

    SUN_R = 2
    PLANET_R = 1
    RING_R = 5


    def get_sun(param):
        thetas = np.array([
            0,
            0.1,
            0.2,
            0.4,
            0.9,
        ]) * TAU
        rs = np.array([
            1,
            1.5,
            param,
            param,
            1.5,
        ])
        #thetas = np.array([
        #    0,
        #    0.1,
        #    param,
        #    param+0.1,
        #]) * TAU
        #rs = np.array([
        #    1,
        #    4,
        #    2,
        #    1,
        #])

        sun = Gear(1, thetas, rs)
        return sun

    def get_sun_sweep(param2):
        def get_sun(param):
            miter_width = 0.06
            miter_height = 0.6
            miter2_width = 0.04
            miter2_height = 0.1
            thetas = np.array([
                0,
                miter2_width,
                miter_width + miter2_width,
                miter_width + miter2_width,
                param2+miter_width + miter2_width,
                param2+miter_width + miter2_width,
                param2+2*miter_width + miter2_width,
                param2 + 2*miter_width + 2*miter2_width
            ]) * TAU/SUN_R
            rs = np.array([
                1,
                1+miter2_height,
                1+miter_height,
                param,
                param,
                1+miter_height,
                1+miter2_height,
                1,
            ])
            sun = Gear(SUN_R, thetas, rs)
            return sun
        return get_sun


    #sun_vis = get_sun_sweep(0.04)(1.7987)
    #plt.figure()
    #plt.plot(np.concatenate((sun_vis.thetas, [TAU/SUN_R])),
    #         np.concatenate((sun_vis.rs, [sun_vis.rs[0]])),
    #         '-*')
    #plt.show()
    #exit()

    # TODO I forget what exactly is happening here, but it looks like I'm not doing a binary
    #  search for param2. I simply take the best from this sweep of 20 and go with that
    #  WAIT I remember - param2 is not to do with the correct meshing, I was just searching
    #  over many parameterized suns to find one with the lowest outer radius, I think
    param2s = np.linspace(0.01, 0.2, 20)
    params = []
    lowest = 10
    lowest_param2 = None
    for param2 in param2s:
        get_sun = get_sun_sweep(param2)
        param_opt = Gear.get_planetary_from_sun(get_sun, (1, 10), (1, 10), PLANET_R, RING_R, return_param=True)
        params.append(param_opt)
        if param_opt < lowest:
            lowest_param2 = param2
        lowest = min(param_opt, lowest)
    plt.figure()
    plt.plot(param2s, params, '+')
    plt.show()

    print('lowest param2', lowest_param2, 'lowest', lowest)

    sun, planet, ring = Gear.get_planetary_from_sun(get_sun_sweep(lowest_param2),
                                                    (1, 20), (1, 20), PLANET_R, RING_R)
    planet_dist = sun.rs[0] + planet.rs[0]

    #fig = plt.figure()
    #ax = fig.add_subplot()
    #sun.plot(ax)
    #planet_curves = planet.plot(ax)
    #planet.update_plot([planet_dist, 0], TAU/2, planet_curves)
    #ring.plot(ax)
    #plt.show()


    Assembly.mesh_planetary(sun, planet, ring)
