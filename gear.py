import numpy as np
import scipy
import matplotlib.pyplot as plt
from functools import lru_cache

TAU = np.pi*2

class Gear:
    def __init__(self, repetitions, thetas, rs, is_outer=False, mirror=False, ignore_checks=False):
        if isinstance(repetitions, tuple):
            self.repetitions_numerator = repetitions[0]
            self.repetitions_denominator = repetitions[1]
            self.repetitions = repetitions[0] / repetitions[1]
        else:
            self.repetitions_numerator = repetitions
            self.repetitions_denominator = 1
            self.repetitions = repetitions
        self.thetas = thetas
        if not ignore_checks:
            assert thetas[0] == 0
            assert thetas[-1] < TAU/self.repetitions
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
        if isinstance(partner_repetitions, tuple):
            partner_r = partner_repetitions[0] / partner_repetitions[1]
        else:
            partner_r = partner_repetitions

        assert not (self.is_outer and partner_outer)
        if self.is_outer:
            assert self.repetitions > partner_r
        if partner_outer:
            assert partner_r > self.repetitions


        if not partner_outer and not self.is_outer:
            a_min_0 = np.min(self.rs) * (1 + partner_r / self.repetitions)
            a_min = max(np.max(self.rs), a_min_0)
            a_max = np.max(self.rs) * (1 + partner_r / self.repetitions)
        elif self.is_outer:
            a_min = np.min(self.rs) * (1 - partner_r / self.repetitions)
            a_max_0 = np.max(self.rs) * (1 - partner_r / self.repetitions)
            a_max = min(np.min(self.rs), a_max_0)
        elif partner_outer:
            # min and max kinda get swapped because a is always negative
            a_min = np.max(self.rs) * (1 - partner_r / self.repetitions)
            a_max = np.min(self.rs) * (1 - partner_r / self.repetitions)
        else:
            assert False


        def error(xs):
            a = xs[0]
            partner_dts = self.get_partner_dts_from_dist(a, partner_outer)
            return (sum(partner_dts) - TAU/partner_r)**2
        res = scipy.optimize.minimize(error, [(a_min+a_max)/2], bounds=[(a_min, a_max)], method='Nelder-Mead')
        assert res.success
        a_opt = res.x[0]
        print('Optimizer result:', a_opt)

        if False:
            # optimization visualization
            xs = np.linspace(a_min, a_max, 100)
            ys = np.array([sum(self.get_partner_dts_from_dist(x, partner_outer)) for x in xs])
            plt.plot([xs[0], xs[-1]], [TAU/partner_r]*2, '--')
            plt.plot(xs, ys, '*')
            plt.plot([a_opt], [sum(self.get_partner_dts_from_dist(a_opt, partner_outer))], 'x')
            plt.show()

        flip = -1 if self.is_outer or partner_outer else 1
        partner_rs = (a_opt - self.rs) * flip
        partner_dthetas = self.get_partner_dts_from_dist(a_opt, partner_outer) # self.fun(dts, drs, self.rs, a_opt, self_is_outer=self.is_outer, partner_is_outer=partner_outer)

        assert (sum(partner_dthetas) - TAU/partner_r < 1e-2)

        partner_thetas = np.concatenate(([0], np.cumsum(partner_dthetas)[:-1]))

        partner = type(self)(
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
            planet_error = np.sum(planet_dthetas) - TAU/planet.repetitions
            ring_fake = Gear(ring_repetitions, [0], [0])
            ring_error = np.sum(ring_dthetas) - TAU/ring_fake.repetitions
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
        if any(any(abs(opt-b)<1e-3 for b in bound) for opt, bound in zip(res.x, bounds)):
            assert False, 'Check bounds'

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

    # I cache this because it gets called every frame of animation
    @lru_cache
    def get_r_vs_theta(self, M, repeat_numerator=True):
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

        if repeat_numerator:
            sample_thetas = np.concatenate([sample_thetas+(TAU/self.repetitions*i) for i in range(self.repetitions_numerator)])
            sample_rs = np.concatenate([sample_rs]*self.repetitions_numerator)

        return sample_thetas, sample_rs


    def get_plot_coords(self, center, angle):
        assert len(center) == 2
        M = 4*5*7 * 4

        sample_thetas, sample_rs = self.get_r_vs_theta(M)

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

        fine, = ax.plot(xs_fine, ys_fine, '-')
        #coarse, = ax.plot(xs, ys, '*')
        coarse, = ax.plot([], [], '*')
        point, = ax.plot([0], [0], '+')
        return [fine, coarse, point]

    def update_plot(self, center, angle, curves):
        fine, coarse, point = curves
        xs, ys, xs_fine, ys_fine = self.get_plot_coords(center, angle)
        fine.set_data(xs_fine, ys_fine)
        #coarse.set_data(xs, ys)
        point.set_data([center[0]], [center[1]])
        return [fine, coarse, point]
