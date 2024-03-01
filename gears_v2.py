import numpy as np
import scipy
import matplotlib.pyplot as plt
TAU = np.pi*2

class Gear:
    def __init__(self, repetitions, thetas, rs, is_outer=False, mirror=False):
        self.repetitions = repetitions
        self.thetas = thetas
        self.rs = rs
        self.is_outer = is_outer
        self.mirror = mirror

    def get_partner(self, partner_repetitions, partner_outer=False):
        # consider [t0, r0] -> [t1, r1] on self
        # if it's linear in radial space, and centers are distance a apart, the angle on the partner is:
        # integral 0 to t1 of r(t) / (a - r(t)) dt, where r(t) = (r0 + m*t), and m=(r1-r0)/(t1-t0)
        # Indefinite
        # -t - a/m * log(a - r0 - m*t)
        # Definite, from t=0 to t=t1
        # -t1 - a/m * (log(a - r0 - m*t1) - log(a - r0))
        # -t1 - a/m * log(1 - m*t1/(a-r0))
        # annoyingly we need a separate equation for the case r0=r1. But this one I can do in my head
        # = (t1-t0)(r0/(a-r0))
        def fun_linear(dt, dr, r0, a):
            m = dr/dt
            return -dt - a / m * np.log(1 - m*dt/(a - r0))

        def fun_const(dt, r0, a):
            return dt * r0 / (a-r0)

        def fun(dts, drs, r0s, a):
            const = fun_const(dts, r0s, a)
            linear = fun_linear(dts, drs, r0s, a)
            return np.where(drs==0, const, linear)


        dts = np.diff(self.thetas, append=self.thetas[0:1]+TAU/self.repetitions)
        drs = np.diff(self.rs, append=self.rs[0:1])

        a_min = np.min(self.rs) * (1 + partner_repetitions / self.repetitions)
        a_max = np.max(self.rs) * (1 + partner_repetitions / self.repetitions)

        #test1 = fun(dts, drs, self.rs, a_min)
        #test2 = fun(dts, drs, self.rs, a_max)

        def error(xs):
            a = xs[0]
            partner_dts = fun(dts, drs, self.rs, a)
            return (sum(partner_dts) - TAU/partner_repetitions)**2
        res = scipy.optimize.minimize(error, [(a_min+a_max)/2], bounds=[(a_min, a_max)], method='Nelder-Mead')
        a_opt = res.x[0]

        partner_rs_orig = a_opt - self.rs
        partner_dthetas_orig = fun(dts, drs, self.rs, a_opt)

        if False:#not partner_outer:
            partner_rs = np.concatenate(([partner_rs_orig[0]], partner_rs_orig[1:][::-1]))
            partner_dthetas = partner_dthetas_orig[::-1]
        else:
            partner_rs = partner_rs_orig
            partner_dthetas = partner_dthetas_orig
        assert (sum(partner_dthetas) - TAU/partner_repetitions < 1e-4)

        partner_thetas = np.concatenate(([0], np.cumsum(partner_dthetas)[:-1]))

        partner = Gear(
            partner_repetitions,
            partner_thetas,
            partner_rs,
            mirror=not self.mirror
        )
        return partner

    def polar_to_rect(self, thetas, rs):
        return rs*np.cos(thetas), rs*np.sin(thetas)

    def transform_thetas(self, thetas):
        if self.mirror:
            return TAU/2 - thetas
        else:
            return thetas
    def plot(self):
        sample_thetas = np.linspace(0, TAU, 100, endpoint=False)
        # TODO this breaks with repetitions
        sample_rs = np.interp(sample_thetas,
                              np.concatenate((self.thetas, [TAU/self.repetitions])),
                              np.concatenate((self.rs, [self.rs[0]])),
                              period=TAU/self.repetitions)
        fig = plt.figure()
        SIZE = max(self.rs)*1.02
        ax = fig.add_subplot()
        ax.set_xlim([-SIZE, SIZE])
        ax.set_ylim([-SIZE, SIZE])
        ax.set_aspect('equal')

        ax.plot(*self.polar_to_rect(self.transform_thetas(sample_thetas), sample_rs), '-')
        ax.plot(*self.polar_to_rect(self.transform_thetas(self.thetas), self.rs), '*')
        ax.plot([0], [0], '+')

thetas = np.array([
    0.0,
    0.1,
    0.2,
    0.4,
]) * TAU
rs = np.array([
    1.0,
    1.1,
    1.5,
    2.6,
])

g1 = Gear(1, thetas, rs)
g2 = g1.get_partner(1)

g1.plot()
g2.plot()

plt.show()
