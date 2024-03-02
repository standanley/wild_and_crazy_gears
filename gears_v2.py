import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
TAU = np.pi*2

class Gear:
    def __init__(self, repetitions, thetas, rs, is_outer=False, mirror=False):
        self.repetitions = repetitions
        self.thetas = thetas
        assert thetas[0] == 0
        assert thetas[-1] < TAU/repetitions
        self.rs = rs
        self.N = len(thetas)
        self.is_outer = is_outer
        self.mirror = mirror


    @classmethod
    def fun(cls, dts, drs, r0s, a):
        def fun_linear(dt, dr, r0, a):
            m = dr / dt
            return -dt - a / m * np.log(1 - m * dt / (a - r0))

        def fun_const(dt, r0, a):
            return dt * r0 / (a - r0)

        const = fun_const(dts, r0s, a)
        linear = fun_linear(dts, drs, r0s, a)
        zero = np.zeros(dts.shape)
        temp = np.where(drs==0, const, linear)
        result = np.where(dts==0, zero, temp)
        return result

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


        dts = np.diff(self.thetas, append=self.thetas[0:1]+TAU/self.repetitions)
        drs = np.diff(self.rs, append=self.rs[0:1])

        a_min = np.min(self.rs) * (1 + partner_repetitions / self.repetitions)
        a_max = np.max(self.rs) * (1 + partner_repetitions / self.repetitions)

        #test1 = fun(dts, drs, self.rs, a_min)
        #test2 = fun(dts, drs, self.rs, a_max)

        def error(xs):
            a = xs[0]
            partner_dts = self.fun(dts, drs, self.rs, a)
            return (sum(partner_dts) - TAU/partner_repetitions)**2
        res = scipy.optimize.minimize(error, [(a_min+a_max)/2], bounds=[(a_min, a_max)], method='Nelder-Mead')
        a_opt = res.x[0]

        partner_rs_orig = a_opt - self.rs
        partner_dthetas_orig = self.fun(dts, drs, self.rs, a_opt)

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

    def polar_to_rect(self, thetas, rs, center):
        return rs*np.cos(thetas) + center[0], rs*np.sin(thetas) + center[1]

    def transform_thetas(self, thetas):
        if self.mirror:
            return TAU/2 - thetas
        else:
            return thetas

    def get_plot_coords(self, center, angle):
        sample_thetas = np.linspace(0, TAU, 100, endpoint=False)
        # TODO this breaks with repetitions
        sample_rs = np.interp(sample_thetas,
                              np.concatenate((self.thetas, [TAU/self.repetitions])),
                              np.concatenate((self.rs, [self.rs[0]])),
                              period=TAU/self.repetitions)

        xs, ys = self.polar_to_rect(self.transform_thetas(self.thetas + angle), self.rs, center)
        xs_fine, ys_fine = self.polar_to_rect(self.transform_thetas(sample_thetas + angle), sample_rs, center)
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
        # TODO centers need to be able to move?
        self.ts = ts
        self.M = len(self.ts)
        self.gears = gears
        self.num_gears = len(self.gears)
        self.angles = angles
        assert len(self.angles) == self.num_gears
        assert all(len(x)==self.M for x in self.angles)
        self.centers = centers
        #assert all(len(x)==self.M for x in self.centers)

    @classmethod
    def mesh(cls, g1, g2):
        assert g1.N == g2.N
        M = 100
        ts = np.linspace(0, 1, M, endpoint=False)
        angles1 = ts * TAU

        distance = np.mean(g1.rs + g2.rs)
        assert all(abs(g1.rs + g2.rs - distance) < 1e-4)

        # g1 goes from r1 to r2 as g2 goes from d-r1 to d-r2
        angles2 = []
        segment = 0
        for angle in angles1:
            while segment < len(g1.thetas)-1 and angle > g1.thetas[segment+1]:
                segment += 1
            if segment == len(g1.thetas):
                print('in exit case')
                break

            dtheta2 = Gear.fun(angle - g1.thetas[segment],
                               g1.rs[(segment+1)%g1.N] - g1.rs[segment],
                               g1.rs[segment],
                               distance)
            angles2.append(g2.thetas[segment] + dtheta2)

        assert len(angles2) == M

        a = Assembly(ts,
                     [g1, g2],
                     [angles1, angles2],
                     [[-distance/2, 0], [distance/2, 0]]
                     )
        return a

    def animate(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_aspect('equal')

        curves_lists = []
        for g in self.gears:
            g_curves = g.plot(ax)
            curves_lists.append(g_curves)

        def update(frame_num):

            all_curves = []
            for g, g_curves, center, angles in zip(self.gears, curves_lists, self.centers, self.angles):
                all_curves += g.update_plot(center, angles[frame_num], g_curves)
            return all_curves
        #update = self.set_up_animation(ax)
        ani = FuncAnimation(fig, partial(update), frames=range(self.M),
                            blit=True, interval=100)
        plt.show()


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

#g1.plot()
#g2.plot()
#plt.show()

assembly = Assembly.mesh(g1, g2)
assembly.animate()
