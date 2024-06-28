import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

from interp import Interp
from gear import Gear

TAU = np.pi*2


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
    def mesh(cls, g1, g2, debug=False):
        assert g1.N == g2.N
        M = 4*5*7*2 * 4
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

        if debug:
            plt.plot(angles1)
            plt.plot(angles2)
            plt.grid()
            plt.show()

        # TODO I'm not sure this next line is right
        if g1.mirror:
            angles1 += TAU/2

        # mirror, not outer -> no
        # no mirror, is outer -> no
        if (not g2.mirror) ^ (g1.is_outer or g2.is_outer):
            angles2 += TAU/2

        assert len(angles2) == M

        center1, center2 = [-distance/2, 0], [distance/2, 0]

        a = cls(ts,
                     [g1, g2],
                     [angles1, angles2],
                     [[center1]*M, [center2]*M]
                     )
        return a

    @classmethod
    def mesh_planetary(cls, sun, planet, ring, planet_skip=1):
        debug = False
        def repeat(xs, R, period):
            return np.concatenate([xs + i*period for i in range(R)])
        sp = cls.mesh(sun, planet, debug=False)
        if debug:
            sp.animate()
        pr = cls.mesh(planet, ring, debug=False)
        if debug:
            pr.animate()
        M = sp.M
        # TODO I think the lcm is better than the product here
        R = sun.repetitions_numerator * planet.repetitions_numerator * ring.repetitions_numerator

        # If we drive the pr from the planet gear, we already have a mapping from planet thing to gear thing.
        # We can just warp the speed of that and put the sun in the center.
        #ring_angles = np.interp(sp.angles[1], *pr.angles, period=TAU/planet.repetitions)
        temp_interp = Interp(*pr.angles, TAU/planet.repetitions, TAU/ring.repetitions)
        #temp_interp.visualize()
        ring_angles = temp_interp(sp.angles[1])
        planet_centers = np.array(sp.centers[1]) - np.array(sp.centers[0])


        if debug:
            plt.figure()
            plt.plot(*pr.angles, '--')
            plt.plot(sp.angles[1], ring_angles)
            plt.show()

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
        if debug:
            spr.animate()

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
        if debug:
            spr2.animate()


        # now warp time to make sun rotation constant speed
        ts_warped = (sun_angles2 - sun_angles2[0]) / (sun_angles2[-1] - sun_angles2[0]) * R
        ts_warped_inverse = Interp(ts_warped, spr2.ts, ts_warped[-1])(spr2.ts)
        spr2.time_warp(ts_warped_inverse)

        # num_planets is I think the numerator of the sum
        #num_planets = sun.repetitions + ring.repetitions
        sr_sum_numerator = sun.repetitions_numerator*ring.repetitions_denominator + ring.repetitions_numerator*sun.repetitions_denominator
        sr_sum_denominator = sun.repetitions_denominator * ring.repetitions_denominator
        sr_sum_gcd = np.gcd(sr_sum_numerator, sr_sum_denominator)
        num_planets = sr_sum_numerator // sr_sum_gcd
        # TODO there might be a better way to get this
        wrap_amount = round((spr.angles[2][-1] - spr.angles[2][0])/TAU)
        for i in range(planet_skip, num_planets, planet_skip):
        #for i in range(1, 10, 1):
            print('planet instantiation', i, num_planets, planet_skip)
            # TODO I'm not 100% sure the next line is correct
            #offset = (len(spr2.ts)*i)//(num_planets * sun.repetitions_numerator // sun.repetitions_denominator)
            offset = (len(spr2.ts)*i)//(num_planets * wrap_amount)
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


    def get_fig_ax(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        SIZE = 3
        ax.set_xlim([-SIZE, SIZE])
        ax.set_ylim([-SIZE, SIZE])
        ax.set_aspect('equal')
        return fig, ax

    def animate(self):
        fig, ax = self.get_fig_ax()

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
