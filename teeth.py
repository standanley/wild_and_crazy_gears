import numpy as np
import matplotlib.pyplot as plt

import gears_v2
from assembly import Assembly
from interp import Interp
from gear import Gear


TAU = 2*np.pi


class ToothProfile:
    # What is fun(x)?
    # as x ranges from 0 to 1, we cut two matching teeth A and B
    # fun(x) = (dA, pA), (dB, pB)
    # Move through the mesh animation until the contact point is distance d along the gear edge,
    # and make a cut at offset p.
    # p is measured in a reference frame where the gear contact point is the origin, and coord is
    # (perpendicular gear edge, tangent to gear edge)
    # When we then find p in the new space, I'm not sure whether we should use tangent to the new gear edge
    # or perpendicular to the line between centers (they are the same for circular gears)
    @classmethod
    def fun(cls, x):
        assert False, 'subclass and override this method'

    # what do we actually need from gA and gB?
    # Given a distance along the edge, we need the rotation and pre-tooth radius. We might also need the pre-tooth
    # edge tangent direction if we choose to use that for the coordinate system. I won't for now.
    @classmethod
    def cut_teeth(cls, gA, gB, N):
        # cuts N teeth around gA and gB
        # TODO if ratio is not 1 to 1

        def get_interps(g):
            # we need a mappping dist -> theta
            # we will just oversample and not do anything fancy
            # start with theta -> r
            # turn that into theta -> dist
            # then just Interp the inverse of that
            # ALSO we aren't necessarily using the euclidean distance, we are just lazy and dtheta*r

            N_thetas_approx = 1024
            #thetas = np.linspace(0, TAU/g.repetitions, N_thetas, endpoint=False)
            thetas, rs = g.get_r_vs_theta(N_thetas_approx, repeat_numerator=False)
            end_theta = TAU/g.repetitions
            assert 0 <= end_theta - thetas[-1] < TAU/N_thetas_approx * 4, 'bad assumption about end_theta'
            dist = 0
            dists = []
            for i in range(len(thetas)):
                theta = thetas[i]
                next_theta = end_theta if i == len(thetas)-1 else thetas[i+1]
                r = rs[i]
                next_r = rs[(i+1) % len(rs)]
                d_dist = (next_theta - theta) * (r + next_r)/2
                dists.append(dist)
                dist += d_dist
            dists = np.array(dists)

            theta_vs_dist = Interp(dists, thetas, dist, period_y=end_theta)
            #theta_vs_dist.visualize()
            r_vs_dist = Interp(dists, rs, dist)
            #r_vs_dist.visualize()

            return theta_vs_dist, r_vs_dist


        tooth_N = 24
        ts = np.linspace(0, N, N*tooth_N, endpoint=False)
        profile_a, profile_b = cls.fun(ts)

        results = []
        for gear, profile in [gA, profile_a], [gB, profile_b]:
            dist_fractions, offsets = profile
            theta_vs_dist, r_vs_dist = get_interps(gear)
            # when we scale dist_fractions into dists, we should scale the offsets too
            tooth_scale = theta_vs_dist.period_x / N
            dists = dist_fractions * tooth_scale
            thetas = theta_vs_dist(dists)
            rs = r_vs_dist(dists)

            flip = -1 if gear.mirror else 1
            xs = rs + offsets[0]*tooth_scale*flip
            ys = offsets[1]*tooth_scale

            new_thetas = thetas + np.arctan2(ys, xs)
            new_rs = np.sqrt(xs**2 + ys**2)

            #plt.plot(new_thetas, new_rs)
            #plt.grid()
            #plt.show()

            new_gear = Gear((gear.repetitions_numerator, gear.repetitions_denominator),
                            new_thetas,
                            new_rs,
                            is_outer=gear.is_outer,
                            mirror=gear.mirror)
            results.append(new_gear)

            #new_gear.plot()
            #plt.show()

        old_assembly = Assembly.mesh(gA, gB)
        new_assembly = Assembly(old_assembly.ts, results, old_assembly.angles, old_assembly.centers)
        new_assembly.animate()
        pass


class SineProfile(ToothProfile):
    @classmethod
    def fun(cls, x):
        offset = [np.sin(x*TAU)/TAU, 0]
        return (x, offset), (x, offset)


assembly = gears_v2.test_simple()
SineProfile.cut_teeth(*assembly.gears, 9)