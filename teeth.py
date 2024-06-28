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
    def fun_internal(cls, x):
        assert False, 'subclass and override this method'

    @classmethod
    def fun(cls, x):
        print('in new fun')
        x_int, x_frac = np.divmod(x, 1)

        # Vectorize
        #def fun_nocls(x):
        #    return cls.fun_internal(x)
        #fun_vec = np.vectorize(fun_nocls, signature='()->(2),(3)')

        res = np.array([cls.fun_internal(xf) for xf in x_frac])
        assert res.shape == (len(x), 2, 3)
        res_transpose = np.transpose(res, axes=(1,2,0))
        assert res_transpose.shape == (2, 3, len(x))

        # deal with x being outside the range 0,1
        #(da, xa, ya), (db, xb, yb) = res_transpose
        #da_shifted = da + x_int
        #db_shifted = db + x_int
        res_transpose[:,0] += x_int

        return res_transpose




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


        tooth_N = 100
        ts = np.linspace(0, N, N*tooth_N, endpoint=False)
        profile_a, profile_b = cls.fun(ts)

        results = []
        for gear, profile in [gA, profile_a], [gB, profile_b]:
            dist_fractions, offsets_x, offsets_y = profile
            theta_vs_dist, r_vs_dist = get_interps(gear)
            # when we scale dist_fractions into dists, we should scale the offsets too
            tooth_scale = theta_vs_dist.period_x / N
            dists = dist_fractions * tooth_scale
            thetas = theta_vs_dist(dists)
            rs = r_vs_dist(dists)

            flip = -1 if gear.mirror else 1
            xs = rs + offsets_x*tooth_scale*flip
            ys = offsets_y*tooth_scale

            new_thetas = thetas + np.arctan2(ys, xs)
            new_rs = np.sqrt(xs**2 + ys**2)

            if False:
                plt.plot(new_thetas, new_rs)
                plt.grid()
                plt.show()

            new_gear = Gear((gear.repetitions_numerator, gear.repetitions_denominator),
                            new_thetas,
                            new_rs,
                            is_outer=gear.is_outer,
                            mirror=gear.mirror,
                            ignore_checks=True)
            results.append(new_gear)

            #new_gear.plot()
            #plt.show()

        old_assembly = Assembly.mesh(gA, gB)
        new_assembly = Assembly(old_assembly.ts, results, old_assembly.angles, old_assembly.centers)
        new_assembly.animate()
        pass


class SineProfile(ToothProfile):
    @classmethod
    def fun_internal(cls, x):
        offset_x = np.sin(x*TAU)/TAU
        offset_y = 0
        return (x, offset_x, offset_y), (x, offset_x, offset_y)

class InvoluteProfile(ToothProfile):
    pressure_angle_degrees = 30

    @classmethod
    def fun_internal(cls, x):

        pressure_angle = cls.pressure_angle_degrees * TAU / 360
        # as x progresses 0 to 1, dists will also progress (net) 0 to 1.
        # A: x=(0,.25), d=(0,

        # Imagine the line of action as a diagonal from (-loa_x, loa_y) to (loa_x, -loa_y)
        loa_y = 0.45
        loa_x = -1*np.sin(pressure_angle) * loa_y

        # (imagine dots over the letters for time derivative)
        # when they are tangent, we can see d*scale = sqrt(x^2+y^2), implies d > y
        # (WRONG) When we cross the origin, we believe cos(angle)*sqrt(x^2+y^2)=d, hmm
        # I think it's the component of d which must equal all of h
        # sqrt(x^2+y^2)=d*cos(angle)
        # scale here is because the effective d is slower because we are closer to the circle center
        # h = sqrt(x^2+y^2) -> y = cos(angle)*h
        # h = d * cos(angle) from above
        # y = cos(angle)^2*d
        # oooh interesting
        speed_d = loa_y/np.cos(pressure_angle)**2

        cap_offset = 0.5

        # we know the tangent speed of the gear should equal y speed of the dot at (0,0)
        # so the dy/dt of the dot equals dx/dt of the gear
        # as the dot moves from loa_y to -loa_y, the gear must cover 2*loa_y distance
        def interp(start, end, a):
            return start + (end - start)*a
        if x < 0.4:
            # PHASE A: (0, -loa_x, loa_y) -> (2*loa_y, loa_x, -loa_y)
            a = x/0.4
            dist = interp(0, 2*speed_d, a)
            offset_x = interp(-loa_x, loa_x, a)
            offset_y = interp(loa_y, -loa_y, a)
            cap_a = 0
            cap_b = 0
        elif x < 0.5:
            # PHASE B: (2*loa_y, loa_x, -loa_y), (0.5, loa_x, loa_y)
            a = (x-0.4)/0.1
            dist = interp(2*speed_d, 0.5, a)
            offset_x = interp(loa_x, loa_x, a)
            offset_y = interp(-loa_y, loa_y, a)
            cap_a = 1
            cap_b = 0
        elif x < 0.9:
            # PHASE C: (0.5, loa_x, loa_y) -> (0.5 + 2*loa_y, -loa_x, -loa_y)
            a = (x-0.5)/0.4
            dist = interp(0.5, 0.5+2*speed_d, a)
            offset_x = interp(loa_x, -loa_x, a)
            offset_y = interp(loa_y, -loa_y, a)
            cap_a = 0
            cap_b = 0
        elif x < 1.0:
            # PHASE D: (0.5 + 2*loa_y, -loa_x, -loa_y) -> (1, -loa_x, loa_y)
            a = (x-0.9)/0.1
            dist = interp(0.5+2*speed_d, 1, a)
            offset_x = interp(-loa_x, -loa_x, a)
            offset_y = interp(-loa_y, loa_y, a)
            cap_a = 0
            cap_b = 1
        else:
            assert False, 'bad x input?'


        return (dist, offset_x-cap_a*cap_offset, offset_y), (dist, offset_x+cap_b*cap_offset, offset_y)

assembly = gears_v2.test_simple()
#SineProfile.cut_teeth(*assembly.gears, 4)
InvoluteProfile.cut_teeth(*assembly.gears, 16)