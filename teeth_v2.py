import numpy as np
import matplotlib.pyplot as plt

import gears_v2
from assembly import Assembly
from interp import Interp
from gear import Gear

TAU = np.pi*2


class ToothCutter:

    def __init__(self, teeth_per_repeat, pressure_angle_degrees, overlap=0.1):
        # overlap is the fraction of time (per unit distance) with 3 surfaces touching
        self.teeth_per_repeat = teeth_per_repeat
        self.pressure_angle = pressure_angle_degrees * TAU / 360
        self.overlap = overlap


    def get_gear_info(self, g):
        # we need a mappping dist -> theta
        # we will just oversample and not do anything fancy
        # start with theta -> r
        # turn that into theta -> dist
        # then just Interp the inverse of that
        # ALSO we aren't necessarily using the euclidean distance, we are just lazy and dtheta*r

        N_thetas_approx = 1024
        # thetas = np.linspace(0, TAU/g.repetitions, N_thetas, endpoint=False)
        thetas, rs = g.get_r_vs_theta(N_thetas_approx, repeat_numerator=False)
        end_theta = TAU / g.repetitions
        assert 0 <= end_theta - thetas[-1] < TAU / N_thetas_approx * 4, 'bad assumption about end_theta'
        dist = 0
        dists = []
        for i in range(len(thetas)):
            theta = thetas[i]
            next_theta = end_theta if i == len(thetas) - 1 else thetas[i + 1]
            r = rs[i]
            next_r = rs[(i + 1) % len(rs)]
            d_dist = (next_theta - theta) * (r + next_r) / 2
            dists.append(dist)
            dist += d_dist
        dists = np.array(dists)

        theta_vs_dist = Interp(dists, thetas, dist, period_y=end_theta)
        # theta_vs_dist.visualize()
        r_vs_dist = Interp(dists, rs, dist)
        # r_vs_dist.visualize()

        drs_ddists = np.diff(rs, append=rs[0]) / np.diff(dists, append=dist)
        drs_vs_dist = Interp(dists, drs_ddists, dist)

        #return theta_vs_dist, r_vs_dist, drs_vs_dist
        return thetas, rs, dists, dist, drs_ddists

    def cut(self, gear):
        thetas, rs, dists, total_dist, drs = self.get_gear_info(gear)
        r_vs_dist = Interp(dists, rs, total_dist)
        theta_vs_dist = Interp(dists, thetas, total_dist, TAU/gear.repetitions)

        tooth_face_center_dists = np.linspace(0, total_dist, self.teeth_per_repeat, endpoint=False)
        tooth_face_length = total_dist/self.teeth_per_repeat * (1+self.overlap)

        tooth_faces = []

        for dist_center in tooth_face_center_dists:
            dist_start, dist_end = dist_center + tooth_face_length/2, dist_center - tooth_face_length/2

            n_steps = 32
            dist_step_size = tooth_face_length / n_steps

            def cut_half_surface(flip):
                backwards_cut = []
                laser_x = r_vs_dist(dist_center)
                laser_y = 0

                # NOTE "theta" refers specifically to an angle measured relative to gear center, for polar form locations
                current_dist = dist_center
                for i in range(n_steps//2):
                    laser_theta = np.arctan2(laser_y, laser_x)
                    laser_r = np.sqrt(laser_x**2 + laser_y**2)
                    # gear direction/speed at the laser point
                    gear_direction = laser_theta - TAU/4
                    contact_r = r_vs_dist(current_dist)
                    # I guess speed=1 is defined to be the gear speed at the toothless contact point,
                    # and other things are proportional to that
                    gear_speed = laser_r / contact_r * flip
                    if i == 0:
                        # I believe chanign this by 180 degrees does nothing, because the speed will be chosen
                        # with the right magnitude to make it go the correct way
                        laser_direction = -TAU/4 + self.pressure_angle
                    else:
                        # laser direction is defined to be away from the toothless contact point
                        laser_direction = np.arctan2(laser_y, laser_x - contact_r)

                    # laser speed is chosen to match the component of gear velocity in the laser direction
                    laser_speed = gear_speed * np.cos(laser_direction - gear_direction) * flip

                    laser_vx = laser_speed * np.cos(laser_direction)
                    laser_vy = laser_speed * np.sin(laser_direction)

                    laser_x += laser_vx * dist_step_size * flip
                    laser_y += laser_vy * dist_step_size * flip

                    current_dist += dist_step_size * flip
                    backwards_cut.append((current_dist, laser_x, laser_y))

                return backwards_cut

            #backwards_cut = cut_half_surface(1)[::-1] + cut_half_surface(-1)
            backwards_cut = []
            backwards_cut += [(dist_center, r_vs_dist(dist_center), 0)]
            backwards_cut += cut_half_surface(1)
            backwards_cut += cut_half_surface(-1)

            #tooth_faces.append(np.array(backwards_cut))
            cut_info = np.array(backwards_cut)
            thetas_orig = theta_vs_dist(cut_info[:, 0])
            #center_theta = theta_vs_dist(dist_center)
            thetas_new = thetas_orig + np.arctan2(cut_info[:, 2], cut_info[:, 1])
            rs_new = np.sqrt(cut_info[:, 1]**2 + cut_info[:, 2]**2)

            tooth_faces.append((thetas_new, rs_new))


        all_thetas = []
        all_rs = []
        for thetas_new, rs_new in tooth_faces:
            all_thetas += [thetas_new[0]] + list(thetas_new) + [thetas_new[-1]]
            all_rs += [0.1] + list(rs_new) + [0.1]

        test_g = Gear((gear.repetitions_numerator, gear.repetitions_denominator),
                np.array(all_thetas),
                np.array(all_rs),
                is_outer=gear.is_outer,
                mirror=gear.mirror,
                ignore_checks=True)

        test_g.plot()
        plt.grid()
        plt.show()

tooth_cutter = ToothCutter(8, 20)

assembly = gears_v2.test_simple()
#assembly.animate()

g0 = assembly.gears[0]
tooth_cutter.cut(g0)


