import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

import gears_v2
from assembly import Assembly
from interp import Interp
from gear import Gear

TAU = np.pi*2


class ToothCutter:

    def __init__(self, teeth_per_repeat, pressure_angle_degrees, overlap=0.1, offset=0.0):
        # overlap is the fraction of time (per unit distance) with 3 surfaces touching
        self.teeth_per_repeat = teeth_per_repeat
        self.pressure_angle = pressure_angle_degrees * TAU / 360
        self.overlap = overlap
        self.offset = offset


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

        coverage_data_dist = []
        coverage_data_theta = []

        tooth_face_center_dists = np.linspace(0, total_dist, self.teeth_per_repeat, endpoint=False)
        tooth_face_center_dists += self.offset
        tooth_face_length = total_dist/self.teeth_per_repeat * (1+self.overlap)

        tooth_faces = []

        for bottom_dist_center in tooth_face_center_dists:
            #dist_start, dist_end = dist_center + tooth_face_length/2, dist_center - tooth_face_length/2
            top_dist_center = bottom_dist_center + total_dist/(self.teeth_per_repeat *2)

            step_multiplier = 2
            n_steps = 32*step_multiplier
            dist_step_size = tooth_face_length / n_steps

            def cut_half_surface(flip, flip2, dist_center):
                # flip changes the direction the cut leaves from the center point
                # flip2 changes which side of the tooth this cuts
                backwards_cut = []
                laser_x = r_vs_dist(dist_center)
                laser_y = 0

                # I think we are assuming the contact point is directly right of gear center, so we need to rotate PA
                pressure_angle_vx = np.cos(flip2*self.pressure_angle+np.pi/2)
                pressure_angle_vy = np.sin(flip2*self.pressure_angle+np.pi/2)

                # NOTE "theta" refers specifically to an angle measured relative to gear center, for polar form locations
                #laser_direction = None
                #laser_speed = None
                #laser_vx = None
                #laser_vy = None
                #max_difference = 0
                current_dist = dist_center
                debug_xs = []
                debug_ys = []
                debug_contact_xs = []
                debug_contact_ys = []
                for i in range(n_steps//2):
                    laser_theta = np.arctan2(laser_y, laser_x)
                    laser_r = np.sqrt(laser_x**2 + laser_y**2)
                    # gear direction/speed at the laser point
                    gear_direction = laser_theta - TAU/4
                    contact_r = r_vs_dist(current_dist)
                    debug_contact_xs.append(contact_r)
                    debug_contact_ys.append(0)
                    # I guess speed=1 is defined to be the gear speed at the toothless contact point,
                    # and other things are proportional to that
                    gear_speed = laser_r / contact_r * flip

                    # required movement is just gear speed in gear direction
                    # gear direction is directly away from the contact point (pitch point)
                    if i == 0:
                        required_direction = -TAU/4 + self.pressure_angle*flip2
                    else:
                        required_direction = np.arctan2(laser_y, laser_x - contact_r)
                    # the speed must match the component of the gear's speed in this direction
                    required_speed = gear_speed * np.cos(required_direction - gear_direction) * flip
                    required_dx = required_speed * dist_step_size * np.cos(required_direction)
                    required_dy = required_speed * dist_step_size * np.sin(required_direction)
                    laser_x += required_dx * flip
                    laser_y += required_dy * flip
                    debug_xs.append(laser_x)
                    debug_ys.append(laser_y)

                    # now we can move any amount in the perpendicular direction
                    perp_direction = required_direction + np.pi/2
                    perp_vx = np.cos(perp_direction)
                    perp_vy = np.sin(perp_direction)

                    # we want contact + u*pressure_angle = laser + v*perp_direction
                    # (laser - contact) = [pressure_angle, perp_direction]
                    # u, v = inv([pressure_angle, -perp_direction]) * (laser - contact)
                    measure_pa_from_original_contact = False
                    if measure_pa_from_original_contact:
                        contact_x = r_vs_dist(dist_center)
                    else:
                        contact_x = contact_r
                    contact_y = 0
                    M = np.array([[pressure_angle_vx, -perp_vx], [pressure_angle_vy, -perp_vy]])
                    xy = np.array([laser_x - contact_x, laser_y - contact_y])
                    u, v = np.linalg.inv(M) @ xy
                    #u = 0
                    #laser_x += u * pressure_angle_vx
                    #laser_y += u * pressure_angle_vy
                    #v = 0.0002 * flip2
                    laser_x += v * perp_vx
                    laser_y += v * perp_vy








                    '''
                    if i == 0:
                        # I believe chanign this by 180 degrees does nothing, because the speed will be chosen
                        # with the right magnitude to make it go the correct way
                        laser_direction = -TAU/4 + self.pressure_angle*flip2
                    else:
                        # laser direction is chosen to be away from the toothless contact point
                        prev_laser_direction = laser_direction
                        laser_direction = np.arctan2(laser_y, laser_x - contact_r)

                    # laser speed is chosen to match the component of gear velocity in the laser direction
                    prev_laser_speed = laser_speed
                    laser_speed = gear_speed * np.cos(laser_direction - gear_direction) * flip

                    difference = abs((laser_direction - (gear_direction + (0 if laser_speed > 0 else TAU/2)) + TAU/2) % TAU - TAU / 2)
                    if difference < 1e-2:
                        print('exiting early 2!', i)
                        break

                    if i != 0:
                        dir_prev = prev_laser_direction + (0 if prev_laser_speed > 0 else TAU/2)
                        dir_current = laser_direction + (0 if laser_speed > 0 else TAU/2)
                        difference = abs((dir_prev - dir_current + TAU/2)%TAU - TAU/2)
                        max_difference = max(difference, max_difference)
                        #print('checking')
                        # flag any turns more than 90 degrees
                        if difference > TAU/4:
                            print('exiting early!', i)
                            break


                    laser_vx_prev = laser_vx
                    laser_vy_prev = laser_vy
                    laser_vx = laser_speed * np.cos(laser_direction)
                    laser_vy = laser_speed * np.sin(laser_direction)
                    if i != 0 and laser_vx_prev * laser_vx < 0:
                        print('HEY!', prev_laser_direction, prev_laser_speed, dir_prev, laser_direction, laser_speed, dir_current)

                    laser_x += laser_vx * dist_step_size * flip
                    laser_y += laser_vy * dist_step_size * flip

                    # the goal is to add perp movement such that the angle between the laser and pitch point is
                    # the desired pressure angle
                    # pp + a*u = ll + b*v
                    # b is the unknown we care about, a is an unknown we don't care about
                    # I think we make a vector a,b and then it's matrix algebra?
                    # 
                    laser_speed_perp = laser_speed * 0.5
                    laser_direction_perp = laser_direction + TAU/4 * flip2
                    laser_vx_perp = laser_speed_perp * np.cos(laser_direction_perp)
                    laser_vy_perp = laser_speed_perp * np.sin(laser_direction_perp)
                    laser_x += laser_vx_perp * dist_step_size * flip
                    laser_y += laser_vy_perp * dist_step_size * flip


                    if i == 0 or i == 1:
                        print('starting move', i)
                        print(laser_x, laser_y)
                        print(laser_direction, laser_speed)
                        print(flip, flip2)

                    '''
                    current_dist += dist_step_size * flip
                    if (i+1) % step_multiplier == 0:
                        backwards_cut.append((current_dist, laser_x, laser_y))

                if False:
                    print('quick laser plot')
                    plt.plot(debug_contact_xs, debug_contact_ys, '+', label='contact')
                    plt.plot([bc[1] for bc in backwards_cut], [bc[2] for bc in backwards_cut], 'x', label='laser before pa')
                    plt.plot(debug_xs, debug_ys, 'o', label='laser')
                    plt.grid()
                    plt.gca().set_aspect('equal')
                    plt.show()

                if False:
                    # animate laser cut
                    fig = plt.figure()
                    ax = fig.add_subplot()
                    SIZE = 3
                    ax.set_xlim([-SIZE, SIZE])
                    ax.set_ylim([-SIZE, SIZE])
                    ax.set_aspect('equal')

                    curve_lists = []
                    for i in range(3):
                        g_curve = ax.plot([], [])
                        curve_lists += g_curve

                    frame_offset = 0
                    def update(frame_num):

                        all_curves = []
                        N = (frame_num + frame_offset) % len(backwards_cut)

                        laser_curve = curve_lists[0]
                        backwards_cut_partial = backwards_cut[:N]
                        xs = [txy[1] for txy in backwards_cut_partial]
                        ys = [txy[2] for txy in backwards_cut_partial]
                        laser_curve.set_data(xs, ys)

                        a_curve_x = []
                        a_curve_y = []
                        for i in range(N):
                            # this is difficult because we need to work from dist back to theta for both gears
                            print(i)

                        return all_curves

                    # show just the first frame
                    # update(0)
                    # plt.show()

                    # update = self.set_up_animation(ax)
                    ani = FuncAnimation(fig, partial(update), frames=range(len(backwards_cut)),
                                        blit=True, interval=30)
                    plt.show()

                    pass

                #print('max', max_difference)
                return backwards_cut

            #backwards_cut = cut_half_surface(1)[::-1] + cut_half_surface(-1)
            backwards_cut = []
            inverse_teeth = -1 if gear.mirror else 1

            cut_a = cut_half_surface(1, inverse_teeth, bottom_dist_center)[::-1]
            cut_b = [(bottom_dist_center, r_vs_dist(bottom_dist_center), 0)]
            cut_c = cut_half_surface(-1, inverse_teeth, bottom_dist_center)
            cut_d = cut_half_surface(1, -1*inverse_teeth, top_dist_center)[::-1]
            cut_e = [(top_dist_center, r_vs_dist(top_dist_center), 0)]
            cut_f = cut_half_surface(-1, -1*inverse_teeth, top_dist_center)

            surface_a = cut_a + cut_b + cut_c
            surface_d = cut_d + cut_e + cut_f

            surface_a = surface_a[::-1]
            surface_d = surface_d[::-1]

            def pad(surface):
                ans = []
                ans += [(surface[0][0], 0.1, 0)]
                ans += surface
                ans += [(surface[-1][0], 0.1, 0)]
                return ans

            #surface_a = pad(surface_a)
            #surface_d = pad(surface_d)

            #def append_nan(surface):
            #    nan = float('nan')
            #    surface.append((surface[-1][0], nan, nan))

            def cutout(surface0, surface1):
                theta0, x0, y0 = surface0[-1]
                theta1, x1, y1 = surface1[0]
                ans = [
                    (theta0, x0 * 0.9, y0),
                    (theta1, x1 * 0.9, y1)
                ]
                return ans


            if inverse_teeth == -1:
                cutout_temp = cutout(surface_a, surface_d)
                #cut_info = surface_a + ['root'] + surface_d
                cut_info = surface_a + surface_d
            else:
                #cut_info = surface_d + surface_a
                cutout_temp = cutout(surface_a, surface_d)
                #cut_info = surface_a + surface_d + ['root']
                cut_info = surface_a + surface_d

            tooth_faces += cut_info

        # convert from theta,x,y to theta,r
        #tooth_faces.append(np.array(backwards_cut))
        tooth_faces = np.array(tooth_faces)
        thetas_orig = theta_vs_dist(tooth_faces[:, 0])
        #center_theta = theta_vs_dist(dist_center)
        thetas_new = thetas_orig + np.arctan2(tooth_faces[:, 2], tooth_faces[:, 1])
        rs_new = np.sqrt(tooth_faces[:, 1]**2 + tooth_faces[:, 2]**2)

        # need to update this visualization with new cut_info format
        #coverage_data_dist += list(cut_info[:, 0])
        #coverage_data_theta += list(thetas_new)

        #tooth_faces.append((thetas_new, rs_new))

        ## post-process tooth_faces
        #N = len(tooth_faces)
        #tf = []
        #for i in range(N):
        #    if tooth_faces[i] == 'root':
        #        pass
        #    else:
        #        tf.append(tooth_faces[i])
        #tooth_faces = tf


        if False:
            plt.plot(coverage_data_dist, coverage_data_theta, 'x')
            plt.grid()
            plt.show()

        #all_thetas = []
        #all_rs = []
        #for thetas_new, rs_new in tooth_faces:
        #    #all_thetas += [thetas_new[0]] + list(thetas_new) + [thetas_new[-1]]
        #    #all_rs += [0.1] + list(rs_new) + [0.1]
        #    all_thetas += list(thetas_new)
        #    all_rs += list(rs_new)
        all_thetas = thetas_new
        all_rs = rs_new

        test_g = Gear((gear.repetitions_numerator, gear.repetitions_denominator),
                np.array(all_thetas),
                np.array(all_rs),
                is_outer=gear.is_outer,
                mirror=gear.mirror,
                ignore_checks=True)

        if False:
            test_g.plot()
            plt.grid()
            plt.show()
        return test_g


#tooth_cutter = ToothCutter(16, 20, overlap=0.3, offset=0)
tooth_cutter = ToothCutter(16, 35, overlap=0.1, offset=0)

#old_assembly = gears_v2.test_simple()
#old_assembly = gears_v2.test_circle()
old_assembly = gears_v2.rubiks()
#assembly.animate()

g0 = old_assembly.gears[0]
g1 = old_assembly.gears[1]
g0_teeth = tooth_cutter.cut(g0)
g1_teeth = tooth_cutter.cut(g1)

new_assembly = Assembly(old_assembly.ts, [g0_teeth, g1_teeth], old_assembly.angles, old_assembly.centers)

new_assembly.animate(frame_offset=800)
