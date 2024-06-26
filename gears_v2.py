import numpy as np
import matplotlib.pyplot as plt

from assembly_3d import Assembly3D
from gear import Gear
from gear_3d import Gear3D
from assembly import Assembly

TAU = np.pi*2

def test_simple():
    g1_R = (1, 2)
    g2_R = (5, 3)
    thetas = np.array([
        0.0,
        0.4,
        0.4,
        0.6,
        0.9,
    ]) * TAU / (g1_R[0]/g1_R[1])
    rs = np.array([
        1,
        4,
        3,
        3,
        1.2,
    ]) / 6.6025 * 2.5
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
    #g1 = Gear3D(g1_R, thetas, rs, is_outer=False, mirror=False)
    g2 = g1.get_partner(g2_R, partner_outer=True)
    print('finished creating gears')

    # g1.plot()
    # g2.plot()
    # plt.show()

    assembly = Assembly.mesh(g1, g2)
    #assembly = Assembly3D.mesh(g1, g2)
    assembly.animate()

    exit()


if __name__ == '__main__':

    test_simple()

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
            miter2_width = 0.05
            miter2_height = 0.1
            thetas = np.array([
                0,
                miter2_width,
                miter_width + miter2_width,
                miter_width + miter2_width,
                param2+miter_width + miter2_width,
                param2+miter_width + miter2_width,
                param2+(2*miter_width + miter2_width)*1.5,
                param2 + (2*miter_width + 2*miter2_width)*1.5
            ]) * TAU/SUN_R
            rs = np.array([
                1,
                1+miter2_height,
                1+miter_height,
                param,
                param,
                1+miter_height*1.3,
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
