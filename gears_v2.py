import numpy as np
#from scipy.optimize import minimize
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

from assembly_3d import Assembly3D
from gear import Gear
from gear_3d import Gear3D
from assembly import Assembly

TAU = np.pi*2

visualization = []

def test_circle():
    g1_R = (1, 1)
    g2_R = (2, 1)
    thetas = np.array([
        0.0,
        0.1
    ]) * TAU / (g1_R[0]/g1_R[1])
    rs = np.array([
        3,
        3,
    ]) / 6.6025 * 2.5

    g1 = Gear(g1_R, thetas, rs, is_outer=False, mirror=False)
    #g1 = Gear3D(g1_R, thetas, rs, is_outer=False, mirror=False)
    g2 = g1.get_partner(g2_R, partner_outer=False)
    print('finished creating gears')

    # g1.plot()
    # g2.plot()
    # plt.show()

    assembly = Assembly.mesh(g1, g2)
    #assembly.animate()
    return assembly

def test_simple():
    g1_R = (1, 1)
    g2_R = (2, 1)
    thetas = np.array([
        0.0,
        0.1
    ]) * TAU / (g1_R[0]/g1_R[1])
    rs = np.array([
        2,
        4,
    ]) / 6.6025 * 2.5
    #rs = np.array([
    #    2, 2, 2, 2, 2,
    #])
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
    g2 = g1.get_partner(g2_R, partner_outer=False)
    print('finished creating gears')

    # g1.plot()
    # g2.plot()
    # plt.show()

    assembly = Assembly.mesh(g1, g2)
    #assembly = Assembly3D.mesh(g1, g2)
    #assembly.animate()

    return assembly


def test_planetary():

    SUN_R = (3, 2)
    PLANET_R = (1, 1)
    RING_R = (11, 3)


    def get_sun(param):
        thetas = np.array([
            0,
            0.2,
            0.3,
            0.5,
            0.9,
        ]) * TAU / (SUN_R[0]/SUN_R[1])
        rs = np.array([
            1.0,
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
        #]) * TAU / (SUN_R[0]/SUN_R[1])
        #rs = np.array([
        #    1,
        #    1.6,
        #    1.1,
        #    1,
        #])

        sun = Gear(SUN_R, thetas, rs)
        return sun

    sun, planet, ring = Gear.get_planetary_from_sun(get_sun, (1, 10), (1, 10), PLANET_R, RING_R)
    #sun, planet, ring = Gear.get_planetary_from_sun(get_sun, (0.11, 0.89), (1, 10), PLANET_R, RING_R)
    Assembly.mesh_planetary(sun, planet, ring, planet_skip=5)

    exit()

def rubiks():
    # goal is to have two meshing gears where 1/3 turn on one maps to half turn on another,
    # and the remaining 2/3 maps to the other half.
    # eventually this will need to be a method that takes a param and returns the gear

    bounds_simple = [1, 2]
    def fun_simple(param):
        thetas = np.array([
            0,
            0.1,
            0.3,
            0.4,
            0.5,
        ]) * TAU
        #param = 1.7
        rs = np.array([
            1,
            param,
            param,
            1,
            1
        ])
        g1 = Gear(1, thetas, rs, is_outer=False, mirror=False)
        #g1 = Gear3D(g1_R, thetas, rs, is_outer=False, mirror=False)
        g2 = g1.get_partner(1, partner_outer=False)
        assembly = Assembly.mesh(g1, g2)
        return assembly

    bounds = [0, 0.7]
    N = 50
    def fun(param):
        thetas = np.linspace(0, TAU, N, endpoint=False)
        rs = 1 + param * np.sin(thetas)

        #x = cos(u)
        #y = param + sin(u)
        # atan(y/x) = theta, find u
        # Wolfram Alpha says it's messy analytically
        # I could move the theta points to make it easier but I need the ones at 0 and 0.5
        # actually ... those specific points are easier? Just when y is zero
        # u = asin(-param)
        # r = x = sqrt(1-param^2)
        # ok, we have our rs at our special points
        # ... and I realized I had a mistake in my earlier attempt, let me fix that instead of
        # trying this crazy thing
        g1 = Gear(1, thetas, rs, is_outer=False, mirror=False)
        g2 = g1.get_partner(1, partner_outer=False)
        assembly = Assembly.mesh(g1, g2)
        return assembly



    # I was frustrated that I couldn't get the minimizer to work; turns out I had a bug in my fun
    # So this might work but I think root_scalar is better for this 1d parameter anyway
    #def fun_minimize(xs):
    #    param = xs[0]
    #    assembly = fun(param)
    #    result = assembly.gears[1].thetas[4] / TAU
    #    print('param, result', param, result)
    #    error = (result - 2/3) **2
    #    return error
    #

    #bounds = np.array([[1, 2]])
    #opt = minimize(fun_minimize, np.average(bounds, axis=1), bounds=bounds, method='Nelder-Mead', tol=1e-7)

    def fun_root(x):
        assembly = fun(x)
        assert N%2 == 0
        result = assembly.gears[1].thetas[N//2] / TAU
        error = result - 2/3
        print('x, error', x, error)
        return error

    if False:
        xs = np.linspace(*bounds, 20)
        ys = [fun_root(x) for x in xs]
        plt.plot(xs, ys, '+')
        plt.show()

        return fun(0.7)

    opt = root_scalar(fun_root, bracket=bounds)

    best_param = opt.root
    assembly = fun(best_param)


    return assembly

if __name__ == '__main__':
    assembly = rubiks()
    assembly.animate()
    exit()
    assembly = test_simple()
    exit()
    test_planetary()
    exit()

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
