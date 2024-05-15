import numpy as np
import matplotlib.pyplot as plt

class Interp:
    # EPS is used in cases where period_y is not None; this class will insert another point at
    # [xs[0] + period_x - EPS, ys[0]+period_y]
    EPS = 1e-10

    def __init__(self, xs, ys, period_x, period_y=None):
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        #self.ys = ys
        self.period_x = period_x
        self.period_y = period_y

        self.N = len(xs)
        assert len(ys) == self.N

        diffs = np.diff(xs)
        self.x_increasing = True
        nonmonotonic = False
        if all(diffs >= 0):
            pass
        elif all(diffs <= 0):
            self.x_increasing = False
        else:
            nonmonotonic = True


        self.xs_interp = xs
        self.ys_interp = ys
        if self.period_y:
            sign = (1 if self.x_increasing else -1)
            x_final = self.xs_interp[0] + sign * (abs(self.period_x) - self.EPS)
            self.xs_interp = np.append(self.xs_interp, x_final)
            self.ys_interp = np.append(self.ys_interp, self.ys_interp[0] + self.period_y)

        def fun(x):
            y = np.interp(x, self.xs_interp, self.ys_interp, period=self.period_x)
            if self.period_y is not None:
                i = np.floor((x - self.xs_interp[0]) / self.period_x)
                y += i * self.period_y
            return y

        self.fun = fun

        #if (DO_VISUALIZE and len(self.ys.shape) == 1):
        #    self.visualize()
        #assert not nonmonotonic, 'interp is not monotonic'
        if nonmonotonic:
            raise ValueError('Nonmonotonic')

    def visualize(self):
        #xs_fake = np.linspace(-.2*self.period_x, self.period_x*2.2, 30000)
        xs_fake = np.linspace(-2.1, 15, 30000)
        ys_fake = self.fun(xs_fake)
        plt.plot(xs_fake, ys_fake, '*')
        plt.plot(self.xs_interp, self.ys_interp, '+')
        plt.show()

    @classmethod
    def from_fun_xs(cls, fun, xs, period_x, period_y=None):
        ys = fun(xs)
        temp = cls(xs, ys, period_x, period_y)

        # we override the normal interpolation function
        temp.fun = fun
        return temp

    @classmethod
    def from_fun(cls, fun, N, xmin, xmax, period_x, period_y=None):
        xs = np.linspace(xmin, xmax, N, endpoint=False)
        return cls.from_fun_xs(fun, xs, period_x, period_y)

    def __call__(self, x):
        return self.fun(x)
