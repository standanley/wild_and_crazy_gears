import numpy as np
TAU = 2*np.pi


class ToothProfile:
    # What is fun(x)?
    # as x ranges from 0 to 1, we cut two matching teeth A and B
    # fun(x) = (dA, pA), (dB, pB)
    # Move through the mesh animation until the contact point is distance d along the gear edge,
    # and make a cut at offset p.
    # p is measured in a reference frame where the gear contact point is the origin, and coord is
    # (perpendicular gear edge, tangent to gear edge)

    def __init__(self, fun):
        self.fun = fun

    def cut_teeth(self, gA, gB, N):
        # cuts N teeth around gA and gB
        # TODO if ratio is not 1 to 1
        pass


def profile1(x):
    offset = np.sin(x*TAU)
    return (x, offset), (x, offset)