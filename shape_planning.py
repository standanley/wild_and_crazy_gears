import numpy as np
import scipy

radius_buckets = np.array([0.1, 0.3, 0.5, 0.7, 0.9])


D = 1
def fun_sun(r):
    return np.ones(r.shape)

def fun_planet(r):
    return r / (D-r)

def fun_ring(r):
    return r / (2*D-r)

effect_matrix = np.concatenate([[f(radius_buckets)] for f in [fun_sun, fun_planet, fun_ring]])

print(radius_buckets)
print(effect_matrix)
print()

desired_results = np.array([[1], [1], [1/4]])
ans = np.linalg.pinv(effect_matrix) @ desired_results

print(ans)

check = effect_matrix @ ans
#print(check)

nullspace = scipy.linalg.null_space(effect_matrix)

print(nullspace)