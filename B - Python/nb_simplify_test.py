import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ex77
Zb = np.array([True, False, True])
Zb[:] = ~Zb
print("ex77 bool:", Zb)

Zf = np.array([-1.5, 2.5, -3.5])
Zf *= -1
print("ex77 float:", Zf)

# ex79
P = np.array([[0,0],[1,1],[2,2]])
A = np.array([[1,1],[2,2]])
B = np.array([[2,2],[3,3]])
AB = B - A
PA = P[:, None, :] - A[None, :, :]
t = np.sum(PA * AB[None, :, :], axis=2) / np.sum(AB**2, axis=1)[None, :]
proj = A[None, :, :] + t[:, :, None] * AB[None, :, :]
distances = np.linalg.norm(proj - P[:, None, :], axis=2)
print("ex79 distances:", distances)

# ex81
Z = np.arange(10)
result = sliding_window_view(Z, 3)
print("ex81 result:\n", result)

# ex90

def cartesian(*arrays):
    grids = np.meshgrid(*arrays, indexing='ij')
    return np.stack(grids, axis=-1).reshape(-1, len(arrays))

X = np.array([1,2,3])
Y = np.array([4,5])
print("ex90 cart:", cartesian(X,Y))
