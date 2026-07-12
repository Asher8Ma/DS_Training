
try:
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
print('--- Begin executing cells 77..100 ---')


# --- cell break ---

# Negate a boolean or change the sign of a float inplace
Z = np.array([True, False, True])
Z[:] = ~Z
print(Z)

Z = np.array([-1.5, 2.5, -3.5])
Z *= -1
print(Z)

# --- cell break ---

# Compute distance from point p to each line
P = np.array([0, 0])
A = np.array([[1, 1], [2, 2], [3, 3]])
B = np.array([[2, 2], [3, 3], [4, 4]])
# Distance from point to line defined by A and B
t = np.sum((P - A) * (B - A), axis=1) / np.sum((B - A) ** 2, axis=1)
d = np.sqrt(np.sum((A + t[:, np.newaxis] * (B - A) - P) ** 2, axis=1))
print(d)

# --- cell break ---

# Compute distance from each point in P to each line
P = np.array([[0, 0], [1, 1], [2, 2]])
A = np.array([[1, 1], [2, 2]])
B = np.array([[2, 2], [3, 3]])
AB = B - A
PA = P[:, None, :] - A[None, :, :]
t = np.sum(PA * AB[None, :, :], axis=2) / np.sum(AB**2, axis=1)[None, :]
proj = A[None, :, :] + t[:, :, None] * AB[None, :, :]
distances = np.linalg.norm(proj - P[:, None, :], axis=2)
print(distances)

# --- cell break ---

# Extract a subpart with fixed shape centered on a given element
Z = np.random.randint(0, 10, (10, 10))
shape = (5, 5)
pos = (5, 5)

# Use np.clip for boundary management - more NumPy-idiomatic and efficient
half_shape = (shape[0] // 2, shape[1] // 2)
start = np.array([pos[0] - half_shape[0], pos[1] - half_shape[1]])
end = np.array([pos[0] + half_shape[0] + 1, pos[1] + half_shape[1] + 1])

# Clip indices to array bounds
start = np.clip(start, 0, np.array(Z.shape))
end = np.clip(end, 0, np.array(Z.shape))

R = Z[start[0]:end[0], start[1]:end[1]]
print(R.shape)
print(R)

# --- cell break ---

# Generate an array with consecutive overlapping subarrays
from numpy.lib.stride_tricks import sliding_window_view
Z = np.arange(10)
n = 3
result = sliding_window_view(Z, n)
print(result)

# --- cell break ---

# Compute a matrix rank
Z = np.random.rand(5, 5)
rank = np.linalg.matrix_rank(Z)
print(f"Matrix rank: {rank}")

# --- cell break ---

# Find the most frequent value in an array
Z = np.array([1, 2, 1, 3, 2, 1, 1, 2, 3])
values, counts = np.unique(Z, return_counts=True)
most_frequent = values[np.argmax(counts)]
print(f"Most frequent value: {most_frequent}")

# --- cell break ---

# Extract all contiguous 3x3 blocks from a 10x10 matrix
Z = np.random.randint(0, 5, (10, 10))
n = 3
blocks = np.lib.stride_tricks.as_strided(Z, shape=(8, 8, n, n), strides=Z.strides + Z.strides)
print(f"Number of blocks: {blocks.shape[0]} x {blocks.shape[1]}")
print(f"Shape of blocks: {blocks.shape}")

# --- cell break ---

# Create a 2D array subclass where Z[i,j] == Z[j,i]
class Symmetric(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if isinstance(i, int) and isinstance(j, int):
                super().__setitem__((j, i), value)

Z = Symmetric(np.zeros((5, 5)))
Z[0, 1] = 1
print(Z)

# --- cell break ---

# Compute the sum of p matrix products at once
n, p = 5, 3
M = np.random.rand(n, n, p)
# Compute sum of M[:, :, i] @ M[:, :, i] for all i
result = np.einsum('ijk,ljk->il', M, M).sum(axis=-1)
# Alternative: simpler approach
result = np.sum([M[:, :, i] @ M[:, :, i] for i in range(p)], axis=0)
print(result.shape)

# --- cell break ---

# Get block-sum for 16x16 array with 4x4 block size
Z = np.random.randint(0, 10, (16, 16))
block_size = 4
result = Z.reshape(4, 4, 4, 4).sum(axis=(1, 3))
print(result.shape)

# --- cell break ---

# Implement Game of Life
def game_of_life_step(Z):
    # Count neighbors
    neighbors = np.zeros_like(Z)
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue
            neighbors += np.roll(np.roll(Z, i, axis=0), j, axis=1)
    # Apply rules
    birth = (Z == 0) & (neighbors == 3)
    survive = (Z == 1) & ((neighbors == 2) | (neighbors == 3))
    return np.asarray(birth | survive, dtype=int)

Z = np.random.choice([0, 1], (10, 10))
for _ in range(10):
    Z = game_of_life_step(Z)
print(Z)

# --- cell break ---

# Get the n largest values of an array
Z = np.random.randint(0, 100, 20)
n = 5
largest = np.argsort(Z)[-n:][::-1]
print(f"Indices of {n} largest: {largest}")
print(f"Values: {Z[largest]}")

# --- cell break ---

# Build the cartesian product of arbitrary number of vectors
def cartesian(*arrays):
    grids = np.meshgrid(*arrays, indexing='ij')
    return np.stack(grids, axis=-1).reshape(-1, len(arrays))

X = np.array([1, 2, 3])
Y = np.array([4, 5])
result = cartesian(X, Y)
print(result)

# --- cell break ---

# Create a record array from a regular array
Z = np.array([(1, 'first'), (2, 'second'), (3, 'third')], dtype=[('id', 'i4'), ('name', 'U10')])
print(Z)
print(Z['id'])
print(Z['name'])

# --- cell break ---

# Compute Z^3 using 3 different methods
Z = np.array([1, 2, 3, 4])

# Method 1: multiplication
result1 = Z * Z * Z

# Method 2: power function
result2 = np.power(Z, 3)

# Method 3: using einsum
result3 = np.einsum('i,i,i->', Z, Z, Z) if Z.size == 1 else Z * Z * Z

print(result1)
print(result2)
print(result3)

# --- cell break ---

# Find rows of A that contain elements of each row of B
A = np.array([[1, 2, 3], [2, 3, 4], [5, 6, 7], [1, 3, 5]])
B = np.array([[2, 3], [5, 6]])
# For each row in B, find rows in A containing all elements
for b in B:
    mask = np.all(np.isin(A, b), axis=1)
    print(f"Rows of A containing {b}: {np.where(mask)[0]}")

# --- cell break ---

# Extract rows with unequal values from 10x3 matrix
Z = np.random.randint(0, 5, (10, 3))
# Keep only rows where not all elements are equal
mask = ~np.all(Z == Z[:, :1], axis=1)
result = Z[mask]
print(result)

# --- cell break ---

# Convert vector of ints to binary matrix
Z = np.array([0, 1, 2, 3, 4, 5, 6, 7])
B = ((Z[:, np.newaxis] & (1 << np.arange(8))) > 0).astype(int)
print(B)

# --- cell break ---

# Extract unique rows from 2D array
Z = np.array([[1, 2], [3, 4], [1, 2], [5, 6], [3, 4]])
unique_rows = np.unique(Z, axis=0)
print(unique_rows)

# --- cell break ---

# Einsum equivalents of inner, outer, sum, and mul
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

# Inner product
inner = np.einsum('i,i->', A, B)
print(f"Inner: {inner}")

# Outer product
outer = np.einsum('i,j->ij', A, B)
print(f"Outer:\n{outer}")

# Sum
sum_result = np.einsum('i->', A)
print(f"Sum: {sum_result}")

# Element-wise multiply
mul = np.einsum('i,i->i', A, B)
print(f"Multiply: {mul}")

# --- cell break ---

# Sample a path using equidistant samples
def sample_equidistant(curve, n_samples):
    # Compute cumulative distance along curve
    distances = np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))
    distances = np.insert(distances, 0, 0)
    # Interpolate at equidistant points
    equidist_points = np.linspace(0, distances[-1], n_samples)
    sampled = np.array([np.interp(equidist_points, distances, curve[:, i]) for i in range(curve.shape[1])]).T
    return sampled

curve = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]])
sampled = sample_equidistant(curve, 20)
print(sampled)

# --- cell break ---

# Select rows from multinomial distribution
def is_multinomial(row, n_samples=10):
    # Check if row could be result of multinomial draw
    return np.sum(row) == n_samples and np.all(row >= 0) and np.all(row == np.round(row))

Z = np.array([[10, 0, 0], [8, 1, 1], [5, 3, 2]])
mask = np.array([is_multinomial(row) for row in Z])
result = Z[mask]
print(result)

# --- cell break ---

# Compute bootstrapped 95% confidence intervals for mean
Z = np.random.normal(100, 15, 100)
N = 1000
bootstrap_means = np.array([np.random.choice(Z, len(Z)).mean() for _ in range(N)])
ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"Sample mean: {Z.mean():.2f}")
except Exception as e:
    print('ERROR during execution:')
    traceback.print_exc()
    raise
