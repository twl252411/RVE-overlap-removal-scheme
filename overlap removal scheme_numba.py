import time, numpy as np
from scipy.spatial import cKDTree
from numba import njit, prange

# =========================== Parameters =============================
dim = 2  # 2D or 3D case
rve_size, inc_size, inc_vf, inc_enlg = np.ones([dim])*500.0, 20., 0.63, 1.025   # rve_size, particle diameter, volume fraction, enlargement factor
inc_num = int(np.ceil(np.prod(rve_size) * inc_vf / (np.pi * inc_size**dim / (2.*dim))))  # number of inclusions
inc_size *= inc_enlg  # Enlarged particle size

inc_vf1 = inc_num * (np.pi * inc_size**dim / (2.*dim)) / np.prod(rve_size)

alpha, tolerance = 1.0, (inc_size * 1.e-3) ** 2  # Step size for gradient descent, convergence tolerance

# =========================== Initialization =========================
points = np.random.rand(inc_num, dim) * rve_size

# =========================== Numba function =========================
@njit(parallel=True)
def compute_gradients_and_potential(points, pairs, rve_size, inc_size):

    gradients, potentials = np.zeros_like(points), np.zeros(pairs.shape[0])

    for k in prange(pairs.shape[0]):
        p1, p2 = pairs[k]
        delta = points[p1] - points[p2]
        for d in range(points.shape[1]):
            delta[d] -= rve_size[d] * round(delta[d] / rve_size[d])
        norm = np.sqrt(np.sum(delta**2))

        psi_ij = inc_size - norm
        if psi_ij > 0.0:
            potentials[k] = 0.5 * psi_ij**2
            grad = - psi_ij * delta / norm
            for d in range(points.shape[1]):
                gradients[p1, d] += grad[d]
                gradients[p2, d] -= grad[d]
                
    return gradients, np.sum(potentials)

# Trigger compilation
_ = compute_gradients_and_potential(np.zeros((1, dim)), np.empty((0, 2), dtype=np.int64), np.ones(dim), 1.0)

# =========================== Main loop ==============================
# start_time = time.time()
for i in range(1):
    for iter_num in range(int(1e5)):

        tree = cKDTree(points, boxsize=rve_size)
        raw_pairs = list(tree.query_pairs(r=inc_size))
        pairs = np.array(raw_pairs, dtype=np.int64).reshape(-1, 2) if raw_pairs else np.empty((0, 2), dtype=np.int64)
        gradients, potential = compute_gradients_and_potential(points, pairs, rve_size, inc_size)

        if iter_num % 20 == 0 or potential < tolerance: print(f"Iteration {iter_num}: Potential = {potential:.6f}")
        if potential < tolerance: break

        points = (points - alpha * gradients) % rve_size # Update positions and apply periodic boundary conditions

    # ======================== Add periodic images ========================
    shifts = np.array(np.meshgrid(*[[-1, 0, 1]] * dim)).T.reshape(-1, dim) * rve_size
    images = np.vstack([points + shift for shift in shifts])
    mask = ~np.any(((images - rve_size / 2.) / (rve_size / 2. + inc_size / 2.)).astype(int) != 0, axis=1)
    all_points = images[mask]

    # =========================== Save results ============================
    np.savetxt(f"points{dim}d-{i+1}.txt", all_points)
    # print(f"Execution time: {time.time() - start_time:.6f} seconds")