import time
import numpy as np
from scipy.spatial import cKDTree
from numba import njit
from initial_point_modes import generate_points

# =========================== Parameters =============================
dim = 2  # 2 or 3
rve_size = np.ones([dim]) * 10000.0
inc_size = 20.0
inc_vf = 0.70
inc_enlg = 1.025
# approximate number of inclusions (kept your formula)
inc_num = int(np.ceil(np.prod(rve_size) * inc_vf / (np.pi * inc_size ** dim / (2. * dim))))
inc_size *= inc_enlg

alpha = 1.0
tolerance = (inc_size * 1e-3) ** 2

# Verlet / skin params
skin_factor = 0.5        # skin = skin_factor * inc_size; smaller -> more frequent rebuilds
rebuild_interval_min = 1  # 最少间隔多少迭代允许重建（避免每步都重建）
max_attempts = int(1e5)   # outer limit for iterations (kept similar)

# ==================== Numba: gradient + potential (serial, safe) ====================
@njit
def compute_gradients_and_potential(points, pairs, rve_size, inc_size):
    """
    points: (N, dim) float64
    pairs:  (M, 2) int64
    returns: gradients (N, dim), total_potential (float64)
    """
    N, dim = points.shape
    M = pairs.shape[0]
    gradients = np.zeros((N, dim), dtype=np.float64)
    total_pot = 0.0

    cutoff = inc_size
    cutoff2 = cutoff * cutoff

    for k in range(M):
        p1 = pairs[k, 0]
        p2 = pairs[k, 1]

        # compute minimum-image delta
        d2 = 0.0
        delta = np.zeros(dim, dtype=np.float64)
        for d in range(dim):
            dd = points[p1, d] - points[p2, d]
            # periodic minimum image
            box = rve_size[d]
            # bring dd into [-box/2, box/2)
            if dd > 0.5 * box:
                dd -= box
            elif dd < -0.5 * box:
                dd += box
            delta[d] = dd
            d2 += dd * dd

        if d2 <= 0.0:
            # extremely rare (identical positions); skip contribution to avoid div0
            continue

        if d2 < cutoff2:
            dist = np.sqrt(d2)
            psi = inc_size - dist
            if psi > 0.0:
                total_pot += 0.5 * psi * psi
                # gradient magnitude = - psi / dist
                coef = - psi / dist
                for d in range(dim):
                    g = coef * delta[d]
                    gradients[p1, d] += g
                    gradients[p2, d] -= g

    return gradients, total_pot

# ==================== Helper: neighbor-list build using cKDTree ====================
def build_pairs_via_kdtree(points, rve_size, inc_size):
    """
    Try to use scipy.spatial.cKDTree.query_pairs with output_type='ndarray' if available,
    otherwise convert the set to a numpy array efficiently.
    """
    tree = cKDTree(points, boxsize=rve_size)
    # try to request ndarray output (supported in newer SciPy)
    try:
        raw = tree.query_pairs(r=inc_size, output_type='ndarray')
        # raw is ndarray shape (M,2) with dtype=intp
        if raw is None or raw.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        else:
            # ensure int64 dtype
            pairs = np.asarray(raw, dtype=np.int64)
            return pairs
    except TypeError:
        # older SciPy: query_pairs returns a set of tuple pairs
        raw_set = tree.query_pairs(r=inc_size)
        if not raw_set:
            return np.empty((0, 2), dtype=np.int64)
        # convert set to ndarray: preallocate
        M = len(raw_set)
        pairs = np.empty((M, 2), dtype=np.int64)
        idx = 0
        for a, b in raw_set:
            pairs[idx, 0] = a
            pairs[idx, 1] = b
            idx += 1
        return pairs

# ==================== Main routine with Verlet-like neighbor list ====================
def run_optimization(method_name, inc_num, rve_size, dim, inc_size,
                     alpha, tolerance, skin_factor, rebuild_interval_min, max_iters=100000):
    """
    Returns: points (N, dim), timings, initial_overlap_potential
    """
    # initial points generation
    points = generate_points(method_name, inc_num, rve_size, dim).astype(np.float64)
    N = points.shape[0]
    print(f"initial points: {N}")

    # neighbor list bookkeeping
    prev_points = points.copy()
    # skin distance: neighbors are queried with inc_size + skin
    skin = skin_factor * inc_size
    neighbor_cut = inc_size + skin
    neighbor_cut2 = neighbor_cut * neighbor_cut

    # initial build
    pairs = build_pairs_via_kdtree(points, rve_size, neighbor_cut)
    # ensure pairs dtype int64 and shape (M,2)
    if pairs.shape[0] == 0:
        pairs = np.empty((0, 2), dtype=np.int64)

    # track displacement since last build
    max_disp2_since_build = 0.0
    iter_since_rebuild = 0

    # initial potential (for logging)
    grads, pot = compute_gradients_and_potential(points, pairs, rve_size, inc_size)
    init_pot = pot

    start_time = time.time()
    for it in range(max_iters):
        # compute gradients & potential using current pairs (neighbor list)
        gradients, potential = compute_gradients_and_potential(points, pairs, rve_size, inc_size)

        if it % 20 == 0 or potential < tolerance:
            print(f"Iteration {it}: Potential = {potential:.8e}, pairs={pairs.shape[0]}")

        if it == 0:
            init_pot = potential

        if potential < tolerance:
            break

        # update positions
        points = (points - alpha * gradients) % rve_size

        # update displacement since last build (max squared displacement)
        disp = points - prev_points
        # minimum-image for disp (when wrap happened)
        for d in range(dim):
            box = rve_size[d]
            # map disp to [-box/2, box/2)
            larger = disp[:, d] > 0.5 * box
            if np.any(larger):
                disp[larger, d] -= box
            smaller = disp[:, d] < -0.5 * box
            if np.any(smaller):
                disp[smaller, d] += box

        # compute squared displacements per particle, and max
        disp2 = np.sum(disp[:, :dim] * disp[:, :dim], axis=1)
        max_disp2 = np.max(disp2)
        max_disp2_since_build = max(max_disp2_since_build, max_disp2)
        iter_since_rebuild += 1

        # decide whether to rebuild neighbor list
        # rebuild when sqrt(max_disp2_since_build) >= skin/2 (common Verlet criterion)
        if iter_since_rebuild >= rebuild_interval_min and max_disp2_since_build >= (0.5 * skin) ** 2:
            # rebuild
            pairs = build_pairs_via_kdtree(points, rve_size, neighbor_cut)
            if pairs.shape[0] == 0:
                pairs = np.empty((0, 2), dtype=np.int64)
            prev_points = points.copy()
            max_disp2_since_build = 0.0
            iter_since_rebuild = 0

    elapsed = time.time() - start_time
    print(f"Finished after {it+1} iters, time = {elapsed:.3f}s, final potential = {potential:.8e}")
    return points, elapsed, init_pot

# ========================= run multiple trials (as in your original looping) =========================
if __name__ == "__main__":

    index = 0
    methods = ['poisson', 'sobol', 'halton', 'thomas', 'bridson']
    method = methods[index]

    pts, elapsed, init_pot = run_optimization(method, inc_num, rve_size, dim, inc_size,
                                             alpha, tolerance, skin_factor, rebuild_interval_min,
                                             max_iters=50000)
    # expand periodic images and save (your existing code)
    shifts = np.array(np.meshgrid(*[[-1, 0, 1]] * dim)).T.reshape(-1, dim) * rve_size
    images = np.vstack([pts + shift for shift in shifts])
    mask = ~np.any(((images - rve_size / 2.) / (rve_size / 2. + inc_size / 2.)).astype(int) != 0, axis=1)
    all_points = images[mask]

    np.savetxt(f"points{dim}d-{method}.txt", all_points)

