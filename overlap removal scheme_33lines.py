import time
import numpy as np
from scipy.spatial import cKDTree

# =========================== Parameters =============================
#
dim = 3  # 2D or 3D case
# RVE dimensions, Inclusion parameters (initial size, volume fraction, enlargement factor)
rve_size, inc_size, inc_vf, inc_enlg = np.ones([dim])*100.0, 20., 0.5, 1.05
inc_num = int(np.ceil(np.prod(rve_size) * inc_vf / (np.pi * inc_size**dim / (2.*dim))))  # number of inclusions
inc_size *= inc_enlg  # Enlarged inclusion size
alpha, tolerance = 0.4, (inc_size * 1.e-3) ** 2  # Step size for gradient descent, convergence tolerance

# =========================== Initial positions ======================
#
points = np.random.rand(inc_num, dim) * rve_size

# =========================== Main loop ==============================
#
start_time = time.time()  # Start timing
#
for iter_num in range(int(1e5)):  # Set maximum iteration limit
    #
    # Build spatial index using cKDTree with periodic boundary conditions
    tree = cKDTree(points, boxsize=rve_size)
    pairs = tree.query_pairs(r=inc_size)  # Find all pairs of points within the interaction radius

    # Compute gradients and potential
    gradients, potential = np.zeros_like(points), 0.0
    for p1, p2 in pairs:
        delta = points[p1] - points[p2]  # Vector between two points
        delta -= rve_size * np.round(delta / rve_size)  # Apply periodic boundary condition
        psi_ij = inc_size - np.linalg.norm(delta)  # Compute overlap
        if psi_ij > 0:  # Only compute for overlapping pairs
            potential += 0.5 * psi_ij**2  # Accumulate potential energy
            grad = - psi_ij * delta / np.linalg.norm(delta)  # Compute gradient
            gradients[p1] += grad
            gradients[p2] -= grad

    # Log potential and check for convergence
    print(f"Iteration {iter_num}: Potential = {potential:.6f}")
    if potential < tolerance ** 1: break  # Exit the loop

    # Update positions using gradient descent
    points -= alpha * gradients
    points %= rve_size  # Apply periodic boundary condition

# ======================== Add periodic images ========================
#
shifts = np.array(np.meshgrid(*[np.array([-1, 0, 1]) for _ in range(dim)])).T.reshape(-1, dim) * rve_size
images = np.vstack([points + ishift for ishift in shifts])
# Filter out points outside the RVE
mask = ~np.any(((images - rve_size / 2.) / (rve_size / 2. + inc_size / 2.)).astype(int) != 0, axis=1)
all_points = images[mask]

# =========================== Combine and save results =================
#
np.savetxt('points3d.txt', all_points)
print(f"Execution time: {time.time() - start_time:.6f} seconds")