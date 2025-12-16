import numpy as np
from numba import njit, prange
import time

@njit
def _int_div_floor(a, b):
    # safe int division floor for positive numbers
    return int(a // b)

@njit
def get_cell_index_3d(pos, cell_size, ncell):
    ix = int(pos[0] / cell_size[0])
    iy = int(pos[1] / cell_size[1])
    iz = int(pos[2] / cell_size[2])
    if ix >= ncell[0]:
        ix = ncell[0] - 1
    if iy >= ncell[1]:
        iy = ncell[1] - 1
    if iz >= ncell[2]:
        iz = ncell[2] - 1
    if ix < 0:
        ix = 0
    if iy < 0:
        iy = 0
    if iz < 0:
        iz = 0
    return ix, iy, iz

@njit
def place_particles_cell_list_generic(dim, rve_size_in, radius, inc_num, l_min, l_max, max_attempts, max_per_cell):
    """
    dim: 2 or 3
    rve_size_in: length-dim numpy array (float64)
    returns: (n_placed, points_array) where points_array shape = (n_placed, dim)
    """
    # normalize to 3D internal representation: if dim==2, set z-size = 1.0 and ignore z component
    rve_size = np.empty(3, dtype=np.float64)
    for i in range(3):
        if i < dim:
            rve_size[i] = rve_size_in[i]
        else:
            rve_size[i] = 1.0  # dummy

    # cell size per axis
    cell_size = np.empty(3, dtype=np.float64)
    for k in range(3):
        cell_size[k] = 2.0 * radius + l_min
        # avoid zero division
        if cell_size[k] <= 0.0:
            cell_size[k] = 1.0

    # number of cells per axis (at least 1)
    ncell = np.empty(3, dtype=np.int64)
    for k in range(3):
        ncell_k = int(rve_size[k] / cell_size[k])
        if ncell_k < 1:
            ncell_k = 1
        ncell[k] = ncell_k

    # allocate cell list arrays: shape (nx, ny, nz, max_per_cell)
    nx = ncell[0]; ny = ncell[1]; nz = ncell[2]
    cell_list = -np.ones((nx, ny, nz, max_per_cell), dtype=np.int64)
    cell_count = np.zeros((nx, ny, nz), dtype=np.int64)

    # prepare points storage (use 3 columns internally)
    points = np.zeros((inc_num, 3), dtype=np.float64)
    n_points = 0

    # place first particle in center region (randomized)
    center = np.empty(3, dtype=np.float64)
    init_region = np.empty(3, dtype=np.float64)
    for k in range(3):
        center[k] = rve_size[k] * 0.5
        init_region[k] = rve_size[k] * 0.1

    # random first point (if dim==2, z=0.5)
    first = np.empty(3, dtype=np.float64)
    for k in range(dim):
        first[k] = center[k] + init_region[k] * (2.0 * np.random.random() - 1.0)
    if dim == 2:
        first[2] = 0.5  # dummy

    points[0, 0] = first[0]; points[0, 1] = first[1]; points[0, 2] = first[2]
    n_points = 1

    # insert first to cell list
    ix, iy, iz = get_cell_index_3d(first, cell_size, ncell)
    cell_list[ix, iy, iz, 0] = 0
    cell_count[ix, iy, iz] = 1

    cutoff2 = (2.0 * radius + l_min) ** 2

    # main RSE loop
    for idx in range(inc_num):
        # if idx refers to not-yet-placed index, break
        if idx >= n_points:
            break

        base = np.empty(3, dtype=np.float64)
        base[0] = points[idx, 0]; base[1] = points[idx, 1]; base[2] = points[idx, 2]

        attempts = 0
        while n_points < inc_num and attempts < max_attempts:
            attempts += 1

            # sample random distance
            l = (l_min + 2.0*radius) + np.random.random() * ((l_max + 2.0*radius) - (l_min + 2.0*radius))

            # random direction (2D or 3D)
            offset = np.zeros(3, dtype=np.float64)
            if dim == 2:
                theta = 2.0 * np.pi * np.random.random()
                offset[0] = l * np.cos(theta)
                offset[1] = l * np.sin(theta)
                offset[2] = 0.0
            else:
                phi = 2.0 * np.pi * np.random.random()
                costheta = 2.0 * np.random.random() - 1.0
                sintheta = np.sqrt(1.0 - costheta * costheta)
                offset[0] = l * sintheta * np.cos(phi)
                offset[1] = l * sintheta * np.sin(phi)
                offset[2] = l * costheta

            # apply periodic BC to form new position
            newp = np.empty(3, dtype=np.float64)
            for k in range(3):
                val = base[k] + offset[k]
                # periodic wrap only meaningful for actual dims; for dummy dimension it's fine
                if val >= rve_size[k]:
                    val -= rve_size[k] * np.floor(val / rve_size[k])
                elif val < 0.0:
                    val += rve_size[k] * (1 + np.floor(-val / rve_size[k]))
                    # normalize to [0, rve_size)
                    if val >= rve_size[k]:
                        val = val % rve_size[k]
                newp[k] = val

            # compute cell index for candidate
            cx = int(newp[0] / cell_size[0])
            cy = int(newp[1] / cell_size[1])
            cz = int(newp[2] / cell_size[2])
            if cx >= nx:
                cx = nx - 1
            if cy >= ny:
                cy = ny - 1
            if cz >= nz:
                cz = nz - 1
            if cx < 0:
                cx = 0
            if cy < 0:
                cy = 0
            if cz < 0:
                cz = 0

            # neighbor search over 3^3 neighbors (works for dim==2 too since nz==1)
            too_close = False
            for di in range(-1, 2):
                ii = cx + di
                # periodic wrap of cell index
                if ii < 0:
                    ii += nx
                elif ii >= nx:
                    ii -= nx
                for dj in range(-1, 2):
                    jj = cy + dj
                    if jj < 0:
                        jj += ny
                    elif jj >= ny:
                        jj -= ny
                    for dk in range(-1, 2):
                        kk = cz + dk
                        if kk < 0:
                            kk += nz
                        elif kk >= nz:
                            kk -= nz

                        cnt = cell_count[ii, jj, kk]
                        for m in range(cnt):
                            pid = cell_list[ii, jj, kk, m]
                            # compute squared distance with minimum image (only use first `dim` components)
                            d2 = 0.0
                            for k in range(dim):
                                dd = newp[k] - points[pid, k]
                                # minimum image
                                half_box = 0.5 * rve_size[k]
                                if dd > half_box:
                                    dd -= rve_size[k]
                                elif dd < -half_box:
                                    dd += rve_size[k]
                                d2 += dd * dd
                                # early exit
                                if d2 >= cutoff2:
                                    break
                            if d2 < cutoff2:
                                too_close = True
                                break
                        if too_close:
                            break
                    if too_close:
                        break
                if too_close:
                    break

            if too_close:
                continue

            # accept new point
            points[n_points, 0] = newp[0]
            points[n_points, 1] = newp[1]
            points[n_points, 2] = newp[2]

            # add to cell list
            add_idx = int(cell_count[cx, cy, cz])
            if add_idx < max_per_cell:
                cell_list[cx, cy, cz, add_idx] = n_points
                cell_count[cx, cy, cz] = add_idx + 1
            else:
                # cell overflow: simple fallback - reject this candidate (could enlarge max_per_cell)
                # continue attempts
                # (We choose to reject rather than crash)
                # Note: if overflow occurs frequently, increase max_per_cell.
                continue

            n_points += 1

        # end attempts while
        if n_points >= inc_num:
            break

    # prepare return array trimming to dim columns
    res = np.zeros((n_points, dim), dtype=np.float64)
    for i in range(n_points):
        for k in range(dim):
            res[i, k] = points[i, k]

    return n_points, res

# =========================
# Example usage (main)
# =========================
if __name__ == "__main__":
    # parameters
    dim = 2                       # set 2 or 3
    ar = 250
    rve_size = ar * np.array([10.0, 10.0, 10.0])[:dim]
    inc_size = 20.0
    radius = inc_size * 0.5
    vf = 0.65

    # compute particle "volume"
    if dim == 2:
        pv = np.pi * radius * radius
    else:
        pv = 4.0/3.0 * np.pi * radius**3

    inc_num = int(np.ceil(np.prod(rve_size) * vf / pv))

    l_min = 0.005 * radius
    l_max = 0.01 * radius
    max_attempts = 8000
    max_per_cell = 64  # adjust if cell overflow happens

    print("target particles:", inc_num)
    nplaced, points = place_particles_cell_list_generic(dim, rve_size, radius,
                                                    inc_num, l_min, l_max,
                                                    max_attempts, max_per_cell)
    print("placed:", nplaced)

    shifts = np.array(np.meshgrid(*[[-1, 0, 1]] * dim)).T.reshape(-1, dim) * rve_size
    images = np.vstack([points + shift for shift in shifts])
    mask = ~np.any(((images - rve_size / 2.) / (rve_size / 2. + radius)).astype(int) != 0, axis=1)
    all_points = images[mask]

    np.savetxt(f"points{dim}d.txt", all_points)