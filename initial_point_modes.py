#
import numpy as np
from scipy.stats import qmc

def generate_points(method, inc_num, rve_size, dim=2, **kwargs):
    """
    Generate points inside an RVE using different sampling methods.

    Args:
        method (str): Sampling method. Options: 'poisson', 'sobol', 'halton', 'thomas', 'bridson'.
        inc_num (int): Target number of points (for Thomas/Bridson, approximate number before clipping).
        rve_size (array-like): RVE dimensions, e.g., [Lx, Ly] or [Lx, Ly, Lz].
        dim (int, optional): Spatial dimension (2 or 3). Defaults to 2.
        **kwargs: Additional method-specific parameters.
            - Thomas: mu, sigma, lambda_p
            - Bridson/PoissonDisk: radius, seed

    Returns:
        np.ndarray: Array of shape (num_points, dim) containing the generated points.

    Raises:
        ValueError: If an unknown method is provided.
    """
    rve_size = np.array(rve_size)

    if method.lower() == 'poisson':
        # Uniform random points (Poisson point process approximation)
        points = np.random.uniform(0, 1, size=(inc_num, dim)) * rve_size

    elif method.lower() == 'sobol':
        sampler = qmc.Sobol(d=dim, scramble=True)
        # Sobol requires n to be power of 2 for full balance
        n_pow2 = 2**int(np.ceil(np.log2(inc_num)))
        points = sampler.random(n_pow2) * rve_size
        points = points[:inc_num]  # clip to desired number

    elif method.lower() == 'halton':
        sampler = qmc.Halton(d=dim, scramble=True)
        points = sampler.random(n=inc_num) * rve_size

    elif method.lower() == 'thomas':
        sigma = kwargs.get('sigma', min(rve_size) / 10)
        Np = kwargs.get('Np', max(1, int(np.sqrt(inc_num))))  # 默认父点数量
        points_per_cluster = int(np.ceil(inc_num / Np))
        # Generate fixed parent points
        parents = np.random.uniform(0, 1, size=(Np, dim)) * rve_size
        pts_list = []
        for p in parents:
            offsets = np.random.normal(0, sigma, size=(points_per_cluster, dim))
            pts_list.append(p + offsets)
        points = np.vstack(pts_list)
        # Clip to RVE
        mask = np.all((points >= 0) & (points <= rve_size), axis=1)
        points = points[mask]
        # Randomly select exactly inc_num points
        if len(points) >= inc_num:
            idx = np.random.choice(len(points), inc_num, replace=False)
            points = points[idx]
        else:
            while len(points) < inc_num:
                p = np.random.uniform(0, 1, size=(1, dim)) * rve_size
                offsets = np.random.normal(0, sigma, size=(points_per_cluster, dim))
                new_pts = p + offsets
                mask = np.all((new_pts >= 0) & (new_pts <= rve_size), axis=1)
                points = np.vstack([points, new_pts[mask]])
            points = points[:inc_num]

    elif method.lower() in ['bridson', 'poissondisk']:
        from scipy.stats.qmc import PoissonDisk
        radius = kwargs.get('radius', None)
        seed = kwargs.get('seed', None)

        if radius is None:
            # Estimate radius to approximate inc_num points
            volume_per_point = np.prod(rve_size) / inc_num
            if dim == 2:
                radius = (volume_per_point/np.pi)**0.5*0.001
            elif dim == 3:
                radius = (3/4 * volume_per_point / np.pi)**(1/3)*0.0025

        engine = PoissonDisk(d=dim, radius=radius, seed=seed)
        points = engine.fill_space()
        points = np.array(points) * rve_size

        if len(points) > inc_num:
            idx = np.random.choice(len(points), inc_num, replace=False)
            points = points[idx]
        elif len(points) < inc_num:
            print(f"Warning: could only generate {len(points)} points with radius={radius}")

    else:
        raise ValueError(f"Unknown method '{method}'")

    return points


