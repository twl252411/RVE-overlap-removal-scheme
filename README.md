# RVE-overlap-removal-scheme

## Method

(1). Input: Radius $r$ and volume fraction $v_1$ of particles, dimensions $[\boldsymbol{0}, \boldsymbol{l}]$ of RVE, maximum loop number $n_{iter}$ and tolerance $tol$ for termination criterion;

(2). Compute the required number $n$ of the particles within the RVE;

(3).  Perform a Poisson process to position the centers $\boldsymbol{c}$ of $n$ particles in the bounded spatial domain of the RVE, resulting in an initial configuration of the particles;

(4). Initialize iterative indicator $i \gets 0$;

(5). while $i < n_{iter}$:

    [1]. Find all pairs of the overlapped particles with the distance of their centers less than the given distance $2r$;
    
    [2]. Compute the overlap potential $\psi^{ij}$ of the found overlap pair $p^i$ and $p^j$ and then the total overlap potential $\Psi$ of all the particles;
    
    [3]. Calculate the derivative of $\Psi$ regarding the centers of the particles;
    
    [4]. if $\Psi>tol$:
    
        Update the centers of the particles employing a gradient descent method regarding the overlap potential $\Psi$;
        
        else:
        
        Acquire the legal configuration of the particles and exit the loop;
        
    [5]. $i \gets i + 1$\;
    
(6). Fulfil the periodic constraint of the particles at the legal configuration;
 
(7). Output: The centers of the particles.

## overlap removal scheme_numba_optimized program

`overlap removal scheme_numba_optimized.py` is a numba-accelerated overlap removal implementation intended to converge quickly for larger particle counts.

### Dependencies

- NumPy
- SciPy (`scipy.spatial.cKDTree`)
- Numba

### Key parameters

Adjust the parameters at the top of the script as needed:

- `dim`: dimension (2D or 3D).
- `rve_size`: RVE size (length in each direction).
- `inc_size`: particle diameter (scaled by `inc_enlg`).
- `inc_vf`: volume fraction.
- `inc_enlg`: particle enlargement factor (used for initialization).
- `alpha`: gradient descent step size.
- `tolerance`: convergence threshold (scaled with `inc_size`).

### Usage

```bash
python "overlap removal scheme_numba_optimized.py"
```

### Output

- The console prints the iteration number and potential (every 20 steps or on convergence).
- Results are written to `points{dim}d.txt`, containing particle centers and periodic images.

## overlap removal scheme_numba_optimized 程序

`overlap removal scheme_numba_optimized.py` 是基于 numba 加速的重叠消除实现，适合在较大粒子数下快速迭代收敛。

### 依赖

- NumPy
- SciPy（`scipy.spatial.cKDTree`）
- Numba

### 关键参数

脚本开头的参数可直接调整：

- `dim`: 维度（2D 或 3D）。
- `rve_size`: RVE 尺寸（每个方向的长度）。
- `inc_size`: 颗粒直径（会乘以 `inc_enlg` 放大）。
- `inc_vf`: 体积分数。
- `inc_enlg`: 颗粒放大因子（用于初始化）。
- `alpha`: 梯度下降步长。
- `tolerance`: 收敛阈值（与 `inc_size` 相关）。

### 运行方式

```bash
python "overlap removal scheme_numba_optimized.py"
```

### 输出

- 控制台打印迭代步数与当前势能（每 20 步或收敛时）。
- 结果保存在 `points{dim}d.txt`，包含周期边界下的粒子中心及其镜像点。

## Citation

If the codes are helpful, plese cite the paper:

[1] Wenlong Tian, Ying Ye, Xujiang Chao, Lehua Qi, Efficient generation of composite RVEs with densely packed particles, submitted to Mechanics of Materials, 2025.
