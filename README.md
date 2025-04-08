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

## Citation

If the codes are helpful, plese cite the paper:

[1] Wenlong Tian, Ying Ye, Xujiang Chao, Lehua Qi, Efficient generation of composite RVEs with densely packed
particles: 33 lines of python code, submitted to Composites Science and Technology, 2025.
