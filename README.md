# Chladni Patterns

A **class mini-project** for the Numerical Methods course at Ferdowsi University of Mashhad, implementing a numerical simulation of **Chladni patterns** in a thin elastic plate using **Julia**, finite-difference methods, and a wave-based partial differential equation.

Chladni patterns are formed when a vibrating plate is excited at resonant frequencies, causing particles such as sand to accumulate along lines of approximately zero displacement. This project numerically models the plate response and visualizes the resulting patterns.

## Mathematical Model

The plate response is modeled using the **inhomogeneous Helmholtz equation with damping**:

$(\nabla^2 + (k+i\gamma)^2)\Psi(\mathbf r)=F(\mathbf r)$

where $\Psi$ is the response function, $F$ is the external forcing, $k$ is the wave number, and $\gamma$ represents damping.

The solution is constructed using the eigenmodes of the Laplacian. The relation between wave number and frequency is given by

$f(k)=Ck^2$

with

$C=\frac{1}{2\pi}\sqrt{\frac{Ed^2}{12\rho(1-\nu^2)}}$

where $E$ is Young's modulus, $\nu$ is Poisson's ratio, $\rho$ is the mass density, and $d$ is the plate thickness.

## Numerical Method

The Laplacian is discretized on a square grid using **finite differences**. The resulting sparse Helmholtz operator is combined with the external forcing and damping term, and the resulting linear system is solved numerically.

The simulation scans a range of wave numbers and identifies resonances from the response amplitude. The response at the resonant wave number is then used to visualize the corresponding Chladni pattern.

A particle simulation is also included to illustrate the formation of patterns as particles move toward nodal regions of the vibrating plate.

## Main Features

* Finite-difference discretization of the Laplacian
* Sparse matrix formulation of the Helmholtz operator
* Damped, externally forced plate response
* Numerical solution of the resulting linear system
* Wave-number scan and resonance detection
* Visualization of plate response and Chladni patterns
* Particle-based visualization of pattern formation

## Implementation

The main computational workflow is:

1. Construct the finite-difference Laplacian.
2. Build the damped Helmholtz operator.
3. Apply the external forcing.
4. Solve for the plate response.
5. Scan over wave numbers to locate resonances.
6. Visualize the response and corresponding Chladni pattern.
7. Simulate particle motion on the plate.

## Dependencies

The project is implemented in **Julia** and uses:

* `LinearAlgebra`
* `SparseArrays`
* `Plots`
* `PlutoUI`
* `Random`

The simulation is provided as a **Pluto notebook**.

## Reference

Tuan, P. H., et al. (2015). *Exploring the resonant vibration of thin plates: Reconstruction of Chladni patterns and determination of resonant wave numbers*. **The Journal of the Acoustical Society of America, 137(4), 2113–2123.**

## Author

**Ali Bolourian**

Ferdowsi University of Mashhad

Numerical Methods — Spring 2025
**Class Mini-Project**
