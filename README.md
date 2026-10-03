# Chladni Pattern Simulation

Class project for **Numerical Methods** (Spring 2025), Ferdowsi University of Mashhad.
**Author:** Ali Bolourian | **Professor:** Dr. Hamed Mousavian

A Julia ([Pluto.jl](https://plutojl.org/)) notebook that simulates Chladni patterns, the lines where sand collects on a vibrating plate.

## How it works

The plate is modeled with the damped, forced Helmholtz equation (based on [Tuan et al., 2015](https://doi.org/10.1121/1.4916704)):

```
(∇² + (k + iγ)²) Ψ = F
```

1. Build the Laplacian with finite differences on a N×N grid (Neumann boundaries).
2. Add the Helmholtz term and a point source at the plate's center.
3. Solve the sparse linear system for Ψ.
4. Sweep the wave number `k` to find resonances (peaks in response amplitude).
5. Plot the Chladni pattern as the zero contour of Re(Ψ).
6. Animate random particles that settle on the nodal lines.

## Run it

```julia
using Pkg; Pkg.add("Pluto")
using Pluto; Pluto.run()
```

Open the `.jl` notebook, then use the sliders for `k` and the damping `γ`. The `result/` folder has the exported HTML with all outputs, so you can view it without installing Julia.

> Two cells load screenshots from local Windows paths, so you'll need to change or remove them to run on another machine.
