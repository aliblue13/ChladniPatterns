# chladni plates tests
using LinearAlgebra,Plots
using DifferentialEquations


function Laplacian(u)
    n1,n2=size(u)
    # internal nodes
    for i in 2:n-1 , j in 2:n-1
     Δu[i, j] = u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] - 4 * u[i, j]
    end

    # boundery
    