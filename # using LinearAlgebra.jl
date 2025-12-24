# using LinearAlgebra
# using SparseArrays

# function solve_fdm_pde(a, b, c, d, n, m, s)
#     h = (b - a) / (n - 1)
#     k = (d - c) / (m - 1)
#     r = (k^2) / (h^2)
    
#     N = (n - 2) * (m - 2)  # number of interior points
#     A = spzeros(N, N)
#     b_vec = zeros(N)
    
#     # Helper to map 2D grid (i, j) to 1D index
#     index(i, j) = (j - 1) * (n - 2) + i

#     for j in 1:m-2
#         for i in 1:n-2
#             idx = index(i, j)

#             # Diagonal (u(i,j))
#             A[idx, idx] = -2r - 2 + s^2 * h^2
            
#             # Left neighbor (i-1, j)
#             if i > 1
#                 A[idx, index(i-1, j)] = r
#             end
#             # Right neighbor (i+1, j)
#             if i < n - 2
#                 A[idx, index(i+1, j)] = r
#             end
#             # Bottom neighbor (i, j-1)
#             if j > 1
#                 A[idx, index(i, j-1)] = 1
#             end
#             # Top neighbor (i, j+1)
#             if j < m - 2
#                 A[idx, index(i, j+1)] = 1
#             end
#         end
#     end

#     # Solve the linear system
#     u_vec = A \ b_vec

#     # Reconstruct full solution grid with boundaries = 0
#     u = zeros(n, m)
#     for j in 1:m-2
#         for i in 1:n-2
#             u[i+1, j+1] = u_vec[index(i, j)]
#         end
#     end

#     return u
# end

# # Example usage
# a, b = 0.0, 10.0
# c, d = 0.0, 10.0
# n, m = 10, 10
# s = 3.0

# u = solve_fdm_pde(a, b, c, d, n, m, s)


using LinearAlgebra
using SparseArrays

function solve_fdm_neumann_nonhomogeneous(a, b, c, d, n, m, s;
        g_left = y -> 0.0,
        g_right = y -> 0.0,
        g_bottom = x -> 0.0,
        g_top = x -> 0.0
    )

    h = (b - a) / (n - 1)
    k = (d - c) / (m - 1)
    r = (k^2) / (h^2)
    N = n * m

    A = spzeros(N, N)
    b_vec = zeros(N)

    index(i, j) = (j - 1) * n + i
    x(i) = a + (i - 1) * h
    y(j) = c + (j - 1) * k

    for j in 1:m
        for i in 1:n
            idx = index(i, j)

            if 1 < i < n && 1 < j < m
                # Interior point
                A[idx, index(i+1, j)] = r
                A[idx, index(i-1, j)] = r
                A[idx, index(i, j+1)] = 1
                A[idx, index(i, j-1)] = 1
                A[idx, idx] = -2r - 2 + s^2 * h^2

            elseif i == 1  # Left boundary
                A[idx, index(i, j)] = 1
                A[idx, index(i+1, j)] = -1
                b_vec[idx] = -h * g_left(y(j))

            elseif i == n  # Right boundary
                A[idx, index(i, j)] = 1
                A[idx, index(i-1, j)] = -1
                b_vec[idx] = h * g_right(y(j))

            elseif j == 1  # Bottom boundary
                A[idx, index(i, j)] = 1
                A[idx, index(i, j+1)] = -1
                b_vec[idx] = -k * g_bottom(x(i))

            elseif j == m  # Top boundary
                A[idx, index(i, j)] = 1
                A[idx, index(i, j-1)] = -1
                b_vec[idx] = k * g_top(x(i))
            end
        end
    end

    u_vec = A \ b_vec
    u = reshape(u_vec, n, m)
    return u
end

# Example usage
a, b = 0.0, 1.0
c, d = 0.0, 1.0
n, m = 20, 20
s = 1.0

# Define Neumann boundary functions
g_left(y) = cos(pi * y)      # du/dx(a, y) = cos(πy)
g_right(y) = 0.0             # du/dx(b, y) = 0
g_bottom(x) = 0.0            # du/dy(x, c) = 0
g_top(x) = sin(pi * x)       # du/dy(x, d) = sin(πx)

u = solve_fdm_neumann_nonhomogeneous(
    a, b, c, d, n, m, s;
    g_left=g_left, g_right=g_right,
    g_bottom=g_bottom, g_top=g_top
)

# Print or plot result
using Plots
x = range(a, b, length=n)
y = range(c, d, length=m)
heatmap(x, y, u', title="Solution u(x,y)", xlabel="x", ylabel="y", colorbar=true)







############

############
using Plots


function solve_helmholtz_time(
        a, b, c, d,       # spatial domain
        n, m,             # grid size
        T, dt,            # total time and time step
        s, ω              # parameters s and omega
    )

    h = (b - a) / (n - 1)
    k = (d - c) / (m - 1)
    r_x = dt / h^2
    r_y = dt / k^2
    nt = Int(T/dt)
    
    # Grid
    x = range(a, b, length=n)
    y = range(c, d, length=m)
    
    # Initialize u at t = 0
    u = zeros(n, m)
    u_new = copy(u)

    for t_step in 1:nt
        t = t_step * dt
        f = sin(ω * t)

        for j in 2:m-1
            for i in 2:n-1
                u_new[i, j] = u[i, j] + r_x * (u[i+1,j] - 2u[i,j] + u[i-1,j]) +
                                        r_y * (u[i,j+1] - 2u[i,j] + u[i,j-1]) +
                                        dt * (s^2 * u[i,j] + f)
            end
        end

        # Neumann BCs (zero-gradient for simplicity)
        u_new[1, :] .= u_new[2, :]
        u_new[n, :] .= u_new[n-1, :]
        u_new[:, 1] .= u_new[:, 2]
        u_new[:, m] .= u_new[:, m-1]

        u .= u_new
    end

    return x, y, u
end

# Parameters
a, b = 0.0, 1.0
c, d = 0.0, 1.0
n, m = 50, 50
T = 3.0
dt = 0.001
s = 1.0
ω = 5π

x, y, u = solve_helmholtz_time(a, b, c, d, n, m, T, dt, s, ω)

# Plot the final result
heatmap(x, y, u', xlabel="x", ylabel="y", title="u(x,y) at t = $T")
