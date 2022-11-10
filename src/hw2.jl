using LinearAlgebra

# IN SOME FUNCTIONS, IT MAY BE NECESSARY TO ADD KEYWORD ARGUMENTS

# In the template hw2.jl there is P in the argument list of the function
# On the other hand, in the description of the homework it is explicitly NOT stated
function generate_solutions(f, g, x_min, x_max; ε=1e-4, points=1e4)
    dimension = length(x_min)

    minimas = zeros(dimension, 0)
    for _ in 1:points
        xₚ = (x_max .- x_min) .* rand(dimension) .+ x_min

        optimal_x = optim(f, g, x -> P(x, x_min, x_max), xₚ)
        if !any(optimum -> norm(optimum .- optimal_x) < ε, eachcol(minimas))
            minimas = hcat(minimas, optimal_x)
        end
    end

    return minimas
end

f_griewank_prod(x) = prod(tuple -> ((i, xᵢ) = tuple; cos(xᵢ) / sqrt(i)), enumerate(x))

# If you need to write multiple methods based on input types, it is fine.
f_griewank(x) = 1 + 1 / 4000 * sum(x .^ 2) - f_griewank_prod(x)

# If you manage to write only one method, it is fine.
function g_griewank(x::AbstractVector)
    df = Float64[]
    for (i, xᵢ) in enumerate(x)
        index_range = filter(index -> index != i, 1:length(x))
        rest_of_terms = isempty(index_range) ? 1 : f_griewank_prod(x[index_range])

        push!(df, 1 / 4000 * 2 * xᵢ + 1 / sqrt(i) * sin(xᵢ) * rest_of_terms)
    end
    return df
end

g_griewank(x::Real) = g_griewank([x])

function optim(f, g, P, x; α=0.01, max_iter=10000)
    for _ in 1:max_iter
        y = x .- α .* g(x)
        x = P(y)
    end
    return x
end

P(x, x_min, x_max) = min.(max.(x, x_min), x_max)