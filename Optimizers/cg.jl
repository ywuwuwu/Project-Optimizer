import Pkg; Pkg.add("Optim")
using Optim

using Pkg
Pkg.add("ProximalOperators")
using ProximalOperators, LinearAlgebra

rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
result = optimize(rosenbrock, zeros(2), ConjugateGradient())