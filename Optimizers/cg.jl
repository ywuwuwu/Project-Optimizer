struct ConjugateGradient{Tf, T, Tprep, IL, L} <: FirstOrderOptimizer
    eta::Tf
    P::T
    precondprep!::Tprep
    alphaguess!::IL
    linesearch!::L
    manifold::Manifold
end

Base.summary(::ConjugateGradient) = "Conjugate Gradient"

"""
# Conjugate Gradient Descent
## Constructor
```julia
ConjugateGradient(; alphaguess = LineSearches.InitialHagerZhang(),
linesearch = LineSearches.HagerZhang(),
eta = 0.4,
P = nothing,
precondprep = (P, x) -> nothing,
manifold = Flat())
```
The strictly positive constant ``eta`` is used in determining
the next step direction, and the default here deviates from the one used in the
original paper (where it was ``0.01``). See more details in the original papers
referenced below.
## Description
The `ConjugateGradient` method implements Hager and Zhang (2006) and elements
from Hager and Zhang (2013). Notice, the default `linesearch` is `HagerZhang`
from LineSearches.jl. This line search is exactly the one proposed in Hager and
Zhang (2006).
## References
 - W. W. Hager and H. Zhang (2006) Algorithm 851: CG_DESCENT, a conjugate gradient method with guaranteed descent. ACM Transactions on Mathematical Software 32: 113-137.
 - W. W. Hager and H. Zhang (2013), The Limited Memory Conjugate Gradient Method. SIAM Journal on Optimization, 23, pp. 2150-2168.
"""
function ConjugateGradient(; alphaguess = LineSearches.InitialHagerZhang(),
                             linesearch = LineSearches.HagerZhang(),
                             eta::Real = 0.4,
                             P::Any = nothing,
                             precondprep = (P, x) -> nothing,
                             manifold::Manifold=Flat())

    ConjugateGradient(eta,
                      P, precondprep,
                      _alphaguess(alphaguess), linesearch,
                      manifold)
end

mutable struct ConjugateGradientState{Tx,T,G} <: AbstractOptimizerState
    x::Tx
    x_previous::Tx
    g_previous::G
    f_x_previous::T
    y::Tx
    py::Tx
    pg::Tx
    s::Tx
    @add_linesearch_fields()
end

function reset!(cg, cgs::ConjugateGradientState, obj, x)
    cgs.x .= x
    cg.precondprep!(cg.P, x)
    ldiv!(cgs.pg, cg.P, gradient(obj))
    if cg.P !== nothing
        project_tangent!(cg.manifold, cgs.pg, x)
    end
    cgs.s .= -cgs.pg 
    cgs.f_x_previous = typeof(cgs.f_x_previous)(NaN)
end
function initial_state(method::ConjugateGradient, options, d, initial_x)
    T = eltype(initial_x)
    initial_x = copy(initial_x)
    retract!(method.manifold, initial_x)

    value_gradient!!(d, initial_x)

    project_tangent!(method.manifold, gradient(d), initial_x)
    pg = copy(gradient(d))

    # Could move this out? as a general check?
    #=
    # Output messages
    isfinite(value(d)) || error("Initial f(x) is not finite ($(value(d)))")
    if !all(isfinite, gradient(d))
        @show gradient(d)
        @show find(.!isfinite.(gradient(d)))
        error("Gradient must have all finite values at starting point")
    end
    =#
    # Determine the intial search direction
    #    if we don't precondition, then this is an extra superfluous copy
    #    TODO: consider allowing a reference for pg instead of a copy
    method.precondprep!(method.P, initial_x)
    ldiv!(pg, method.P, gradient(d))
    if method.P !== nothing
        project_tangent!(method.manifold, pg, initial_x)
    end

    ConjugateGradientState(initial_x, # Maintain current state in state.x
                         0 .*(initial_x), # Maintain previous state in state.x_previous
                         0 .*(gradient(d)), # Store previous gradient in state.g_previous
                         real(T)(NaN), # Store previous f in state.f_x_previous
                         0 .*(initial_x), # Intermediate value in CG calculation
                         0 .*(initial_x), # Preconditioned intermediate value in CG calculation
                         pg, # Maintain the preconditioned gradient in pg
                         -pg, # Maintain current search direction in state.s
                         @initial_linesearch()...)
end

function update_state!(d, state::ConjugateGradientState, method::ConjugateGradient)
        # Search direction is predetermined

        # Maintain a record of the previous gradient
        copyto!(state.g_previous, gradient(d))

        # Determine the distance of movement along the search line
        lssuccess = perform_linesearch!(state, method, ManifoldObjective(method.manifold, d))

        # Update current position # x = x + alpha * s
        @. state.x = state.x + state.alpha * state.s
        retract!(method.manifold, state.x)

        # Update the function value and gradient
        value_gradient!(d, state.x)
        project_tangent!(method.manifold, gradient(d), state.x)

        # Check sanity of function and gradient
        isfinite(value(d)) || error("Non-finite f(x) while optimizing ($(value(d)))")

        # Determine the next search direction using HZ's CG rule
        #  Calculate the beta factor (HZ2013)
        # -----------------
        # Comment on py: one could replace the computation of py with
        #    ydotpgprev = dot(y, pg)
        #    dot(y, py)  >>>  dot(y, pg) - ydotpgprev
        # but I am worried about round-off here, so instead we make an
        # extra copy, which is probably minimal overhead.
        # -----------------
        method.precondprep!(method.P, state.x)
        @compat dPd = real(dot(state.s, method.P, state.s))
        etak = method.eta * real(dot(state.s, state.g_previous)) / dPd # New in HZ2013
        state.y .= gradient(d) .- state.g_previous
        ydots = real(dot(state.y, state.s))
        copyto!(state.py, state.pg)        # below, store pg - pg_previous in py
        ldiv!(state.pg, method.P, gradient(d))
        state.py .= state.pg .- state.py
        # ydots may be zero if f is not strongly convex or the line search does not satisfy Wolfe
        betak = (real(dot(state.y, state.pg)) - real(dot(state.y, state.py)) * real(dot(gradient(d), state.s)) / ydots) / ydots
        # betak may be undefined if ydots is zero (may due to f not strongly convex or non-Wolfe linesearch)
        beta = NaNMath.max(betak, etak) # TODO: Set to zero if betak is NaN?
        state.s .= beta.*state.s .- state.pg
        project_tangent!(method.manifold, state.s, state.x)
        lssuccess == false # break on linesearch error
end

update_g!(d, state, method::ConjugateGradient) = nothing

function trace!(tr, d, state, iteration, method::ConjugateGradient, options, curr_time=time())
  common_trace!(tr, d, state, iteration, method, options, curr_time)
end