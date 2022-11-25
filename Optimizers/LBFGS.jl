### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# ╔═╡ 6bba6acc-8202-497f-884e-87f0dc440d89
abstract type AbstractOptimiser end

# ╔═╡ 4e21fa68-5fd7-4bb8-bff6-a113fdd225ea
# heavily inspired by https://github.com/simonbatzner/L-BFGS-Julia/blob/master/L-BFGS_Project.ipynb/src/SLBFGS.jl
# use https://github.com/baggepinnen/FluxOptTools.jl to get f 
mutable struct LBFGS <: AbstractOptimiser
	n::Int # number of variables
	lossfun::Function # loss function for backtracking search
    m::Int # Memory length, was ∈ [2, 54] in paper
	prev_g::Float64 # gradient at previous timestep 
	prev_x::Float64 # x at previous timestep 
	Sm # previous m x's
	Ym # previous m gradients
	k::Int # Internal iteration index

	function LBFGS(n)
		m = 20 
		prev_g = 0 
		prev_x = 0 
		Sm = zeros(n,m)
		Ym = zeros(n,m)
		k = 0 

		new(n, 0, m, prev_g, prev_x, Sm, Ym, k)
	end
end

# ╔═╡ 47f702a4-0113-460a-9ea9-9abe6601a8f2
function backtracking(F,d,x,r=0.5,c=1e-4,nmax=10)
    
    # params
    # F: function to be optimized
    # x: variable
    # d: direction
    # r: factor by which to reduce step size at each iteration
    # c: parameter [0,1]
    # nmax: max iteration

    # return
    # α step size
    # fk1: function value at new x
    # gkk: gradient at new x

    #https://en.wikipedia.org/wiki/Backtracking_line_search
    α=1
    fk,gk=F(x)
    xx=x
    x=x+α*d
    fk1,gk1=F(x)
    n=1
    
    while fk1>fk+c*α*(gk'*d) && n < nmax
        n=n+1
        α=α*0.5
        x=xx+α*d
        fk1,gk1=F(x)
    end
    
    return α, fk1, gk1
end

# ╔═╡ e61911d8-2d8b-4cf7-a14f-62afd15f1f53
function approxInvHess(g,S,Y,H0)
    #INPUT

    #g: gradient nx1 vector
    #S: nxk matrixs storing S[i]=x[i+1]-x[i]
    #Y: nxk matrixs storing Y[i]=g[i+1]-g[i]
    #H0: initial hessian diagnol scalar

    #OUTPUT
    # p:  the approximate inverse hessian multiplied by the gradient g
    #     which is the new direction
    #notation follows:
    #https://en.wikipedia.org/wiki/Limited-memory_BFGS

    n,k=size(S)
    rho=zeros(k)
    for i=1:k
        rho[i].=1/(Y[:,i]'*S[:,i])
        if rho[i]<0
            rho[i]=-rho[i]
        end
    end


    q=zeros(n,k+1)
    r=zeros(n,1)
    α=zeros(k,1)
    β=zeros(k,1)

    q[:,k+1]=g

    for i=k:-1:1
        α[i].=rho[i]*S[:,i]'*q[:,i+1]
        q[:,i].=q[:,i+1]-α[i]*Y[:,i]
    end

    z=zeros(2)
    z.=H0*q[:,1]


    for i=1:k
        β[i].=rho[i]*Y[:,i]'*z
        z.=z+S[:,i]*(α[i]-β[i])
    end

    p=copy(z)

    return p
end


# ╔═╡ 7bb665d1-5ca9-4121-a7a6-1542a7c3f8f1
function apply!(o::LBFGS, x, Δ)

	o.k += 1
	m = o.m
	
	gnorm = norm(g0)
	
	# if gnorm < τgrad # tolerance for the norm of the slope 
	# 	return; 
	# end
	
	s0 = x-o.prev_x
	y0 = ∇-o.prev_g
	
	# println("y0=$y0")
	H0 = s0'*y0/(y0'*y0) # hessian diagonal satisfying secant condition

	# update Sm and Ym
	if o.k <= m
		o.Sm[:,k].=s0
		o.Ym[:,k].=y0
		p=-approxInvHess(∇,o.Sm[:,1:k],o.Ym[:,1:k],H0) 
	# only keep m entries in Sm and Ym so purge the old ones
		
	else
		o.Sm[:,1:(m-1)].=o.Sm[:,2:m]
		o.Ym[:,1:(m-1)].=o.Sm[:,2:m]
		o.Sm[:,m].=s0
		o.Ym[:,m].=y0
		p.=-approxInvHess(∇,o.Sm,o.Ym,H0)
	end
	
	# new direction=p, find new step size
	α, fs, gs=backtracking(F,p,x)
	
	# update for next iteration
	o.prev_x = x
	o.prev_g = ∇
	x .= x + α.*p
	f1=fs
	g1=gs
	k=k+1
	
	if verbose == 1 
		println("Iteration: $k -- x = $x1")
	end
    
end

# ╔═╡ Cell order:
# ╠═6bba6acc-8202-497f-884e-87f0dc440d89
# ╠═4e21fa68-5fd7-4bb8-bff6-a113fdd225ea
# ╠═47f702a4-0113-460a-9ea9-9abe6601a8f2
# ╠═e61911d8-2d8b-4cf7-a14f-62afd15f1f53
# ╠═7bb665d1-5ca9-4121-a7a6-1542a7c3f8f1
