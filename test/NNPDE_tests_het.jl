begin
push!(LOAD_PATH, "/home/zobot/.julia/dev/NeuralPDE.jl/src")
using Revise
using Flux
println("NNPDE_tests_heterogeneous")
using DiffEqFlux
println("Starting Soon!")
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
println("Starting Soon!")
using GalacticOptim
using Optim
using Quadrature,Cubature, Cuba
using QuasiMonteCarlo
using SciMLBase
using OrdinaryDiffEq
using Plots
using LineSearches
using Zygote

using Random
end

begin
Random.seed!(100)
cb = function (p,l)
    println("Current loss is: $l")
    return false
end

fnty(x) = fieldnames(typeof(x))
function domain_bounds_check(domains; epsilon=1e-2, print=false)
        lower_bounds = [domain.domain.lower for domain in domains]
        upper_bounds = [domain.domain.upper for domain in domains]
        function bounds_check(input, p)
                Zygote.@ignore begin
                        if print @show input end
                        if size(input)[1] != size(domains)[1]
                                @show input
                                @show domains
                                throw(DomainError((input_size=size(input), domain_size=size(domains)), "Input not correct size"))
                        end
                        bottom_out_of_bounds = input .<= lower_bounds .- epsilon
                        top_out_of_bounds = input .>= upper_bounds .+ epsilon
                        out_of_bounds = bottom_out_of_bounds .| top_out_of_bounds
                        if any(out_of_bounds)
                                @show input
                                @show domains
                                @show bottom_out_of_bounds
                                @show top_out_of_bounds
                                @show input[bottom_out_of_bounds]
                                @show input[top_out_of_bounds]
                                out_of_bounds_indices = map(I_bit->I_bit[1], filter(I_bit->I_bit[2], reshape(collect(zip(CartesianIndices(input),out_of_bounds)), prod(size(input)))))
                                @show collect(zip(out_of_bounds_indices, input[out_of_bounds_indices]))

                                throw(DomainError((input=input, domain=domains), "Input not within domain"))
                        end
                end
                input
        end
end

vector_initial_params(fastchains::Array) = vcat(initial_params.(fastchains)...)
vector_initial_params(discretization::NeuralPDE.PhysicsInformedNN) = vcat(discretization.init_params...)

dx = 0.1
grid_strategy = NeuralPDE.GridTraining(dx)
stochastic_strategy = NeuralPDE.StochasticTraining(64) #points
quadrature_strategy = NeuralPDE.QuadratureTraining(quadrature_alg=CubatureJLh(),
                                                    reltol=1e-3,abstol=1e-3,
                                                    maxiters =500, batch=100)
quasirandom_strategy = NeuralPDE.QuasiRandomTraining(100; #points
                                                     sampling_alg = UniformSample(),
                                                     minibatch = 100)

strategies = [grid_strategy, stochastic_strategy, quadrature_strategy,quasirandom_strategy]
strategies = [stochastic_strategy, quadrature_strategy]
#for strategy_ in strategies
    #test_heterogeneous_input(strategy_)
#end
strategy_ = strategies[2]

#=
println("Example 10, Simple Heterogeneous input PDE comparison, strategy: $strategy_")
@parameters x y
@variables r(..)
Dx = Differential(x)
Dy = Differential(y)

# 2D PDE
eq  = Dx(r(x,y)) + r(x, y) ~ 0

# Initial and boundary conditions
bcs = [
        r(x,-1) ~ 0.f0, r(1, y) ~ 0.0f0, 
        ]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
            y ∈ IntervalDomain(-1.0,0.0)]

# chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
numhid = 3
fastchains = FastChain(FastDense(2,numhid,Flux.σ),FastDense(numhid,numhid,Flux.σ),FastDense(numhid,1))
discretization = NeuralPDE.PhysicsInformedNN(fastchains,
                                                strategy_)

pde_system = PDESystem(eq,bcs,domains,[x,y],[r(x,y)])
end
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
@run sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
prob = NeuralPDE.discretize(pde_system,discretization)
initθ = discretization.init_params
initθvec = vcat(initθ...)
prob.f(initθvec, [])
@run prob.f(initθvec, [])
=#

println("Example 10, Simple Heterogeneous input PDE, strategy: $strategy_")
@parameters x y
@variables p(..) q(..) r(..) s(..)
Dx = Differential(x)
Dy = Differential(y)

# 2D PDE
eq  = p(x) + q(y) + r(x, y) + s(y, x) ~ 0
#eq  = Dx(p(x)) + p(x) + Dy(q(y)) + q(y) + Dy(r(x, y)) + r(x, y) + Dx(s(y, x)) + s(y, x) ~ 0
#eq  = Dx(p(x)) + Dy(q(y)) + Dx(r(x, y)) + Dy(s(y, x)) + p(x) + q(y) + r(x, y) + s(y, x) ~ 0

# Initial and boundary conditions
#bcs = [p(1) ~ 0.f0, q(-1) ~ 0.0f0,
        #r(x,-1) ~ 0.f0, r(1, y) ~ 0.0f0, 
        #s(y,1) ~ 0.0f0, s(-1, x) ~ 0.0f0]
bcs = [p(1) ~ 0.f0, q(-1) ~ 1.0f0,
        r(x,-1) ~ 0.f0, r(1, y) ~ -1.0f0, 
        s(y,1) ~ -1.0f0, s(-1, x) ~ 0.0f0]
#bcs = [s(y, 1) ~ 0.0f0]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
            y ∈ IntervalDomain(-1.0,0.0)]

# chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
numhid = 3
indomain = [[1], [2], [1,2], [2,1]]
fastchains = [FastChain(domain_bounds_check(domains[indomain[i]]; print=false, epsilon=1e-2),FastDense(length(indomain[i]),numhid,Flux.σ),FastDense(numhid,numhid,Flux.σ),FastDense(numhid,1)) for i in 1:4]
discretization = NeuralPDE.PhysicsInformedNN(fastchains,
                                                strategy_)

pde_system = PDESystem(eq,bcs,domains,[x,y],[p(x), q(y), r(x,y), s(y,x)])
end

begin
input = vcat(0.9 .* ones(1, 6), -0.9 .* ones(1, 6))
[fastchains[index](input[indomain[index], :], initial_params.(fastchains)[index]) for index in 1:4]
end

initθ = vector_initial_params(fastchains)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
#@run sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
prob = NeuralPDE.discretize(pde_system,discretization)
initθ = vector_initial_params(discretization)
prob.f(initθ, [])
#@run prob.f(initθ, [])
resmany_samples = GalacticOptim.solve(prob2, ADAM(3e-4); cb = cb,  maxiters=50_000)

prob2 = remake(prob, u0=resmany_samples.minimizer)
resmany_samples_2 = GalacticOptim.solve(prob2, Optim.LBFGS(;linesearch=LineSearches.MoreThuente()) allow_f_increases=true, cb = cb,  maxiters=1000)
dxs = (0.0:0.01:1.0)
dys = (-1.0:0.01:0.0)
dxdys = [[dx, dy] for dx in dxs, dy in dys]
i = 3
untrained_evaluations = [map(dxdy->discretization.phi[i](dxdy[indomain[i]], initial_params(fastchains[i]))[1], dxdys) for i in 1:4]

param_lengths = (length ∘ initial_params).(fastchains)
indices_in_params = map(zip(param_lengths, cumsum(param_lengths))) do (param_length, cumsum_param)
        cumsum_param - (param_length - 1) : cumsum_param
end
trained_evaluations = [map(dxdy->discretization.phi[i](dxdy[indomain[i]], res[indices_in_params[i]])[1], dxdys) for i in 1:4]
supertrained_evaluations = [map(dxdy->discretization.phi[i](dxdy[indomain[i]], reslong[indices_in_params[i]])[1], dxdys) for i in 1:4]
manysamples_trained_evaluations = [map(dxdy->discretization.phi[i](dxdy[indomain[i]], resmany_samples[indices_in_params[i]])[1], dxdys) for i in 1:4]
begin
p1 = plot(dxs, dys, untrained_evaluations[1], linetype=:contourf,title = "p");
p2 = plot(dxs, dys, untrained_evaluations[2], linetype=:contourf,title = "q");
p3 = plot(dxs, dys, untrained_evaluations[3], linetype=:contourf,title = "r");
p4 = plot(dxs, dys, untrained_evaluations[4], linetype=:contourf,title = "s");
plot(p1,p2,p3, p4)
end
begin
p1 = plot(dxs, dys, trained_evaluations[1], linetype=:contourf,title = "p");
p2 = plot(dxs, dys, trained_evaluations[2], linetype=:contourf,title = "q");
p3 = plot(dxs, dys, trained_evaluations[3], linetype=:contourf,title = "r");
p4 = plot(dxs, dys, trained_evaluations[4], linetype=:contourf,title = "s");
plot(p1,p2,p3, p4)
end
begin
p1 = plot(dxs, dys, supertrained_evaluations[1], linetype=:contourf,title = "p");
p2 = plot(dxs, dys, supertrained_evaluations[2], linetype=:contourf,title = "q");
p3 = plot(dxs, dys, supertrained_evaluations[3], linetype=:contourf,title = "r");
p4 = plot(dxs, dys, supertrained_evaluations[4], linetype=:contourf,title = "s");
plot(p1,p2,p3, p4)
end
begin
p1 = plot(dxs, dys, manysamples_trained_evaluations[1], linetype=:contourf,title = "p");
p2 = plot(dxs, dys, manysamples_trained_evaluations[2], linetype=:contourf,title = "q");
p3 = plot(dxs, dys, manysamples_trained_evaluations[3], linetype=:contourf,title = "r");
p4 = plot(dxs, dys, manysamples_trained_evaluations[4], linetype=:contourf,title = "s");
plot(p1,p2,p3, p4)
end

begin
p1=plot(dxs, dys, sum([map(dxdy->discretization.phi[i](dxdy[indomain[i]], initial_params(fastchains[i]))[1], dxdys) for i in 1:4]), linetype=:contourf,title="p+q+r+s")
p2=plot(dxs, dys, sum([map(dxdy->discretization.phi[i](dxdy[indomain[i]], resmany_samples[indices_in_params[i]])[1], dxdys) for i in 1:4]), linetype=:contourf,title="p+q+r+s")
#p3=plot(dxs, dys, sum([map(dxdy->discretization.phi[i](dxdy[indomain[i]], reslong[indices_in_params[i]])[1], dxdys) for i in 1:4]), linetype=:contourf,title="p+q+r+s")
plot(p1,p2)
end


prob.f(initθvec, [])

#@run prob.f(initθvec, [])

#res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters=3)


#=
phi = discretization.phi
eqs = pde_system.eqs
bcs = pde_system.bcs

domains = pde_system.domain
eq_params = pde_system.ps
defaults = pde_system.defaults
default_p = eq_params == SciMLBase.NullParameters() ? nothing : [defaults[ep] for ep in eq_params]

param_estim = discretization.param_estim
additional_loss = discretization.additional_loss

# dimensionality of equation
dim = length(domains)
depvars,indvars,dict_indvars,dict_depvars,dict_depvar_input = NeuralPDE.get_vars(pde_system.indvars,pde_system.depvars)

chain = discretization.chain
initθ = discretization.init_params
flat_initθ = if (typeof(chain) <: AbstractVector) vcat(initθ...) else  initθ end
flat_initθ = if param_estim == false flat_initθ else vcat(flat_initθ, adapt(DiffEqBase.parameterless_type(flat_initθ),default_p)) end
phi = discretization.phi
derivative = discretization.derivative
strategy = discretization.strategy
if !(eqs isa Array)
    eqs = [eqs]
end
pde_indvars = if strategy isa QuadratureTraining
        NeuralPDE.get_argument(eqs,dict_indvars,dict_depvars)
else
        NeuralPDE.get_variables(eqs,dict_indvars,dict_depvars)
end
_pde_loss_functions = [NeuralPDE.build_loss_function(eq,indvars,depvars,
                                            dict_indvars,dict_depvars,dict_depvar_input,
                                            phi, derivative,chain, initθ,strategy,eq_params=eq_params,param_estim=param_estim,default_p=default_p,
                                            bc_indvars = pde_indvar) for (eq, pde_indvar) in zip(eqs,pde_indvars)]
bc_indvars = if strategy isa QuadratureTraining
        get_argument(bcs,dict_indvars,dict_depvars)
else
        get_variables(bcs,dict_indvars,dict_depvars)
end

_bc_loss_functions = [build_loss_function(bc,indvars,depvars,
                                                dict_indvars,dict_depvars,dict_depvar_input,
                                                phi, derivative,chain, initθ, strategy,eq_params=eq_params,param_estim=param_estim,default_p=default_p;
                                                bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]
=#

begin
println("example 11 different heterogeneous")
@parameters x y z
@variables p(..) q(..) r(..)
Dx = Differential(x)
Dy = Differential(y)
Dz = Differential(z)

# 2D PDE
eqs  = [Dx(p(x)) + q(x, y) ~ 0,
        Dx(p(x)) + r(x, z) ~ 0,
        Dx(q(x, y)) + Dy(Dy(q(x, y))) ~ 0,
        Dx(r(x, z)) + Dz(Dz(r(x, z))) ~ 0]
#eq  = Dx(p(x)) + p(x) + Dy(q(y)) + q(y) + Dy(r(x, y)) + r(x, y) + Dx(s(y, x)) + s(y, x) ~ 0
#eq  = Dx(p(x)) + Dy(q(y)) + Dx(r(x, y)) + Dy(s(y, x)) + p(x) + q(y) + r(x, y) + s(y, x) ~ 0

# Initial and boundary conditions
#bcs = [p(1) ~ 0.f0, q(-1) ~ 0.0f0,
        #r(x,-1) ~ 0.f0, r(1, y) ~ 0.0f0, 
        #s(y,1) ~ 0.0f0, s(-1, x) ~ 0.0f0]
bcs = [p(0) ~ 1.f0, 
       ]
#bcs = [s(y, 1) ~ 0.0f0]
# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
            y ∈ IntervalDomain(-1.0,0.0),
            z ∈ IntervalDomain(1.0, 2.0)]

# chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
numhid = 4
indomain = [[1], [1, 2], [1, 3]]
fastchains = [FastChain(domain_bounds_check(domains[indomain[i]]; print=false, epsilon=1e-2),FastDense(length(indomain[i]),numhid,Flux.σ),FastDense(numhid,numhid,Flux.σ),FastDense(numhid,1)) for i in 1:4]
discretization = NeuralPDE.PhysicsInformedNN(fastchains,
                                                strategy_)

pde_system = PDESystem(eq,bcs,domains,[x,y,z],[p(x), q(x, y), r(x,z)])