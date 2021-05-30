#begin
#using DrWatson
#import DrWatson.savename
#end
#@quickactivate "NeuralPDEWatson"
begin
push!(LOAD_PATH, "/home/zobot/.julia/dev/NeuralPDE.jl/src")
using Revise
using LinearAlgebra
using IfElse
using PyCall
using Flux
println("NNPDE_tests_heterogeneous")
using DiffEqFlux
println("Starting Soon!")
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
using GalacticOptim
using Optim
using Quadrature,Cubature, Cuba
using QuasiMonteCarlo
using SciMLBase
using DelimitedFiles
using CSV
using OrdinaryDiffEq
using Plots
using DataFrames
using LineSearches
using Zygote

using Random
end

begin
@pyimport pickle
@pyimport numpy

function myunpickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename,"rb") as f begin
        r = pickle.load(f)
    end
    return r
end
const rampupr = 0.05
const righttargetrn = 0.8000000000000016
const righttargetrp = 0.6000000000000001
const rightfluxrn = -0.14182855923368468
const rightfluxrp = 0.03237700710041634
function rampupcubic(r, targetflux, targetval)
    IfElse.ifelse(r < 1.0 .- rampupr, targetval, targetval .+ (r .- (1.0 .- rampupr)).^3 .* targetflux ./ (3 .* rampupr.^2) )
end
@register rampupcubic(r,f,v)
function rampupquadratic(r, targetflux, targetval)
    IfElse.ifelse(r < 1.0 .- rampupr, targetval, targetval .+ (r .- (1.0 .- rampupr)).^2 .* targetflux ./ (2rampupr) )
end
@register rampupquadratic(r,f,v)
end

function main_train()
#begin
    pybamm_sols = myunpickle("/home/zobot/.julia/dev/DFN.jl/pybamm/hardcoded_models/MTK_format/pybamm_solutions/SPM.pickle")

    sols_t = pybamm_sols["Time"]
    sols_r_n = pybamm_sols["r_n"][:,1]
    sols_r_p = pybamm_sols["r_p"][:,1]
    sols_c_s_n = pybamm_sols["X-averaged negative particle concentration"]
    sols_c_s_p = pybamm_sols["X-averaged positive particle concentration"]
    begin
    # ('negative particle',) -> rn
    # ('positive particle',) -> rp
    @parameters t rn rp
    # 'Discharge capacity [A.h]' -> Q
    # 'X-averaged negative particle concentration' -> c_s_n_xav
    # 'X-averaged positive particle concentration' -> c_s_p_xav
    @variables Q(..) c_s_n_xav(..) c_s_p_xav(..)
    #@variables Q(t) c_s_n_xav(t, rn) c_s_p_xav(t, rp)
    Dt = Differential(t)
    Drn = Differential(rn)
    Drp = Differential(rp)
    # 'X-averaged negative particle concentration' equation
    #cache_4647021298618652029 = 8.813457647415216 * (1 / rn^2 * Drn(rn^2 * Drn(c_s_n_xav(t, rn))))
    cache_4647021298618652029 = 8.813457647415216 * (Drn(Drn(c_s_n_xav(t, rn))) + 2 / rn * Drn(c_s_n_xav(t, rn)))

    # 'X-averaged positive particle concentration' equation
    #cache_m620026786820028969 = 22.598609352346717 * (1 / rp^2 * Drp(rp^2 * Drp(c_s_p_xav(t, rp))))
    cache_m620026786820028969 = 22.598609352346717 * (Drp(Drp(c_s_p_xav(t, rp))) + 2 / rp * Drp(c_s_p_xav(t, rp)))


    eqs = [
    Dt(Q(t)) ~ 4.27249308415467,
    Dt(c_s_n_xav(t, rn)) ~ cache_4647021298618652029,
    Dt(c_s_p_xav(t, rp)) ~ cache_m620026786820028969,
    ]



    # solve for initial derivative at 0, end derivative at targetflux




    ics_bcs = [
    Q(0) ~ 0.0,
    c_s_n_xav(0, rn) ~ rampupquadratic(rn, rightfluxrn, righttargetrn),
    c_s_p_xav(0, rp) ~ rampupquadratic(rp, rightfluxrp, righttargetrp),
    Drn(c_s_n_xav(t, 0.01)) ~ 0.0,
    Drn(c_s_n_xav(t, 1.0)) ~ -0.14182855923368468,
    Drp(c_s_p_xav(t, 0.01)) ~ 0.0,
    Drp(c_s_p_xav(t, 1.0)) ~ 0.03237700710041634,
    ]

    t_domain = IntervalDomain(0.0, 0.15) # 0.15
    rn_domain = IntervalDomain(0.01, 1.0)
    rp_domain = IntervalDomain(0.01, 1.0)

    domains = [
    t in t_domain,
    rn in rn_domain,
    rp in rp_domain,
    ]
    ind_vars = [t, rn, rp]
    dep_vars = [Q(t), c_s_n_xav(t, rn), c_s_p_xav(t, rp)]

    SPM_pde_system = PDESystem(eqs, ics_bcs, domains, ind_vars, dep_vars)
    end



    ## PINN Part


    begin
    num_dim = 50
    nonlin = Flux.gelu
    #in_dim = 3
    #out_dim = 3
    strategy = NeuralPDE.QuadratureTraining(;quadrature_alg=HCubatureJL(),abstol=1e-4, reltol=1, maxiters=1000, batch=0)
    #strategy_ = NeuralPDE.QuadratureTraining(;abstol=1e-6, reltol=1e-8, maxiters=2000)
    #strategy = NeuralPDE.QuadratureTraining(;quadrature_alg=HCubatureJL(), batch=0)
    #strategy = NeuralPDE.StochasticTraining(128)
    #strategy = NeuralPDE.QuadratureTraining()
    #strategy = NeuralPDE.QuadratureTraining(;quadrature_alg=HCubatureJL(),
                                                        #reltol=1e-3,abstol=1e-5,
                                                        #maxiters =2000, batch=0)

    #in_dims = [3, 3, 3]
    #in_dims = [1, 2, 2]
    in_dims = [1, 2, 2]
    num_hid = 2
    chains_ = [FastChain(FastDense(in_dim,num_dim,nonlin),
                        [FastDense(num_dim,num_dim,nonlin) for i in 1:num_hid]...,
                        FastDense(num_dim,1)) for in_dim in in_dims]
    adaloss = NeuralPDE.NonAdaptiveLossWeights()
    #adaloss = NeuralPDE.LossGradientsAdaptiveLoss(20; α=0.9f0)
    #adaloss = NeuralPDE.MiniMaxAdaptiveLoss(20; α_pde=1e-1, α_bc=1e1, pde_weights_start=1e-3, bc_weights_start=1e3)
    data_dir = "/home/zobot/.julia/dev/NeuralPDEWatson/data"
    exp_name = "minimax_consistent_nonadaptive"
    exp_folder = joinpath(data_dir, exp_name)
    wipe_logs = false
    discretization = NeuralPDE.PhysicsInformedNN(chains_,
                                                    strategy; adaptive_loss=adaloss, exp_name=exp_name, data_dir=data_dir, wipe=wipe_logs)
end
    sym_prob = NeuralPDE.symbolic_discretize(SPM_pde_system,discretization)
    prob = NeuralPDE.discretize(SPM_pde_system,discretization)
begin
                                                    
    end

    #@run sym_prob = NeuralPDE.symbolic_discretize(SPM_pde_system,discretization)
    #@run sym_prob = NeuralPDE.symbolic_discretize(SPM_pde_system,discretization)
    begin
    initθ = vcat(discretization.init_params...)
    #opt = Flux.Optimiser(ClipValue(1e-3), ExpDecay(1, 0.5, 25_000), ADAM(3e-4))
    opt = ADAM(3e-4)
    saveevery = 100
    loss = zeros(Float64, saveevery)

    losssavefile = joinpath(exp_folder, "loss.csv")

    if wipe_logs
        rm(losssavefile; force=true)
    end
    iteration_count_arr = [1]
    cb = function (p,l)
        iteration_count = iteration_count_arr[1]


        println("Current loss is: $l, iteration is: $(iteration_count)")
        loss[((iteration_count - 1) % saveevery) + 1] = l
        if iteration_count % saveevery == 0
            cursavefile = joinpath(exp_folder, string(iteration_count, base=10, pad=5) * ".csv")
            writedlm(cursavefile, p, ",")
            df = DataFrame(loss=loss)
            CSV.write(losssavefile, df, writeheader=false, append=true)
        end
        iteration_count_arr[1] += 1
        return false
    end
    end
    prob.f(initθ, [])
    res = GalacticOptim.solve(prob, ADAM(3e-4); cb = cb, maxiters=50_000)
end
main_train()
