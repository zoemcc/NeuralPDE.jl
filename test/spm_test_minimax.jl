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
end

#function main_train()
begin
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

    rampuptime = 0.01
    ics_bcs = [
    Q(0) ~ 0.0,
    c_s_n_xav(0, rn) ~ 0.8000000000000016,
    c_s_p_xav(0, rp) ~ 0.6000000000000001,
    Drn(c_s_n_xav(t, 0.01)) ~ 0.0,
    Drn(c_s_n_xav(t, 1.0)) ~ -0.14182855923368468,
    #Drn(c_s_n_xav(t, 1.0)) ~ -0.14182855923368468 / (1 .+ exp(-t / rampuptime * 2)),
        #min.(t, rampuptime) / rampuptime * -0.14182855923368468,
        #IfElse.ifelse(t < [rampuptime],  t / rampuptime * -0.14182855923368468, -0.14182855923368468),
          #t / rampuptime * -0.14182855923368468, -0.14182855923368468),

    Drp(c_s_p_xav(t, 0.01)) ~ 0.0,
    Drp(c_s_p_xav(t, 1.0)) ~ 0.03237700710041634,
    #Drp(c_s_p_xav(t, 1.0)) ~  0.03237700710041634 / (1 .+ exp(-t / rampuptime * 2)),
        #IfElse.ifelse(t < [rampuptime],  t / rampuptime * 0.03237700710041634, 0.03237700710041634),
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
    strategy = NeuralPDE.QuadratureTraining(;quadrature_alg=HCubatureJL(),abstol=1e-4, reltol=1, maxiters=2, batch=0)
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
    #adalosspoisson = NeuralPDE.LossGradientsAdaptiveLoss(20; α=0.9f0)
    adaloss = NeuralPDE.MiniMaxAdaptiveLoss(20; α_pde=1e-1, α_bc=1e1, pde_weights_start=1e-3, bc_weights_start=1e3)
    discretization = NeuralPDE.PhysicsInformedNN(chains_,
                                                    strategy; adaptive_loss=adaloss)
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

    experiment_path = "/home/zobot/.julia/dev/NeuralPDEWatson/data/minimax_spm_first/"
    losssavefile = joinpath(experiment_path, "loss.csv")
    rm(losssavefile; force=true)
    iteration_count_arr = [1]
    cb = function (p,l)
        iteration_count = iteration_count_arr[1]


        println("Current loss is: $l, iteration is: $(iteration_count)")
        loss[((iteration_count - 1) % saveevery) + 1] = l
        if iteration_count % saveevery == 0
            cursavefile = joinpath(experiment_path, string(iteration_count, base=10, pad=5) * ".csv")
            writedlm(cursavefile, p, ",")
            df = DataFrame(loss=loss)
            CSV.write(losssavefile, df, writeheader=false, append=true)
        end
        iteration_count_arr[1] += 1
        return false
    end
    end
    prob.f(initθ, [])
    #res = GalacticOptim.solve(prob, ADAM(3e-4); cb = cb, maxiters=50_000)
end
#main_train()


function generate_supervised_loss()
    c_s_n_input_length = length(sols_t) * length(sols_r_n)
    c_s_p_input_length = length(sols_t) * length(sols_r_p)
    c_s_n_inputs = hcat(reshape([[t_i, r_n_i] for r_n_i in sols_r_n, t_i in sols_t], (c_s_n_input_length,))...)
    c_s_p_inputs = hcat(reshape([[t_i, r_p_i] for r_p_i in sols_r_p, t_i in sols_t], (c_s_p_input_length,))...)
    c_s_n_sup = reshape(sols_c_s_n, (1, c_s_n_input_length,)) 
    c_s_p_sup = reshape(sols_c_s_p, (1, c_s_p_input_length,)) 

    param_lengths = (length ∘ initial_params).(chains_)
    indices_in_params = map(zip(param_lengths, cumsum(param_lengths))) do (param_length, cumsum_param)
            cumsum_param - (param_length - 1) : cumsum_param
    end
    function supervised_additional_loss(phi, θ)
        c_s_n_evals = chains_[2](c_s_n_inputs, θ[indices_in_params[2]])
        c_s_p_evals = chains_[3](c_s_p_inputs, θ[indices_in_params[3]])

        c_s_n_l2 = sum(abs2, c_s_n_evals .- c_s_n_sup) / c_s_n_input_length
        c_s_p_l2 = sum(abs2, c_s_p_evals .- c_s_p_sup) / c_s_p_input_length
        c_s_n_l2 + c_s_p_l2
    end
    function supervised_loss_only(θ, p)
        c_s_n_evals = chains_[2](c_s_n_inputs, θ[indices_in_params[2]])
        c_s_p_evals = chains_[3](c_s_p_inputs, θ[indices_in_params[3]])

        c_s_n_l2 = sum(abs2, c_s_n_evals .- c_s_n_sup) / c_s_n_input_length
        c_s_p_l2 = sum(abs2, c_s_p_evals .- c_s_p_sup) / c_s_p_input_length
        c_s_n_l2 + c_s_p_l2
    end

    (supervised_additional_loss, supervised_loss_only)
end


f = OptimizationFunction(generate_supervised_loss()[2], GalacticOptim.AutoZygote())
supervised_opt_prob = GalacticOptim.OptimizationProblem(f, initθ)
supervisedcb = function (p,l)
    println("Current loss is: $l")
    return false
end
#supervised_res = GalacticOptim.solve(supervised_opt_prob, ADAM(3e-4); cb = supervisedcb, maxiters=1000)


discretization_supervised = NeuralPDE.PhysicsInformedNN(chains_,
                                                strategy; )#; additional_loss=generate_supervised_loss()[1])
prob_supervised = NeuralPDE.discretize(SPM_pde_system,discretization_supervised)
prob_post_supervised = remake(prob_supervised, u0=supervised_res)

post_supervised_res = GalacticOptim.solve(prob_post_supervised, ADAM(3e-4); cb = cb, maxiters=50_000)

#(supervised_res, post_supervised_res)

#(supervised_res, post_supervised_res) = train_supervised()

#@run prob.f(initθ, [])
#@run res = GalacticOptim.solve(prob, opt; cb = cb, maxiters=100)

pretrained_params_file = joinpath(experiment_path, "50000.csv")
pretrained_pinn_params = Array(CSV.read(pretrained_params_file, DataFrame; header=false))[:,1]
params = pretrained_pinn_params


quadrature_fixed_first_experiment_path = "/home/zobot/.julia/dev/NeuralPDEWatson/data/quadrature_fixed_first/"
quadrature_params_file = joinpath(quadrature_fixed_first_experiment_path, "17600.csv")
quadrature_params = Array(CSV.read(quadrature_params_file, DataFrame; header=false))[:,1]

dts = (0.0:0.0015:0.15)
drns = (0.01:0.01:1.0)
drps = (0.01:0.01:1.0)


dts = sols_t
drns = sols_r_n
drps = sols_r_p


param_lengths = (length ∘ initial_params).(chains_)
indices_in_params = map(zip(param_lengths, cumsum(param_lengths))) do (param_length, cumsum_param)
        cumsum_param - (param_length - 1) : cumsum_param
end

phi_i_params = [params[indices_in_params_i] for indices_in_params_i in indices_in_params]
phi_i_params_pretrained = [supervised_res[indices_in_params_i] for indices_in_params_i in indices_in_params]
phi_i_params_quadrature = [quadrature_params[indices_in_params_i] for indices_in_params_i in indices_in_params]

Q_evals = [chains_[1]([dt], phi_i_params[1])[1] for dt in sols_t]

c_s_n_evals_pretrained_pinn = [chains_[2]([dt, drn], phi_i_params[2])[1] for dt in dts, drn in drns]
c_s_p_evals_pretrained_pinn = [chains_[3]([dt, drp], phi_i_params[3])[1] for dt in dts, drp in drps]
c_s_n_evals_pretrained = [chains_[2]([dt, drn], phi_i_params_pretrained[2])[1] for dt in dts, drn in drns]
c_s_p_evals_pretrained = [chains_[3]([dt, drp], phi_i_params_pretrained[3])[1] for dt in dts, drp in drps]
c_s_n_evals_quadrature = [chains_[2]([dt, drn], phi_i_params_quadrature[2])[1] for dt in dts, drn in drns]
c_s_p_evals_quadrature = [chains_[3]([dt, drp], phi_i_params_quadrature[3])[1] for dt in dts, drp in drps]

p1 = plot(drns,sols_c_s_n[:, 1];ylims=(0,1),ylabel="c_s_n",xlabel="r_n",legend=true, label="FDM")
p1 = plot!(p1,drns,c_s_n_evals_pretrained_pinn[1,:];ylims=(0,1),ylabel="c_s_n",xlabel="r_n",legend=true, label="Pretrained pinn")
p1 = plot!(p1,drns,c_s_n_evals_pretrained[1,:];ylims=(0,1),ylabel="c_s_n",xlabel="r_n",legend=true, label="Supervised_only")
p1 = plot!(p1,drns,c_s_n_evals_quadrature[1,:];ylims=(0,1),ylabel="c_s_n",xlabel="r_n",legend=true, label="Pinn only")
p2 = plot(drps,sols_c_s_p[:, 1];ylims=(0,1),ylabel="c_s_p",xlabel="r_p",legend=true, label="FDM")
p2 = plot!(p2,drps,c_s_p_evals_pretrained_pinn[1,:];ylims=(0,1),ylabel="c_s_p",xlabel="r_p",legend=true, label="Pretrained pinn")
p2 = plot!(p2,drps,c_s_p_evals_pretrained[1,:];ylims=(0,1),ylabel="c_s_p",xlabel="r_p",legend=true, label="Supervised_only")
p2 = plot!(p2,drps,c_s_p_evals_quadrature[1,:];ylims=(0,1),ylabel="c_s_p",xlabel="r_p",legend=true, label="Pinn only")

anim = @animate for i in 1:length(dts)
    #p1 = plot(drns,c_s_n_evals[i,:];ylims=(0,1),ylabel="c_s_n",xlabel="r_n",legend=false)
    #p2 = plot(drps,c_s_p_evals[i,:];ylims=(0,1),ylabel="c_s_p",xlabel="r_p",legend=false)
    p1 = plot(drns,sols_c_s_n[:, i];ylims=(0,1.3),ylabel="c_s_n",xlabel="r_n",legend=true, label="FDM")
    p1 = plot!(p1,drns,c_s_n_evals_pretrained_pinn[i,:];ylims=(0,1.3),ylabel="c_s_n",xlabel="r_n",legend=true, label="Pretrained pinn", linestyle=:dashdot)
    p1 = plot!(p1,drns,c_s_n_evals_pretrained[i,:];ylims=(0,1.3),ylabel="c_s_n",xlabel="r_n",legend=true, label="Supervised_only", linestyle=:dash)
    p1 = plot!(p1,drns,c_s_n_evals_quadrature[i,:];ylims=(0,1.3),ylabel="c_s_n",xlabel="r_n",legend=true, label="Pinn only", linestyle=:dot)
    p2 = plot(drps,sols_c_s_p[:, i];ylims=(0,1.3),ylabel="c_s_p",xlabel="r_p",legend=true, label="FDM")
    p2 = plot!(p2,drps,c_s_p_evals_pretrained_pinn[i,:];ylims=(0,1.3),ylabel="c_s_p",xlabel="r_p",legend=true, label="Pretrained pinn", linestyle=:dashdot)
    p2 = plot!(p2,drps,c_s_p_evals_pretrained[i,:];ylims=(0,1.3),ylabel="c_s_p",xlabel="r_p",legend=true, label="Supervised_only", linestyle=:dash)
    p2 = plot!(p2,drps,c_s_p_evals_quadrature[i,:];ylims=(0,1.3),ylabel="c_s_p",xlabel="r_p",legend=true, label="Pinn only", linestyle=:dot)
    plot(p1,p2)
end
anim = @animate for i in 1:length(dts)
    p1 = plot(drns,sols_c_s_n[:, i];ylims=(0,1.3),ylabel="c_s_n",xlabel="r_n",legend=true, label="FDM")
    #p1 = plot(drns,c_s_n_evals_pretrained_pinn[i, :];ylims=(0,1.3),ylabel="c_s_n",xlabel="r_n",legend=true, label="FDM")
    p1 = plot!(p1,drns,c_s_n_evals_pretrained_pinn[i,:];ylims=(0,1.3),ylabel="c_s_n",xlabel="r_n",legend=true, label="Fulltrained pinn", linestyle=:dashdot)
    p2 = plot(drps,sols_c_s_p[:, i];ylims=(0,1.3),ylabel="c_s_p",xlabel="r_p",legend=true, label="FDM")
    #p2 = plot(drps,c_s_p_evals_pretrained_pinn[i, :];ylims=(0,1.3),ylabel="c_s_p",xlabel="r_p",legend=true, label="FDM")
    p2 = plot!(p2,drps,c_s_p_evals_pretrained_pinn[i,:];ylims=(0,1.3),ylabel="c_s_p",xlabel="r_p",legend=true, label="Fulltrained pinn", linestyle=:dashdot)
    plot(p1,p2)
end
gif(anim, "/home/zobot/.julia/dev/NeuralPDEWatson/data/minimax_spm_first/SPM_fulltrain.gif",fps=30)


function percent_differences(eval, evalagainst, normtype)
    initial_difference = 100 * norm(eval[1, :] .- evalagainst[:, 1], normtype) / norm(evalagainst[:, 1], normtype)
    end_difference = 100 * norm(eval[end, :] .- evalagainst[:, end], normtype) / norm(evalagainst[:, end], normtype)
    full_difference = 100 * norm(eval' .- evalagainst, normtype) / norm(evalagainst, normtype)
    (initial_difference, end_difference, full_difference)
end

results = [percent_differences(c_s_n_evals_pretrained, sols_c_s_n, 2),
percent_differences(c_s_n_evals_pretrained_pinn, sols_c_s_n, 2),
percent_differences(c_s_n_evals_quadrature, sols_c_s_n, 2),
percent_differences(c_s_n_evals_pretrained, sols_c_s_n, Inf),
percent_differences(c_s_n_evals_pretrained_pinn, sols_c_s_n, Inf),
percent_differences(c_s_n_evals_quadrature, sols_c_s_n, Inf),
percent_differences(c_s_p_evals_pretrained, sols_c_s_p, 2),
percent_differences(c_s_p_evals_pretrained_pinn, sols_c_s_p, 2),
percent_differences(c_s_p_evals_quadrature, sols_c_s_p, 2),
percent_differences(c_s_p_evals_pretrained, sols_c_s_p, Inf),
percent_differences(c_s_p_evals_pretrained_pinn, sols_c_s_p, Inf),
percent_differences(c_s_p_evals_quadrature, sols_c_s_p, Inf)]

#prob2 = remake(prob, u0=params)
#res2 = GalacticOptim.solve(prob2, Optim.LBFGS(;m=20); allow_f_increases=true, cb = cb,  maxiters=5_000)