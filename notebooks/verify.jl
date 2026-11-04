#!/usr/bin/env julia
# Run from any directory with: julia /path/to/notebooks/verify.jl
# This script instantiates the notebook's embedded environment in a temporary directory.
using Pkg
using Test

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

notebook_path = joinpath(@__DIR__, "sciml_self_study.jl")
notebook_source = read(notebook_path, String)
original_project = Base.active_project()

environment = mktempdir(; cleanup = true)
for name in ("Project", "Manifest")
    pattern = Regex("PLUTO_" * uppercase(name) * "_TOML_CONTENTS = \"\"\"\\n(.*?)\"\"\"", "s")
    found = match(pattern, notebook_source)
    isnothing(found) && error("Missing embedded $name.toml")
    write(joinpath(environment, "$name.toml"), found.captures[1])
end
Pkg.activate(environment)
Pkg.instantiate()
include(notebook_path)

@testset "Pluto file structure" begin
    definitions = [m.captures[1] for m in eachmatch(r"^# ╔═╡ ([0-9a-f-]{36})$"m, notebook_source)]
    order = [m.captures[1] for m in eachmatch(r"^# [╠╟][═─]([0-9a-f-]{36})$"m, notebook_source)]
    @test length(definitions) == length(unique(definitions))
    @test definitions == order
end

@testset "ODE accuracy, output and limiting cases" begin
    @test SciML.successful_retcode(growth_solution)
    @test length(growth_solution.u) == 121
    @test growth_solution.u[1] == growth_B0
    @test growth_dense_solution(12.3) ≈ logistic_exact(12.3, growth_problem.p, growth_B0) rtol = 1.0e-6
    @test growth_solution(12.3) ≈ logistic_exact(12.3, growth_problem.p, growth_B0) rtol = 1.0e-3
    @test accuracy_result.max_error < 1.0e-4
    for r in (0.05, 0.35), K in (40.0, 140.0), B0 in (1.0, 10.0)
        grid = collect(0.0:1.0:60.0)
        problem = SciML.remake(growth_problem; p = [r, K], u0 = B0)
        sol = SciML.solve(problem, ODE.Tsit5(); saveat = grid, reltol = 1.0e-9, abstol = 1.0e-11)
        @test SciML.successful_retcode(sol)
        @test maximum(abs.(sol.u .- logistic_exact.(grid, Ref([r, K]), B0))) < 1.0e-5
        @test all(B -> B0 - 1.0e-8 <= B <= K + 1.0e-6, sol.u)
    end
    grid = collect(0.0:0.5:60.0)
    loose = SciML.solve(growth_problem, ODE.Tsit5(); saveat = grid, reltol = 1.0e-3, abstol = 1.0e-5)
    tight = SciML.solve(growth_problem, ODE.Tsit5(); saveat = grid, reltol = 1.0e-9, abstol = 1.0e-11)
    truth = logistic_exact.(grid, Ref(growth_problem.p), growth_B0)
    @test maximum(abs.(tight.u .- truth)) < maximum(abs.(loose.u .- truth))
end

@testset "Resource balance and event time" begin
    @test SciML.successful_retcode(resource_solution)
    @test size(Array(resource_solution), 1) == 2
    @test resource_balance_error < 1.0e-8
    @test minimum(Array(resource_solution)) > -1.0e-7
    @test SciML.successful_retcode(target_solution)
    @test target_solution.retcode == SciML.ReturnCode.Terminated
    @test target_result.mass ≈ 40.0 atol = 1.0e-6
    @test target_result.day ≈ log(39.0) / 0.18 atol = 1.0e-5
end

@testset "Roots and optimization" begin
    @test maximum(abs, equilibrium_results.residuals) < 1.0e-8
    @test equilibrium_results.low ≈ 40 * (1 - sqrt(0.5)) atol = 1.0e-7
    @test equilibrium_results.high ≈ 40 * (1 + sqrt(0.5)) atol = 1.0e-7
    @test equilibrium_results.slopes[1] > 0 > equilibrium_results.slopes[2]
    @test all(SciML.successful_retcode, equilibrium_results.retcodes)
    @test SciML.successful_retcode(bowl_solution)
    @test bowl_solution.u ≈ [2.0, -1.0] atol = 1.0e-6
    @test bowl_solution.objective < 1.0e-12
end

@testset "Parameter fitting and time holdout" begin
    @test maximum(fit_data.times) == 30.0
    @test minimum(learning_data.times[learning_data.holdout]) == 32.0
    @test isempty(intersect(learning_data.train, learning_data.holdout))
    @test length(learning_data.holdout) == 10
    @test SciML.successful_retcode(fit_solution)
    @test fit_report.fitted_training_mse < fit_report.initial_training_mse / 20
    @test abs(fitted_parameters[1] - learning_data.truth_p[1]) < 0.02
    @test abs(fitted_parameters[2] - learning_data.truth_p[2]) < 10.0
    @test fit_report.holdout_rmse < 5.0
    # Recover known parameters with zero observation noise, from two distinct starts.
    noiseless = (
        times = fit_data.times,
        observed = logistic_exact.(fit_data.times, Ref(learning_data.truth_p), 2.0),
    )
    for start in ([0.08, 60.0], [0.25, 120.0])
        problem = OPT.OptimizationProblem(fit_function, log.(start), noiseless)
        result = SciML.solve(problem, OptimJL.BFGS(); maxiters = 300, g_tol = 1.0e-7)
        @test exp.(result.u) ≈ learning_data.truth_p rtol = 1.0e-4
        @test growth_loss(result.u, noiseless) < 1.0e-8
    end
end

@testset "Derivatives and identifiability diagnostic" begin
    @test sensitivity_report.ad ≈ sensitivity_report.analytical rtol = 1.0e-5
    @test sensitivity_report.ad ≈ sensitivity_report.finite_difference rtol = 1.0e-5
    @test sensitivity_report.ad > 0
    q = log.([0.15, 85.0])
    gradient_ad = AD.gradient(x -> growth_loss(x, fit_data), q)
    gradient_fd = map(eachindex(q)) do i
        step = zeros(length(q))
        step[i] = 1.0e-5
        (growth_loss(q + step, fit_data) - growth_loss(q - step, fit_data)) / 2.0e-5
    end
    @test gradient_ad ≈ gradient_fd rtol = 1.0e-4
    @test identifiability_report.early_ratio < identifiability_report.full_ratio
end

@testset "Learned correction" begin
    @test SciML.successful_retcode(hybrid_fit)
    @test hybrid_report.fitted_mse < hybrid_report.initial_mse * 1.0e-5
    @test hybrid_report.learned_a ≈ correction_truth.a rtol = 0.01
    @test hybrid_report.learned_b ≈ correction_truth.b rtol = 0.01
    @test hybrid_report.holdout_rmse < 1.0e-3
    @test hybrid_rhs(0.0, hybrid_fit.u, 0.0) == 0.0
    @test all(x -> correction_term(x, hybrid_fit.u) <= 0, 0.0:0.1:2.0)
end

println("\nGrowth fit: ", fit_report)
println("Sensitivity: ", sensitivity_report)
println("Learned correction: ", hybrid_report)
println("\nAll notebook checks passed on Julia ", VERSION)
!isnothing(original_project) && Pkg.activate(dirname(original_project))
