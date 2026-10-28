#!/usr/bin/env julia
# =============================================================================
#  ReliabilityPRM - derive the planning reserve margin from a reliability target,
#  with HOPE in the loop.
#
#      solve HOPE at PRM_k  ->  read the portfolio it built
#                           ->  score its true LOLE (sequential Monte Carlo)
#                           ->  bisect PRM until LOLE = 0.1 days/year
#
#  The reserve margin and the capacity credits are a matched pair. A margin
#  derived by adding firm capacity is only correct if the marginal resource is
#  firm; HOPE builds whatever is cheapest, so the margin has to be re-derived
#  against the portfolio HOPE actually chooses.
#
#  ---- from the Julia REPL (you are already in the HOPE environment) ----------
#
#      julia> include("ReliabilityPRM.jl")
#      julia> h = reliability_prm("ModelCases/TX8z_GTEP_dc_case", "dc"; reps = 240)
#      julia> h.status, h.PRM
#
#  On convergence the margin is written into the case's single_parameter.csv, so
#  the very next HOPE solve is the on-grid solution AT that margin:
#
#      julia> res = HOPE.run_hope("ModelCases/TX8z_GTEP_dc_case")
#
#  Pure Julia - no Python, no numpy, nothing to configure. The reliability model is
#  ReliabilityPRM_mc.jl, which reads mc_cache_<case>_*.csv from beside this file.
#
#  Handy while iterating - these do not run HOPE:
#
#      julia> res = HOPE.run_hope("ModelCases/TX8z_GTEP_dc_case")   # one solve
#      julia> g, s = built_from_model(res, "ModelCases/TX8z_GTEP_dc_case")
#      julia> score(g, s, "dc"; reps = 96)                          # LOLE only
#      julia> set_prm!("ModelCases/TX8z_GTEP_dc_case", 0.1306)      # just edit the margin
#
#  Re-`include` after editing; every name here is a function, so nothing warns.
#
#  ---- or from a shell -------------------------------------------------------
#
#      julia --project=. ReliabilityPRM.jl ModelCases/TX8z_GTEP_dc_case dc 240 8
#
#  ---- two traps this avoids -------------------------------------------------
#    * model[:x] is a 0-1 SCALING FRACTION over G_new, not megawatts.
#      Built MW = Gendata_candidate."Pmax (MW)" .* x.  Same for storage via z.
#    * output/capacity.csv reports the candidate's Pmax CEILING in
#      "Capacity_FIN (MW)" for candidate rows - HOPE never overwrites it with the
#      invested amount - so parsing that file reads build limits as builds. Only
#      its New_Build flag is meaningful. Hence: read the solved model.
# =============================================================================

using CSV, DataFrames, JSON, Printf, JuMP
using HOPE
# Gurobi must be loaded BEFORE reliability_prm starts, otherwise HOPE loads its Gurobi
# extension lazily inside the first run_hope call and the MOI methods it defines are
# newer than the running function's world age: value.(model[:x]) then falls through to
# MOI's fallback and dies with GetAttributeNotAllowed{TerminationStatus}.
try
    @eval using Gurobi
catch err
    @warn "Gurobi.jl not loadable here; fine if the case uses another solver" err
end

# the reliability model: outage chains, storage dispatch, LOLE. Defines datadir_of too.
include(joinpath(@__DIR__, "ReliabilityPRM_mc.jl"))

# ---------------------------------------------------------------- small helpers
paramfile_of(case_dir) = joinpath(datadir_of(case_dir), "single_parameter.csv")

"""    get_prm(case_dir)  ->  Float64
Current planning reserve margin in the case's single_parameter.csv."""
get_prm(case_dir) = Float64(CSV.read(paramfile_of(case_dir), DataFrame)[1, :planning_reserve_margin])

"""    set_prm!(case_dir, prm)
Write a new margin into single_parameter.csv, leaving every other column alone."""
function set_prm!(case_dir, prm)
    f = paramfile_of(case_dir)
    df = CSV.read(f, DataFrame)
    df[1, :planning_reserve_margin] = prm
    CSV.write(f, df)
    return prm
end

# ---------------------------------------------------------------- read the build
"""    built_from_model(res, case_dir)  ->  (genrows, storows)

Built MW per candidate row, straight out of the solved JuMP model that
`HOPE.run_hope` returns. `res` may be the Results Dict or the model itself.
"""
function built_from_model(res, case_dir)
    model = res isa AbstractDict ? res["solved_model"] : res
    d     = datadir_of(case_dir)
    cand  = CSV.read(joinpath(d, "gendata_candidate.csv"), DataFrame)
    scand = CSV.read(joinpath(d, "storagedata_candidate.csv"), DataFrame)

    xv = Base.invokelatest(v -> value.(v), model[:x])   # over G_new, in 0..1
    xs = [xv[g] for g in axes(xv)[1]]            # file order
    length(xs) == nrow(cand) ||
        error("model[:x] has $(length(xs)) entries, gendata_candidate.csv has $(nrow(cand)) rows")

    zv = Base.invokelatest(v -> value.(v), model[:z])
    zs = [zv[s] for s in axes(zv)[1]]
    length(zs) == nrow(scand) ||
        error("model[:z] has $(length(zs)) entries, storagedata_candidate.csv has $(nrow(scand)) rows")

    genrows = Dict{String,Any}[]
    for (i, r) in enumerate(eachrow(cand))
        mw = Float64(r["Pmax (MW)"]) * xs[i]
        mw > 1.0 && push!(genrows, Dict("zone" => String(r["Zone"]),
                                        "type" => String(r["Type"]), "mw" => mw))
    end
    storows = Dict{String,Any}[]
    for (i, r) in enumerate(eachrow(scand))
        p  = Float64(r["Max Power (MW)"])
        mw = p * zs[i]
        hrs = p > 0 ? Float64(r["Capacity (MWh)"]) / p : 4.0
        mw > 1.0 && push!(storows, Dict("zone" => String(r["Zone"]), "mw" => mw,
                                        "hours" => isfinite(hrs) ? hrs : 4.0))
    end
    return genrows, storows
end

"""    ceiling_check(res, case_dir)

Print any candidate pinned at (or within 1% of) its Pmax build ceiling. A binding
ceiling means the limit chose the portfolio, not the economics - check this before
concluding anything from a margin that will not converge.
"""
function ceiling_check(res, case_dir)
    model = res isa AbstractDict ? res["solved_model"] : res
    cand  = CSV.read(joinpath(datadir_of(case_dir), "gendata_candidate.csv"), DataFrame)
    xv = Base.invokelatest(v -> value.(v), model[:x]); xs = [xv[g] for g in axes(xv)[1]]
    hit = [(cand[i, "Type"], cand[i, "Zone"], Float64(cand[i, "Pmax (MW)"]))
           for i in eachindex(xs) if xs[i] > 0.99]
    isempty(hit) ? (@info "no candidate is at its build ceiling") :
        (@warn "candidates pinned at their Pmax ceiling" pinned = hit)
    return hit
end


"""    accredited_margin(res, case_dir)  ->  (accredited_MW, peak_MW, margin)

The reserve margin the fleet actually carries under the case's own capacity credits -
the left-hand side of HOPE's RA constraint. Compare against the PRM in
single_parameter.csv to see whether that constraint binds at all.
"""
function accredited_margin(res, case_dir, peak)
    model = res isa AbstractDict ? res["solved_model"] : res
    d     = datadir_of(case_dir)
    g     = CSV.read(joinpath(d, "gendata.csv"), DataFrame)
    c     = CSV.read(joinpath(d, "gendata_candidate.csv"), DataFrame)
    se    = CSV.read(joinpath(d, "storagedata.csv"), DataFrame)
    sn    = CSV.read(joinpath(d, "storagedata_candidate.csv"), DataFrame)
    xv = Base.invokelatest(v -> value.(v), model[:x]); xs = [xv[i] for i in axes(xv)[1]]
    zv = Base.invokelatest(v -> value.(v), model[:z]); zs = [zv[i] for i in axes(zv)[1]]
    acc  = sum(Float64(g[i, "Pmax (MW)"]) * Float64(g[i, "CC"]) for i in 1:nrow(g))
    acc += sum(Float64(c[i, "Pmax (MW)"]) * Float64(c[i, "CC"]) * xs[i] for i in 1:nrow(c))
    acc += sum(Float64(se[i, "Max Power (MW)"]) * Float64(se[i, "CC"]) for i in 1:nrow(se))
    acc += sum(Float64(sn[i, "Max Power (MW)"]) * Float64(sn[i, "CC"]) * zs[i] for i in 1:nrow(sn))
    return (accredited = acc, peak = peak, margin = acc/peak - 1)
end

# ---------------------------------------------------------------- Monte Carlo
"""    score(genrows, storows, case_tag; reps, case_dir, ...)  ->  Dict

True LOLE of a portfolio: 2030 existing fleet, eight weather years, per-unit outage
chains, greedy storage dispatch. Native Julia - see ReliabilityPRM_mc.jl.

Set `use_python = true` to call the Python scorer instead (needs numpy); the two agree
within Monte Carlo error but not bit-for-bit, because the random streams differ.
"""
function score(genrows, storows, case_tag;
               reps = 240, case_dir = ".", seed = 20260826,
               cache_dir = @__DIR__, use_python = false,
               python = get(ENV, "PYTHON_BIN", "python3"),
               mc_script = get(ENV, "MC_SCRIPT", joinpath(@__DIR__, "11_mc_portfolio.py")),
               build_dir = "", case_root = "")
    if !use_python
        return lole_portfolio(genrows, storows, case_tag;
                              reps = reps, case_dir = case_dir, seed = seed,
                              cache_dir = cache_dir)
    end
    isfile(mc_script) || error("Python scorer not found at $mc_script")
    spec = Dict("case" => case_tag, "reps" => reps, "gen" => genrows, "storage" => storows)
    inf  = joinpath(case_dir, "_portfolio.json")
    outf = joinpath(case_dir, "_lole.json")
    open(inf, "w") do io; JSON.print(io, spec); end
    envs = Dict{String,String}()
    isempty(build_dir) || (envs["BYONG_BUILD"] = build_dir)
    isempty(case_root) || (envs["BYONG_CASE_ROOT"] = case_root)
    cmd = `$python $mc_script $inf $outf`
    run(isempty(envs) ? cmd : setenv(cmd, merge(ENV, envs)))
    return JSON.parsefile(outf)
end

# ---------------------------------------------------------------- the loop
"""
    reliability_prm(case_dir, case_tag; reps=240, maxiter=8, target=0.1, kwargs...)

Solve HOPE, score the portfolio it built, bisect the planning reserve margin, repeat.
Returns the iteration history and leaves the converged margin in single_parameter.csv.

`case_tag` is "base" or "dc" and tells the Monte Carlo which load record to use.
Remaining keyword arguments are forwarded to [`score`](@ref).
"""
function reliability_prm(case_dir, case_tag;
                            reps    = 240,
                            maxiter = 8,
                            target  = 0.1,
                            tol     = 0.02,
                            prm0    = nothing,
                            kwargs...)
    isdir(case_dir) || error("no case directory at $case_dir")
    case_tag in ("base", "dc") || @warn "case_tag=$case_tag is not \"base\" or \"dc\""

    prm_start = get_prm(case_dir)
    prm = prm0 === nothing ? prm_start : Float64(prm0)
    lo, hi = nothing, nothing          # lo fails the target, hi meets it
    history = Dict{String,Any}[]
    status = "no_fixed_point"

    @printf("\nPRM fixed point   case=%s   target LOLE %.2f d/yr   %d replications\n",
            case_tag, target, reps)
    @printf("starting PRM %.4f%%\n\n", 100prm)

    for it in 1:maxiter
        @printf("--- iteration %d :  PRM %.4f%%\n", it, 100prm)
        set_prm!(case_dir, prm)
        res = HOPE.run_hope(case_dir)
        genrows, storows = built_from_model(res, case_dir)
        r = score(genrows, storows, case_tag; reps = reps, case_dir = case_dir, kwargs...)
        lole = Float64(r["LOLE_days"])
        am   = accredited_margin(res, case_dir, r["peak_MW"])

        newmw = sum(Float64(g["mw"]) for g in genrows; init = 0.0) +
                sum(Float64(s["mw"]) for s in storows; init = 0.0)
        mix = Dict{String,Float64}()
        for g in genrows
            mix[g["type"]] = get(mix, g["type"], 0.0) + Float64(g["mw"])
        end
        isempty(storows) || (mix["Storage"] = sum(Float64(s["mw"]) for s in storows))

        @printf("    built %10.0f MW   %s\n", newmw,
                newmw < 1 ? "(nothing)" :
                join([@sprintf("%s %.1f GW", k, v / 1000) for (k, v) in sort(collect(mix))], " · "))
        @printf("    LOLE  %10.3f d/yr   firm/peak %.3f   accredited margin %.2f%%\n\n",
                lole, r["firm_over_peak"], 100 * am.margin)
        push!(history, Dict("iter" => it, "PRM" => prm, "LOLE" => lole, "built_MW" => newmw,
                            "mix" => mix, "accredited_margin" => am.margin, "mc" => r))

        # The RA constraint is slack: the inherited fleet already meets the criterion and
        # the model has no reason to build. Lowering PRM only loosens it further, so there
        # is no fixed point to find - and that is the answer, not a failure.
        if newmw < 1.0 && lole <= target
            status = "not_binding"
            break
        end
        if abs(lole - target) < tol && it > 1
            status = "converged"
            break
        end
        lole > target ? (lo = prm) : (hi = prm)
        prm = if lo !== nothing && hi !== nothing
            0.5 * (lo + hi)
        elseif lole > target
            prm + max(0.01, 0.02 * log10(max(lole / target, 1.1)))
        else
            prm - 0.01
        end
        it == maxiter && ceiling_check(res, case_dir)
    end

    last = history[end]
    println("=" ^ 78)
    if status == "converged"
        set_prm!(case_dir, last["PRM"])
        @printf("CONVERGED.  PRM* = %.4f%%   (LOLE %.3f d/yr, built %.0f MW)\n",
                100 * last["PRM"], last["LOLE"], last["built_MW"])
        @printf("single_parameter.csv updated to %.4f%%\n", 100 * last["PRM"])
    elseif status == "not_binding"
        set_prm!(case_dir, prm_start)
        @printf("NO FIXED POINT - the reserve-margin constraint never binds.\n")
        @printf("  the fleet carries %.2f%% accredited against %.2f%% required,\n",
                100 * last["accredited_margin"], 100 * last["PRM"])
        @printf("  builds nothing, and measures LOLE %.3f d/yr.\n", last["LOLE"])
        @printf("  Adequacy here is set by the inherited fleet, not by the margin.\n")
        @printf("  single_parameter.csv left at its original %.4f%%.\n", 100 * prm_start)
    else
        set_prm!(case_dir, prm_start)
        @printf("NO FIXED POINT in %d iterations - LOLE is still %.3f at PRM %.2f%%.\n",
                maxiter, last["LOLE"], 100 * last["PRM"])
        @printf("  Raising the margin has stopped buying reliability. Check whether a\n")
        @printf("  candidate is pinned at its Pmax ceiling (warning above); if not,\n")
        @printf("  the capacity credits are wrong at the built penetration.\n")
        @printf("  single_parameter.csv left UNCHANGED at %.4f%%.\n", 100 * prm_start)
    end
    println("=" ^ 78)

    hist_file = joinpath(case_dir, "ReliabilityPRM_history.json")
    open(hist_file, "w") do io
        JSON.print(io, Dict("status" => status, "iterations" => history), 1)
    end
    @printf("wrote %s\n", hist_file)
    return (status = status, history = history,
            PRM = status == "converged" ? last["PRM"] : nothing)
end

# ---------------------------------------------------------------- shell entry
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 ||
        error("usage: julia --project=. ReliabilityPRM.jl <case dir> <base|dc> [reps] [maxiter]")
    reliability_prm(ARGS[1], ARGS[2];
                       reps    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 240,
                       maxiter = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 20)
end

# Kept so anything written against the old name still works.
const run_prm_fixedpoint = reliability_prm
