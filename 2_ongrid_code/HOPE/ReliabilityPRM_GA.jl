#!/usr/bin/env julia
# =============================================================================
#  ReliabilityPRM_GA - the planning reserve margin for the Georgia Power case.
#
#  The reliability model is unchanged: this file includes ReliabilityPRM.jl, which
#  includes ReliabilityPRM_mc.jl, and drives the same fixed point
#
#      solve HOPE at PRM_k  ->  read the portfolio it built
#                           ->  score its true LOLE (sequential Monte Carlo)
#                           ->  bisect PRM until LOLE = 0.1 days/year
#
#  What is Georgian is the DATA, and three things about it need saying out loud before
#  anybody quotes the number this produces.
#
#  1. THE MARGIN IN THE SHIPPED CASE IS NOT A DERIVED FIXED POINT.  It is Georgia
#     Power's own 26% WINTER target reserve margin, set by an economic reliability
#     study, not by a loss-of-load criterion.  ERCOT's, in this study, is derived from
#     0.1 days/year.  They are different objects.  Deriving Georgia's own fixed point
#     makes it comparable with ERCOT; running Georgia at 26% runs it to its own
#     standard.  BOTH belong in the paper, and `compare_targets` below prints them
#     side by side.
#
#  2. ONE WEATHER YEAR.  The ERCOT cache carries eight, 2018-25, and the Monte Carlo
#     rotates through them, so its LOLE contains weather risk as well as machine risk.
#     Only 2025 was available for SOCO, so every replication here reuses the same year
#     and the scorer sees ONLY unit outages.  That understates LOLE, and it understates
#     it exactly where Georgia is exposed: the event Georgia Power plans for is an
#     extreme winter morning of the Elliott / January-2024 kind, which a single normal
#     year does not contain.  READ THE DERIVED PRM AS A LOWER BOUND.  Adding more
#     EIA-930 BALANCE files and re-running 86_georgia_mc_cache.py fixes it; the engine
#     rotates through however many years the cache holds.
#
#  3. THE BINDING SEASON MOVES.  In the 2025 data the highest SOCO hour is 22 January
#     09:00 (46,490 MW) and the second is 28 July 17:00 (46,389 MW) - a 0.2% gap - and
#     adding 12.4 GW of data centres tips the binding hour into June.  That is why the
#     solar capacity credit in the shipped case is DERIVED from each case's own top-20
#     load hours (0.63 base, 0.74 dc) rather than taken as Georgia Power's stated 5%
#     winter figure.  `--cc 0.05` in 84_georgia_hope_case.py writes the utility's
#     convention instead.  The PRM should be derived under whichever credit the run
#     uses, because the margin and the credits are a matched pair.
#
#  ---- from the Julia REPL ---------------------------------------------------
#
#      julia> include("ReliabilityPRM_GA.jl")
#      julia> check_ga_cache()                         # pre-flight, no solves
#      julia> h = reliability_prm_ga("ModelCases/GA1z_GTEP_dc_case", "gadc")
#      julia> compare_targets("ModelCases/GA1z_GTEP_dc_case", "gadc")
#
#  ---- or from a shell -------------------------------------------------------
#
#      julia --project=. ReliabilityPRM_GA.jl ModelCases/GA1z_GTEP_dc_case gadc 240 20
#
#  The three cache files mc_cache_gabase_*.csv and mc_cache_gadc_*.csv must sit beside
#  this file, as the ERCOT ones do.  86_georgia_mc_cache.py writes them.
# =============================================================================

include(joinpath(@__DIR__, "ReliabilityPRM.jl"))

const GA_TAGS = ("gabase", "gadc")
const GA_TRM_WINTER = 0.26      # Georgia Power's own target, 2023 IRP Update

"""
    check_ga_cache(; dir = @__DIR__)

Pre-flight. Confirms the three cache files exist for both Georgia tags, reports how many
weather years each carries, and shouts if there is only one. Runs no solves.
"""
function check_ga_cache(; dir = @__DIR__)
    ok = true
    for tag in GA_TAGS
        for part in ("fleet", "series", "meta")
            f = joinpath(dir, "mc_cache_$(tag)_$(part).csv")
            if !isfile(f)
                @error "missing cache file" file = f
                ok = false
            end
        end
        ok || continue
        m = CSV.read(joinpath(dir, "mc_cache_$(tag)_meta.csv"), DataFrame)
        f = CSV.read(joinpath(dir, "mc_cache_$(tag)_fleet.csv"), DataFrame)
        ny = Int(m[1, :n_years])
        firm = sum(Float64.(f.mw))
        @printf("%-8s peak %9.0f MW   firm %9.0f MW (%.3fx)   %d unit(s) over %d row(s)   %d weather year(s)\n",
                tag, m[1, :peak_MW], firm, firm/m[1, :peak_MW],
                sum(Int.(f.n_units)), nrow(f), ny)
        if ny < 2
            @warn """ONE WEATHER YEAR in $tag. The Monte Carlo will see unit-outage risk
                     but not weather risk, so the PRM it derives is a LOWER BOUND. Add
                     more EIA-930 BALANCE files and re-run 86_georgia_mc_cache.py."""
        end
    end
    return ok
end

"""
    ga_fleet_matches_case(case_dir, tag; dir = @__DIR__)

Guard against the quiet failure mode: the case's `gendata.csv` is edited but the Monte
Carlo cache is not, so HOPE and the scorer are describing different systems. Compares
total firm MW in the two, ignoring VRE, and returns the two numbers with a warning if
they disagree by more than 1%.
"""
function ga_fleet_matches_case(case_dir, tag; dir = @__DIR__)
    g = CSV.read(joinpath(datadir_of(case_dir), "gendata.csv"), DataFrame)
    firm_case = sum(Float64(g[i, "Pmax (MW)"]) for i in 1:nrow(g)
                    if Int(g[i, "Flag_VRE"]) == 0)
    f = CSV.read(joinpath(dir, "mc_cache_$(tag)_fleet.csv"), DataFrame)
    firm_cache = sum(Float64.(f.mw))
    rel = abs(firm_case - firm_cache)/max(firm_case, 1.0)
    if rel > 0.01
        @warn "the case and the Monte Carlo cache do not describe the same fleet" firm_case firm_cache rel
    else
        @info "case and cache agree on firm capacity" firm_case firm_cache
    end
    return (case = firm_case, cache = firm_cache, rel = rel)
end

"""
    reliability_prm_ga(case_dir, tag; reps = 240, maxiter = 20, target = 0.1, kwargs...)

The Georgia fixed point. Identical machinery to `reliability_prm`; this wrapper checks
the cache first, checks that the cache and the case agree on the fleet, and states the
single-weather-year caveat before it starts, so the caveat travels with the number.

`tag` is "gabase" or "gadc".
"""
function reliability_prm_ga(case_dir, tag;
                            reps = 240, maxiter = 20, target = 0.1, kwargs...)
    tag in GA_TAGS || error("tag must be one of $(GA_TAGS), got \"$tag\"")
    isdir(case_dir) || error("no case directory at $case_dir")
    check_ga_cache()
    ga_fleet_matches_case(case_dir, tag)
    @printf("\nGeorgia Power PRM fixed point, target LOLE %.2f d/yr.\n", target)
    @printf("Shipped margin in the case is Georgia Power's own %.1f%% winter target;\n",
            100 * GA_TRM_WINTER)
    @printf("what follows derives the LOLE-consistent margin instead, which is the one\n")
    @printf("comparable with ERCOT. Report both.\n")
    return reliability_prm(case_dir, tag; reps = reps, maxiter = maxiter,
                           target = target, kwargs...)
end

"""
    compare_targets(case_dir, tag; reps = 240)

What Georgia's own 26% winter target actually delivers, against the LOLE-consistent
margin. Solves HOPE twice and scores both portfolios; leaves `single_parameter.csv` as it
found it. This is the table the paper needs: the two standards are different objects and
the difference between them is a result, not an error to be removed.
"""
function compare_targets(case_dir, tag; reps = 240, target = 0.1)
    tag in GA_TAGS || error("tag must be one of $(GA_TAGS)")
    prm0 = get_prm(case_dir)
    out = Dict{String,Any}()
    try
        set_prm!(case_dir, GA_TRM_WINTER)
        res = HOPE.run_hope(case_dir)
        g, s = built_from_model(res, case_dir)
        r = score(g, s, tag; reps = reps, case_dir = case_dir)
        am = accredited_margin(res, case_dir, r["peak_MW"])
        out["gp_target"] = Dict("PRM" => GA_TRM_WINTER, "LOLE" => r["LOLE_days"],
                                "built_MW" => sum(Float64(x["mw"]) for x in g; init = 0.0),
                                "accredited_margin" => am.margin, "mc" => r)
        @printf("\nGeorgia Power's own %.1f%% winter target -> LOLE %.3f d/yr, built %.0f MW\n",
                100 * GA_TRM_WINTER, r["LOLE_days"],
                sum(Float64(x["mw"]) for x in g; init = 0.0))

        h = reliability_prm_ga(case_dir, tag; reps = reps, target = target)
        out["derived"] = Dict("status" => h.status, "PRM" => h.PRM,
                              "history" => h.history)
    finally
        set_prm!(case_dir, prm0)
        @printf("single_parameter.csv restored to %.4f%%\n", 100 * prm0)
    end
    f = joinpath(case_dir, "ReliabilityPRM_GA_compare.json")
    open(f, "w") do io
        JSON.print(io, out, 1)
    end
    @printf("wrote %s\n", f)
    return out
end

# ---------------------------------------------------------------- shell entry
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 ||
        error("usage: julia --project=. ReliabilityPRM_GA.jl <case dir> <gabase|gadc> [reps] [maxiter]")
    reliability_prm_ga(ARGS[1], ARGS[2];
                       reps    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 240,
                       maxiter = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 20)
end
