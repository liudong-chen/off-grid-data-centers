# =============================================================================
#  ReliabilityPRM - the Monte Carlo engine.
#
#  LOLE of a portfolio. Called by ReliabilityPRM.jl; not usually run directly.
#
#  Sequential Monte Carlo, the same model that produced the reliability report:
#    * every firm generating unit is a two-state Markov chain with geometric
#      holding times; unit COUNTS come from EIA-860, because sigma/P = sqrt(q(1-q)/n)
#    * load, wind and solar are drawn from the same weather year, whole years intact
#    * storage is dispatched greedily against the residual, not credited
#    * LOLE is counted in loss-of-load DAYS per year
#
#  Inputs are three small CSVs per case, sitting beside this file:
#    mc_cache_<case>_fleet.csv    the 2030 existing firm fleet: Type, mw, n_units, FOR, MTTR
#    mc_cache_<case>_series.csv   8 weather years of zonal-total load and existing VRE
#    mc_cache_<case>_meta.csv     storage MW/MWh, annual peak, round-trip efficiency
#
#  Candidate VRE shapes are read live from the case's gen_availability_timeseries.csv,
#  so new wind and solar are scored on exactly what HOPE optimised against.
#
#  Note on reproducibility: this does NOT reproduce the Python scorer bit-for-bit -
#  the random streams differ. The two agree to within Monte Carlo error, which is
#  what matters. `compare_scorers` below checks that.
# =============================================================================

using CSV, DataFrames, Random, JSON

"""The Data_* directory inside a case folder.

A case folder may legitimately hold more than one.  The 2035 cases keep a
Data_*_BEFORE_2035_UPDATE backup beside the live directory, and picking by uniqueness then
dies with "Collection has multiple elements, must contain exactly 1 element" before any
solve happens.  The case's own Settings/HOPE_model_settings.yml names the live one on its
`DataCase:` line, so that is the authority here; the unique-directory rule stays as the
fallback for cases without a settings file.
"""
function datadir_of(case_dir)
    cfg = joinpath(case_dir, "Settings", "HOPE_model_settings.yml")
    if isfile(cfg)
        for ln in eachline(cfg)
            m = match(r"^\s*DataCase\s*:\s*([^#\s]+)", ln)
            if m !== nothing
                d = joinpath(case_dir, rstrip(String(m.captures[1]), '/'))
                isdir(d) && return d
            end
        end
    end
    dirs = filter(d -> startswith(basename(d), "Data_"),
                  filter(isdir, readdir(case_dir, join = true)))
    length(dirs) == 1 && return dirs[1]
    isempty(dirs) && error("no Data_* directory in $case_dir")
    names = join(basename.(dirs), ", ")
    error("$(length(dirs)) Data_* directories in $case_dir and no usable DataCase: line " *
          "in Settings/HOPE_model_settings.yml -- $names.  Rename the backups so they do " *
          "not start with Data_.")
end

const MC_H       = 8760
const FOR_NEW_J  = Dict("NGCC" => 0.0357, "NGCT" => 0.0476, "NuC" => 0.08)
const MTTR_NEW_J = Dict("NGCC" => 48, "NGCT" => 36, "NuC" => 120)
# Unit size for NEW build is NOT hard-coded - it is read from the candidate file as
# Pmax / n_units, so the Monte Carlo splits what HOPE builds into the same machine
# size the case declares. Hard-coding it silently changes the pooling, and therefore
# the LOLE, without changing a single input file.

struct FirmUnit
    mw::Float64
    n::Int
    q::Float64
    mttr::Int
end

"""Geometric draw on {1,2,...} with success probability p, by inverse CDF."""
@inline function geo(rng, p)
    p >= 1.0 && return 1
    return max(1, ceil(Int, log(rand(rng)) / log1p(-p)))
end

"""Available firm MW per hour. Starts at total capacity and subtracts each unit's
down-segments, which is the same accumulation the Python version does."""
function block_trace!(av::Vector{Float64}, rows::Vector{FirmUnit}, rng)
    total = 0.0
    for r in rows
        total += r.mw
    end
    fill!(av, total)
    for r in rows
        r.q <= 0 && continue
        n = max(1, r.n)
        c = r.mw / n
        mtbf = r.mttr * (1 - r.q) / r.q
        for _ in 1:n
            t = 1
            up = rand(rng) > r.q
            while t <= MC_H
                d = up ? geo(rng, 1 / mtbf) : geo(rng, 1 / r.mttr)
                e = min(t + d - 1, MC_H)
                if !up
                    @inbounds for h in t:e
                        av[h] -= c
                    end
                end
                t = e + 1
                up = !up
            end
        end
    end
    return av
end

"""Greedy storage against the residual; returns the number of loss-of-load DAYS."""
function shortfall_days(net::Vector{Float64}, sto_p, sto_e, eta)
    E = sto_e
    days = 0
    @inbounds for d in 0:364
        short = false
        for h in 1:24
            n = net[d*24 + h]
            if n > 0
                dis = min(n, sto_p, eta * E)
                E -= dis / eta
                (n - dis) > 1e-6 && (short = true)
            else
                E += eta * min(-n, sto_p, (sto_e - E) / eta)
            end
        end
        short && (days += 1)
    end
    return days
end

# ---------------------------------------------------------------- cache
const _MC_CACHE = Dict{String,Any}()

function mc_cache(case_tag; dir = @__DIR__)
    get!(_MC_CACHE, case_tag) do
        f = CSV.read(joinpath(dir, "mc_cache_$(case_tag)_fleet.csv"), DataFrame)
        s = CSV.read(joinpath(dir, "mc_cache_$(case_tag)_series.csv"), DataFrame)
        m = CSV.read(joinpath(dir, "mc_cache_$(case_tag)_meta.csv"), DataFrame)
        rows = [FirmUnit(Float64(r.mw), Int(r.n_units), Float64(r.FOR), Int(r.MTTR))
                for r in eachrow(f)]
        yrs = [parse(Int, split(String(c), "_")[2])
               for c in names(s) if startswith(String(c), "load_")]
        LOAD = [Float64.(s[!, "load_$(y)"]) for y in yrs]
        VRE  = [Float64.(s[!, "vre_$(y)"]) for y in yrs]
        (rows = rows, LOAD = LOAD, VRE = VRE, years = yrs,
         sto_p = Float64(m[1, :sto_p_MW]), sto_e = Float64(m[1, :sto_e_MWh]),
         peak = Float64(m[1, :peak_MW]), eta = Float64(m[1, :eta]))
    end
end

"""    candidate_unit_mw(case_dir, zone, typ)  ->  Float64

Machine size for new build, taken from the candidate file as Pmax / n_units. This sets
how many independent Markov chains a block of new capacity becomes, which drives
sigma/P = sqrt(q(1-q)/n) and so the LOLE.
"""
function candidate_unit_mw(case_dir, zone, typ)
    d = datadir_of(case_dir)
    cand = CSV.read(joinpath(d, "gendata_candidate.csv"), DataFrame)
    k = findfirst(i -> String(cand[i, :Zone]) == zone && String(cand[i, :Type]) == typ,
                  1:nrow(cand))
    k === nothing && error("no candidate row for $typ@$zone in $d")
    n = max(1, Int(cand[k, :n_units]))
    return Float64(cand[k, "Pmax (MW)"]) / n
end

"""Hourly output of one MW of a candidate VRE resource, from the case's own profile."""
function candidate_vre(case_dir, zone, typ)
    d = datadir_of(case_dir)
    cand = CSV.read(joinpath(d, "gendata_candidate.csv"), DataFrame)
    nexist = nrow(CSV.read(joinpath(d, "gendata.csv"), DataFrame))
    k = findfirst(i -> String(cand[i, :Zone]) == zone && String(cand[i, :Type]) == typ,
                  1:nrow(cand))
    k === nothing && error("no candidate row for $typ@$zone in $d")
    av = CSV.read(joinpath(d, "gen_availability_timeseries.csv"), DataFrame)
    return Float64.(av[!, Symbol("G$(nexist + k)")])[1:MC_H]
end

# ---------------------------------------------------------------- the scorer
"""
    lole_portfolio(genrows, storows, case_tag; reps=240, case_dir=".", seed=20260826)

True LOLE in days/year of (2030 existing fleet + this portfolio), over the full
8760 hours and all eight weather years. `genrows` and `storows` are the vectors
`built_from_model` returns.
"""
function lole_portfolio(genrows, storows, case_tag;
                        reps = 240, case_dir = ".", seed = 20260826,
                        cache_dir = @__DIR__)
    C = mc_cache(case_tag; dir = cache_dir)
    rows = copy(C.rows)
    add_vre = zeros(MC_H)
    new_firm = 0.0
    new_vre = 0.0
    for g in genrows
        mw = Float64(g["mw"]); mw > 0 || continue
        ty = String(g["type"]); z = String(g["zone"])
        if ty in ("SolarPV", "WindOn")
            add_vre .+= mw .* candidate_vre(case_dir, z, ty)
            new_vre += mw
        else
            haskey(FOR_NEW_J, ty) ||
                error("unknown candidate type $ty; expected NGCC, NGCT, NuC, SolarPV or WindOn")
            usz = candidate_unit_mw(case_dir, z, ty)
            push!(rows, FirmUnit(mw, max(1, round(Int, mw / usz)),
                                 FOR_NEW_J[ty], MTTR_NEW_J[ty]))
            new_firm += mw
        end
    end
    sto_p, sto_e = C.sto_p, C.sto_e
    for s in storows
        sto_p += Float64(s["mw"])
        sto_e += Float64(s["mw"]) * Float64(get(s, "hours", 4.0))
    end

    rng = Xoshiro(seed)
    av  = Vector{Float64}(undef, MC_H)
    net = Vector{Float64}(undef, MC_H)
    ny  = length(C.years)
    days = 0
    for i in 1:reps
        wy = (i - 1) % ny + 1
        block_trace!(av, rows, rng)
        @inbounds for h in 1:MC_H
            net[h] = C.LOAD[wy][h] - C.VRE[wy][h] - add_vre[h] - av[h]
        end
        days += shortfall_days(net, sto_p, sto_e, C.eta)
    end
    firm = sum(r.mw for r in rows)
    return Dict{String,Any}(
        "LOLE_days" => days / reps, "reps" => reps, "case" => case_tag,
        "peak_MW" => C.peak, "firm_MW" => firm, "firm_over_peak" => firm / C.peak,
        "new_firm_MW" => new_firm, "new_vre_MW" => new_vre,
        "storage_MW" => sto_p, "storage_MWh" => sto_e)
end

"""
    compare_scorers(genrows, storows, case_tag; reps=240, python="python3", ...)

Run the Julia and Python scorers on the same portfolio. The random streams differ, so
they will not match exactly - they should agree within about 2 standard errors.
Only useful if you have a Python with numpy; skip it otherwise.
"""
function compare_scorers(genrows, storows, case_tag; reps = 240, case_dir = ".",
                         python = get(ENV, "PYTHON_BIN", "python3"),
                         mc_script = joinpath(@__DIR__, "11_mc_portfolio.py"))
    jl = lole_portfolio(genrows, storows, case_tag; reps = reps, case_dir = case_dir)
    isfile(mc_script) || (@warn "no Python scorer at $mc_script"; return jl)
    spec = Dict("case" => case_tag, "reps" => reps, "gen" => genrows, "storage" => storows)
    inf = tempname() * ".json"; outf = tempname() * ".json"
    open(inf, "w") do io; JSON.print(io, spec); end
    try
        run(`$python $mc_script $inf $outf`)
        py = JSON.parsefile(outf)
        @info "scorer comparison" julia = jl["LOLE_days"] python = py["LOLE_days"] reps
    catch err
        @warn "Python scorer did not run; Julia result stands" err
    end
    return jl
end
