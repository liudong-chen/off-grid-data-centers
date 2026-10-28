# Run the eleven Phase-2 (grid-connected) HOPE cases of the diesel-first hybrid.
#
#   cd 2_ongrid_code/HOPE
#   julia --project=. run_hybrid_diesel_cases.jl            # x = 0,10,...,100 %, skipping solved ones
#   julia --project=. run_hybrid_diesel_cases.jl 0 20 50 100
#   HYB_PREFIX=hyb2p julia --project=. run_hybrid_diesel_cases.jl 0 30 60 100   # connection-aware fleets
#
# Cases are TX8z_GTEP_hybrid_dsl<XXX>_vre_case, built by ModelCases/build_hybrid_diesel_cases.py
# from 3_offgrid_code/out/hybrid_diesel/all_sites.json (dsl000 = gas-only off-grid fleet).
# Same settings as TX8z_GTEP_hybrid_vre_case (pure LP, Gurobi, full 8760, PRM 0.13044).
# A finished case has output/postprocess_snapshot/metadata.yml with objective_value.
using HOPE, Dates

shares = isempty(ARGS) ? collect(0:10:100) : parse.(Int, ARGS)
root = @__DIR__
for x in shares
    c = "TX8z_GTEP_$(get(ENV, "HYB_PREFIX", "hybrid"))_dsl$(lpad(x, 3, '0'))_vre_case"
    dir = joinpath(root, "ModelCases", c)
    isdir(dir) || (println("MISSING ", c, " -- run ModelCases/build_hybrid_diesel_cases.py"); continue)
    meta = joinpath(dir, "output", "postprocess_snapshot", "metadata.yml")
    if isfile(meta)
        println("skip ", c, " (already solved)"); continue
    end
    println("\n==== running ", c, "  ", Dates.now())
    t = time()
    HOPE.run_hope("ModelCases/$c/")
    println("==== done ", c, " in ", round((time() - t)/60, digits = 1), " min")
end
println("\nnow:  cd ../../3_offgrid_code && python3 summarize_hybrid_diesel.py")
