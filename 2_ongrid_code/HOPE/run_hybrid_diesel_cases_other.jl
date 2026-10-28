# Run the Phase-2 (grid-connected) HOPE cases of the diesel-first hybrid for the OTHER scenarios.
#
#   cd 2_ongrid_code/HOPE
#   julia --project=. run_hybrid_diesel_cases_other.jl                 # all three scenarios, shares 30 60 100 (+0)
#   julia --project=. run_hybrid_diesel_cases_other.jl hyb20 30 60 100 # one scenario, chosen shares
#
# Scenario prefixes:  hyb20  -> TX8z_GTEP_hyb20_dsl<XXX>_case   (20 GW, 2035 inputs)
#                     hyb120 -> TX8z_GTEP_hyb120_dsl<XXX>_case  (121 GW, 2035 inputs)
#                     ga     -> GA1z_GTEP_hyb_dsl<XXX>_case     (Georgia Power)
# Cases are built by ModelCases/build_hybrid_diesel_cases_other.py.  dsl000 (the gas-only
# off-grid fleet connecting) is optional: it only cross-checks the scenario's dc case.
using HOPE, Dates

names = Dict("hyb20" => "TX8z_GTEP_hyb20_dsl", "hyb120" => "TX8z_GTEP_hyb120_dsl", "ga" => "GA1z_GTEP_hyb_dsl")
scen = isempty(ARGS) ? ["ga", "hyb20", "hyb120"] : [ARGS[1]]
shares = length(ARGS) > 1 ? parse.(Int, ARGS[2:end]) : [30, 60, 100, 0]
root = @__DIR__
for s in scen, x in shares
    c = names[s] * lpad(x, 3, '0') * "_case"
    dir = joinpath(root, "ModelCases", c)
    isdir(dir) || (println("MISSING ", c, " -- run ModelCases/build_hybrid_diesel_cases_other.py"); continue)
    meta = joinpath(dir, "output", "postprocess_snapshot", "metadata.yml")
    if isfile(meta)
        println("skip ", c, " (already solved)"); continue
    end
    println("\n==== running ", c, "  ", Dates.now())
    t = time()
    HOPE.run_hope("ModelCases/$c/")
    println("==== done ", c, " in ", round((time() - t)/60, digits = 1), " min")
end
println("\nnow:  cd ../../3_offgrid_code && python3 summarize_hybrid_diesel_other.py <2035_k0p5|2035_k3|georgia>")
