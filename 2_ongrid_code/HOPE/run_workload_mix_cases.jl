# Run the three grid (HOPE) cases of the workload-mix scenario.
#
#   cd 2_ongrid_code/HOPE
#   julia --project=. run_workload_mix_cases.jl                      # all three, skipping solved ones
#   julia --project=. run_workload_mix_cases.jl training             # just one
#
# Cases are TX8z_GTEP_dc_wl_<mix>_case, built by
# 3_offgrid_code/scenario_workload_mix/build_hope_cases.py from profiles/<mix>.
# A finished case has output/postprocess_snapshot/metadata.yml with objective_value.
using HOPE, Dates

mixes = isempty(ARGS) ? ["training", "inference", "colocation"] : ARGS
root = @__DIR__
cd(root)
for m in mixes
    c = "TX8z_GTEP_dc_wl_$(m)_case"
    dir = joinpath(root, "ModelCases", c)
    isdir(dir) || (println("MISSING ", c); continue)
    if isfile(joinpath(dir, "output", "postprocess_snapshot", "metadata.yml"))
        println("skip ", c, " (already solved)"); continue
    end
    println("\n==== running ", c, "  ", Dates.now())
    t = time()
    HOPE.run_hope("ModelCases/$c/")
    println("==== done ", c, " in ", round((time() - t)/60, digits = 1), " min")
end
println("\nnow:  cd ../../3_offgrid_code/scenario_workload_mix && python3 summarize.py")
