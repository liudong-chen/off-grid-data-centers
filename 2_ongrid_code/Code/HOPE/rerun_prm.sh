#!/bin/bash
# Re-derive the planning reserve margin (LOLE 0.1 day/yr fixed point) for the shipped cases.
# Run from Code/HOPE:   bash rerun_prm.sh 2>&1 | tee rerun_prm.log
# Arguments: case folder, Monte Carlo cache tag (Code/HOPE/mc_cache_<tag>_*.csv), replications, max iterations.
# The paper's solves were done on *_u627_case clones with this script's settings (see rerun_u627.sh and rerun_u627.log);
# the resulting output/ and ReliabilityPRM_history.json were copied to the case names below.
set -e
cd "$(dirname "$0")"
M=ModelCases
julia --project=. ReliabilityPRM.jl    $M/TX8z_GTEP_base_vre_case   base     96  20
julia --project=. ReliabilityPRM.jl    $M/TX8z_GTEP_dc_vre_case     dc       96  20
julia --project=. ReliabilityPRM.jl    $M/TX8z_GTEP_base2035_case   base2035 240 20
julia --project=. ReliabilityPRM.jl    $M/TX8z_GTEP_dc20_case       dc20     240 20
julia --project=. ReliabilityPRM.jl    $M/TX8z_GTEP_dc120_case      dc120    240 20
julia --project=. ReliabilityPRM_GA.jl $M/GA1z_GTEP_base_case       gabase   240 20
julia --project=. ReliabilityPRM_GA.jl $M/GA1z_GTEP_dc_case         gadc     240 20
echo DONE
