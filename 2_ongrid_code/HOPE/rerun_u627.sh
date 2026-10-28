#!/bin/bash
# Re-derive the planning reserve margin with the grid's new gas split into the campus
# unit sizes (NGCC 627 MW, NGCT 419 MW) instead of the candidate file's 57-unit artefact.
# Runs on the *_u627_case clones only; the original cases are untouched.
# Run from 2_ongrid_code/HOPE:   bash rerun_u627.sh 2>&1 | tee rerun_u627.log
set -e
cd "$(dirname "$0")"
M=ModelCases
# ERCOT 2030 (reps 96 as in the originals), 2035 and Georgia (reps 240 as in the originals)
julia --project=. ReliabilityPRM.jl    $M/TX8z_GTEP_base_vre_u627_case  base     96  20
julia --project=. ReliabilityPRM.jl    $M/TX8z_GTEP_dc_vre_u627_case    dc       96  20
julia --project=. ReliabilityPRM.jl    $M/TX8z_GTEP_base2035_u627_case  base2035 240 20
julia --project=. ReliabilityPRM.jl    $M/TX8z_GTEP_dc20_u627_case      dc20     240 20
julia --project=. ReliabilityPRM.jl    $M/TX8z_GTEP_dc120_u627_case     dc120    240 20
julia --project=. ReliabilityPRM_GA.jl $M/GA1z_GTEP_base_u627_case      gabase   240 20
julia --project=. ReliabilityPRM_GA.jl $M/GA1z_GTEP_dc_u627_case        gadc     240 20
echo DONE
