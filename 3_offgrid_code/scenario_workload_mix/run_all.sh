#!/bin/bash
# Off-grid designs (gas only, sensitivity cut, as the main case) for the three workload mixes.
#   bash run_all.sh   (from this folder; ~30 min per mix for the 17 sites, run in parallel; then the pooled plant)
H=$(cd "$(dirname "$0")" && pwd); B=$(cd "$H/../.." && pwd)
export OFFGRID_CASE=$B/2_ongrid_code/HOPE/ModelCases/TX8z_GTEP_dc_vre_case/Data_TX8z_GTEP_dc_vre OFFGRID_SENS=1 OMP_NUM_THREADS=1 OFFGRID_POOL_ZONES=FarWest
cd $B/3_offgrid_code
for m in training inference colocation; do
  mkdir -p $H/out/$m
  ( OFFGRID_LOAD=$H/profiles/$m OFFGRID_OUT=$H/out/$m python3 run_offgrid_mip_gasonly.py all gasonly > $H/out/$m/log_sites.txt 2>&1
    OFFGRID_LOAD=$H/profiles/$m OFFGRID_OUT=$H/out/$m python3 run_offgrid_mip_gasonly.py pool gasonly > $H/out/$m/log_pool.txt 2>&1
    echo done > $H/out/$m/DONE ) &
done
wait
