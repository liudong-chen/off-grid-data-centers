# Off-grid Speed to Power for Data Centers

Code and data accompanying the study *Off-grid speed to power for data centers
comes with costly overbuilding and fossil-fuel dependence*.

## Overview

This project compares grid-connected, off-grid and hybrid electricity supply for
data centers under a common reliability criterion of 0.1 loss-of-load days per
year. The analyses cover ERCOT in 2030 and 2035, Georgia Power, and sensitivities
to demand, workload composition, generation technology and interconnection timing.

The framework combines generation and transmission expansion planning with
[HOPE](https://github.com/HOPE-Model-Project/HOPE.jl), mixed-integer optimization
of off-grid generation, sequential Monte Carlo reliability assessment, and
economic evaluation of early electricity access.

## Repository structure

| Directory | Contents |
|---|---|
| `1_source_data/` | Source datasets, preprocessing scripts, demand profiles, renewable capacity factors, fleet data and fuel prices |
| `2_ongrid_code/` | HOPE source, case builders, reliability routines, model inputs and grid results |
| `3_offgrid_code/` | Off-grid and hybrid models, scenario scripts and results |
| `4_results/` | Figure inputs, plotting scripts, numerical panel data and figures |
| `tools/` | Project assembly, figure generation and file integrity utilities |

`MANIFEST.csv` records file locations, paths in the assembled project, file
sizes and SHA-256 checksums. Base-case inputs are included. Derived-case
builders generate the remaining inputs from the supplied designs and base cases.

## Requirements

- **Python:** 3.10 or later; Python 3.12 for the figure environment specified in
  `requirements-demo.txt`.
- **Python dependencies:** listed in `requirements.txt`. Off-grid optimization
  requires SciPy 1.9 or later.
- **Julia:** 1.12.6, with dependencies specified in
  `2_ongrid_code/Code/HOPE/Project.toml` and `Manifest.toml`.
- **Solver:** Gurobi and a valid licence for the grid optimization runs.

Figure generation from the supplied results requires only the Python figure
environment. It does not require Julia, Gurobi or data-download credentials.

## Installation

Run the following commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
```

For preprocessing and model calculations, install the additional dependencies:

```bash
python -m pip install -r requirements.txt
```

Assemble the source, inputs and results into a separate project directory:

```bash
python tools/reproduction/assemble_workspace.py ../byong-workspace
cd ../byong-workspace
export BYONG_ROOT="$PWD"
julia --project=Code/HOPE -e 'using Pkg; Pkg.instantiate()'
```

The destination must be an empty directory outside the repository. Model commands
below are run from the assembled project root unless stated otherwise.

## Usage

### Main figures

From the repository root, generate Figures 1–5 using the supplied solved inputs:

```bash
python tools/run_demo.py ../byong-figures
```

The output directory must not already exist. It will contain PDF and PNG
figures, numerical panel data, execution logs and `demo_report.json`. The script
compares 14 CSV/JSON outputs against the supplied references at relative and
absolute tolerance 1e-10. Reference runtime is approximately 17 seconds with
Python 3.12 on macOS with Apple Silicon, excluding installation.

Each main figure directory follows the same structure:

```text
01_raw/          Figure inputs
02_code/         Analysis and plotting scripts
03_panel_data/   Numerical panel data
04_figure/       Generated figures
```

| Figures | Script in the assembled project |
|---|---|
| Main Figures 1–3 | `Figures/fig123_package/02_code/73_fig123.py` |
| Main Figure 4 | `Figures/fig4_package/02_code/79_fig4_new.py` |
| Main Figure 5 | `Figures/fig5_package/02_code/96_fig5_new.py` |
| Supplementary Figure 1 | `Figures/figS1_package/02_code/54_fig1_fleet.py` |
| Supplementary Figures 1–5 | `Figures/SI/scripts/make_si_figs.py` |

### Grid optimization

Run the supplied cases with the reliability-based reserve-margin iteration:

```bash
bash Code/HOPE/rerun_prm.sh
```

`ReliabilityPRM.jl` handles ERCOT cases; `ReliabilityPRM_GA.jl` handles Georgia
Power. Case inputs and solver settings are in `Code/HOPE/ModelCases/`.

To compare the main-case builder's calculations with the supplied inputs:

```bash
python Code/HOPE/ModelCases/build_main_cases.py --check
```

This command reports differences without writing case files. Derived cases
must be generated with their corresponding builders before rerunning HOPE.

### Off-grid optimization

Set the input paths and run the gas-only campus and pooled designs:

```bash
export BYONG_CASE="$BYONG_ROOT/Code/HOPE/ModelCases/TX8z_GTEP_dc_vre_case/Data_TX8z_GTEP_dc_vre"
export BYONG_LOAD="$BYONG_ROOT/DC_Load/load_profiles"
export OMP_NUM_THREADS=1
export BYONG_SENS=1
python Code/OffGrid/run_offgrid_mip_gasonly.py all gasonly
python Code/OffGrid/run_offgrid_mip_gasonly.py pool gasonly
```

The gas-only results define the main off-grid comparison. Renewable generation
and storage are evaluated in separate sensitivities. Existing model checks can
be run with:

```bash
python Code/OffGrid/OffGridBYONG.py
```

### Hybrid supply

For the ERCOT 2030 diesel-bridge cases:

```bash
export BYONG_DESIGN="$BYONG_ROOT/Code/OffGrid/out/sensitivity_with_vre/offgrid_mip_final.json"
python Code/OffGrid/run_hybrid_diesel_first.py all
python Code/OffGrid/finalize_hybrid_diesel.py
python Code/HOPE/ModelCases/build_hybrid_diesel_cases.py
julia --project=Code/HOPE Code/HOPE/run_hybrid_diesel_cases.jl
python Code/OffGrid/summarize_hybrid_diesel.py
python Code/OffGrid/tools/connected_util.py
```

The case builder combines the campus generation designs with the grid inputs.
The summary scripts read the resulting HOPE objectives and dispatch outputs.

### Supplementary figures

Generate Supplementary Figure 1 before running the complete supplementary
figure script:

```bash
python Figures/figS1_package/02_code/54_fig1_fleet.py
python Figures/SI/scripts/make_si_figs.py
```

## Scenarios and sensitivities

The following paths are relative to the assembled project.

| Analysis | Entry points and results |
|---|---|
| ERCOT 2035: 20 GW and 121 GW | `scenarios/_code/build_case.py`, `scenarios/_code/91_cf_2035_uplift.py`, `Code/HOPE/ModelCases/75_build_2035_cases.py`; `Figures/fig4_package` |
| Georgia Power | `scenarios/_code/84_georgia_hope_case.py`, `Code/OffGrid/run_offgrid_mip_cases_gasonly.py georgia`; `Figures/fig5_package` |
| Value of speed to power | `Figures/speed_value_sensitivity/` |
| Pure-renewable supply | `Code/OffGrid/scenario_renewable_only/` |
| Renewable hybrid | `Code/HOPE/ModelCases/build_hybrid_renewable_case.py`, `Code/OffGrid/summarize_hybrid_renewable.py` |
| Two generation unit sizes | `Code/OffGrid/scenario_two_unit_sizes/` |
| Workload composition | `Code/OffGrid/scenario_workload_mix/`, `Code/HOPE/run_workload_mix_cases.jl` |
| Off-grid renewables allowed | `Code/OffGrid/run_offgrid_mip.py`; `Code/OffGrid/out/sensitivity_with_vre/` |
| Free siting | `Code/OffGrid/run_offgrid_mip_S1_gasonly.py`; `Code/OffGrid/out/S1_*_gasonly*` |
| Georgia cost decomposition | `Code/OffGrid/run_offgrid_mip_georgia_decomp_gasonly.py` |
| Connection-aware hybrid | `Code/OffGrid/run_hybrid_2phase.py`, `Code/OffGrid/summarize_hybrid_2phase.py`; `Code/OffGrid/out/hybrid_2phase/` |
| Monte Carlo sensitivity cut | `Code/OffGrid/out/sensitivity_no_cut/`, compared with `Code/OffGrid/out/offgrid_mip_gasonly.json` |
| Staged demand | `Code/OffGrid/scenario_ramp/` |
| CO2 emissions | `Code/OffGrid/tools/emissions.py`; `Code/OffGrid/out/emissions.json` |

## Data

Source and processed data are provided in `1_source_data/`. Sources include
public workload traces, ERA5, NSRDB, HRRR, EIA-860/923/930, ERCOT demand and
technology costs, NERC/GADS outage rates, and the Texas2k synthetic network.
The main demand dataset comprises 17 campuses and 8,760 hourly observations for
2025. The manuscript and Supplementary Information describe the assumptions and
source references.

For fresh downloads, set the following environment variables:

| Variable | Used by |
|---|---|
| `NREL_API_KEY`, `NREL_EMAIL` | `download_vre.py` |
| `EIA_API_KEY` | `Code_get_data/main.py` |

File integrity can be verified from the repository root:

```bash
python tools/validate_package.py --deep
```

## Citation

This repository accompanies *Off-grid speed to power for data centers comes
with costly overbuilding and fossil-fuel dependence*.

For the underlying HOPE model, use the citation provided in
[`CITATION.cff`](2_ongrid_code/Code/HOPE/CITATION.cff).

## License

Original project code is distributed under the [MIT License](LICENSE).
HOPE retains its [MIT license](2_ongrid_code/Code/HOPE/LICENSE) and citation.
External datasets and other third-party materials retain their original terms.
