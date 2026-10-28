# Off-grid speed to power for data centers

Code and data for comparing grid-connected, off-grid and hybrid electricity
supply for data centers. The analyses cover ERCOT in 2030 and 2035, Georgia Power,
and sensitivities to workload, generation technology and interconnection timing.

## Repository structure

| Directory | Contents |
|---|---|
| `1_source_data/` | Source datasets, download scripts, demand profiles, renewable profiles, fleet data and fuel prices |
| `2_ongrid_code/` | HOPE, case inputs, case builders, reliability calculations and grid results |
| `3_offgrid_code/` | Off-grid and hybrid models, scenario scripts and results |
| `4_results/` | Figure code, input data, panel data and figures |

## Installation

The Python calculations require Python 3.10 or later and SciPy 1.9 or later.
Dependencies are listed in `requirements.txt`. Grid optimization uses Julia and
a licensed Gurobi installation; the included Julia environment specifies the
versions used in the study.

We tested Python installation and the supplied-data run below on macOS 14.4.1
(Apple silicon, arm64), using Python 3.12.8. The installed Python dependencies were
NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, Matplotlib 3.11.2, Shapely 2.2.0,
pvlib 0.16.1, openpyxl 3.1.5, xlrd 2.0.2, Requests 2.34.2, Numba 0.68.1,
GeoPandas 1.2.0, highspy 1.15.1, Clarabel 0.11.1 and OR-Tools 9.15.6755.
The bundled grid environment records HOPE 1.1.0, Julia 1.12.6, Gurobi.jl 1.9.2
and Gurobi 13.0.2; its full dependency versions are in
`2_ongrid_code/HOPE/Manifest.toml`.

No GPU or other non-standard hardware is required. Python environment creation
and dependency installation took approximately 30 seconds on the tested machine
using prebuilt packages. Download speed and the package cache affect installation
time; this measurement excludes Julia and Gurobi installation.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the commands below from the repository root. Source data and solved results
are included; downloading new data is optional.

To run the load-profile calculation on the supplied source data:

```bash
python 1_source_data/DC_Load/build_dc_load.py
```

This reads `1_source_data/DC_Load/raw_data/` and regenerates six hourly load CSVs,
`site_parameters_and_results.csv` and `fleet_summary.json` in
`1_source_data/DC_Load/load_profiles/`. The run took approximately 2 seconds in
the environment above. Expected results include 17 campuses, 222.3 TWh of annual
energy and an ensemble-mean coincident peak of 32,664 MW. These files are already
included for comparison; rerunning the calculation overwrites them.

## Grid optimization

Grid optimization uses [HOPE](https://github.com/HOPE-Model-Project/HOPE.jl).
Refer to HOPE for installation, Gurobi license configuration and standard case
execution. Install the bundled Julia dependencies from the repository root with:

```bash
julia --project=2_ongrid_code/HOPE -e 'using Pkg; Pkg.instantiate()'
```

The supplied case inputs and settings are in `2_ongrid_code/HOPE/ModelCases/`; derived cases are generated
by the builders in that directory. The included HOPE source and Julia environment
preserve the version used for this study.

Running a case directly through HOPE uses the planning reserve margin stored in
its inputs. `ReliabilityPRM.jl` and `ReliabilityPRM_GA.jl` add Monte Carlo
reliability evaluation and repeated HOPE solves to update that margin. They are
needed when recalculating the reliability-based reserve margin, rather than
running a case at its supplied margin.

## Off-grid and hybrid analyses

Run the setup commands below once from the repository root, before running an
analysis in the same terminal. They set the grid-case input folder, load-profile
folder, reference design, use of Monte Carlo sensitivity cuts and number of
numerical computing threads. The Python scripts read these settings by name.

```bash
export OFFGRID_CASE="$PWD/2_ongrid_code/HOPE/ModelCases/TX8z_GTEP_dc_vre_case/Data_TX8z_GTEP_dc_vre"
export OFFGRID_LOAD="$PWD/1_source_data/DC_Load/load_profiles"
export OFFGRID_DESIGN="$PWD/3_offgrid_code/out/sensitivity_with_vre/offgrid_mip_final.json"
export OFFGRID_SENS=1
export OMP_NUM_THREADS=1
```

Each command below is complete and can be copied as written. Run it from the
repository root. Each row is a separate analysis.

| Analysis | Command |
|---|---|
| ERCOT gas-only supply, separate plants for all campuses | `python 3_offgrid_code/run_offgrid_mip_gasonly.py all gasonly` |
| ERCOT gas-only supply, one pooled plant | `python 3_offgrid_code/run_offgrid_mip_gasonly.py pool gasonly` |
| Georgia gas-only supply | `python 3_offgrid_code/run_offgrid_mip_cases_gasonly.py georgia` |
| ERCOT 2035 gas-only supply, 20 GW fleet | `python 3_offgrid_code/run_offgrid_mip_cases_gasonly.py 2035 0.5` |
| ERCOT 2035 gas-only supply, 121 GW fleet | `python 3_offgrid_code/run_offgrid_mip_cases_gasonly.py 2035 3.0` |
| Free-siting gas-only supply | `python 3_offgrid_code/run_free_siting.py` |
| Georgia cost decomposition, all components | `python 3_offgrid_code/run_georgia_decomposition.py` |
| ERCOT off-grid supply with renewables allowed | `python 3_offgrid_code/run_offgrid_mip.py all all` |
| ERCOT diesel-bridge hybrid supply | `python 3_offgrid_code/run_hybrid_diesel_first.py all` |
| Georgia diesel-bridge hybrid supply | `python 3_offgrid_code/run_hybrid_diesel_first_cases.py georgia` |
| ERCOT 2035 diesel-bridge hybrids, 20 GW fleet | `python 3_offgrid_code/run_hybrid_diesel_first_cases.py 2035 0.5` |
| ERCOT 2035 diesel-bridge hybrids, 121 GW fleet | `python 3_offgrid_code/run_hybrid_diesel_first_cases.py 2035 3.0` |
| Connection-aware hybrid design, all campuses and diesel shares | `python 3_offgrid_code/run_hybrid_2phase.py all` |
| Staged demand | `python 3_offgrid_code/scenario_ramp/ramp_hybrid.py` |
| Off-grid and grid emissions | `python 3_offgrid_code/tools/emissions.py` |
| Campus gas utilization after grid connection | `python 3_offgrid_code/tools/connected_util.py` |

Words and numbers after the filename are inputs to that script. For example,
`all` selects every campus, `pool` selects one plant serving the combined load,
and `gasonly` labels the output files. In the 2035 commands, `0.5` and `3.0` are
data-center demand multipliers corresponding to the 20 GW and 121 GW fleets.

For the following scenarios, first change to the indicated directory with `cd`.
The setup variables above contain absolute paths and remain available after
changing directories.

| Directory | Analysis | Command |
|---|---|---|
| `3_offgrid_code/scenario_renewable_only/` | Renewable-only supply, optimized battery duration | `python run_renewable_only.py` |
| `3_offgrid_code/scenario_renewable_only/` | Renewable-only supply, four-hour batteries | `OFFGRID_BATT_HOURS=4 python run_renewable_only.py` |
| `3_offgrid_code/scenario_two_unit_sizes/` | Two generation unit sizes, separate campus plants | `python run_two_unit_sizes.py sites` |
| `3_offgrid_code/scenario_two_unit_sizes/` | Two generation unit sizes, one pooled plant | `python run_two_unit_sizes.py pool` |

For the workload-mix analysis, run these commands from the repository root:

```bash
python 3_offgrid_code/scenario_workload_mix/build_profiles.py
bash 3_offgrid_code/scenario_workload_mix/run_all.sh
python 3_offgrid_code/scenario_workload_mix/build_hope_cases.py
```

These commands generate the workload profiles, run the off-grid designs and
build the grid cases. Solve the generated `TX8z_GTEP_dc_wl_*_case` cases with
HOPE, then collect the grid and off-grid results:

```bash
python 3_offgrid_code/scenario_workload_mix/summarize.py
```

The following files contain functions called by the analysis scripts:

| File | Contents |
|---|---|
| `3_offgrid_code/offgrid_model.py` | Generator specifications, operating-cost calculations and Monte Carlo reliability functions. Running this file directly checks these functions; it does not run an analysis scenario. |
| `3_offgrid_code/offgrid_mip.py` | Optimization routines that select generation capacity. Run the relevant analysis command above to use these routines. |

Analysis scripts write to `3_offgrid_code/out/` or their scenario directories and
may reuse completed results. Hybrid connected-phase results also require the
corresponding HOPE case to be solved. The `finalize_*` and `summarize_*` files
assemble the designs and results for each hybrid analysis.

## Using other input data

For the ERCOT off-grid runs, copy the supplied load-profile directory and replace
`dc_load_hourly_MW.csv` with your hourly demand in MW. Keep the first column as
`timestamp_local_central`, with one column per campus and 8,760 hourly rows in the
same time order as the case's renewable availability data. In
`site_parameters_and_results.csv`, retain the column names and provide matching
`site_id` values and an `ercot_weather_zone` for each campus. The scripts select
campus columns by `site_id`; the `TOTAL_MW` column is not a campus.

Set `OFFGRID_LOAD` to the absolute path of that directory and run the relevant
command above. If changing grid inputs, copy a supplied HOPE case, retain its CSV
and settings structure, and set `OFFGRID_CASE` to its `Data_*` directory. Keep
zone names consistent across the campus metadata, generator data and renewable
profiles. Hybrid runs also need an updated reference design through
`OFFGRID_DESIGN`; the supplied design applies to the supplied campus inputs.
Georgia and 2035 scenario scripts contain study-specific fleet sizes and scaling
assumptions, which must also be updated when changing those scenarios.

## Figures

Figure code and data are in `4_results/`. Each figure directory contains
`01_raw/` for inputs, `02_code/` for scripts, `03_panel_data/` for numerical
panel data where applicable, and `04_figure/` for figures.

| Figures | Directory | Plotting file |
|---|---|---|
| Figures 1–3 | `4_results/main_figures/figures_1_to_3/` | `02_code/plot_figures_1_to_3.py` |
| Figure 4 | `4_results/main_figures/figure_4/` | `02_code/plot_figure_4.py` |
| Figure 5 | `4_results/main_figures/figure_5/` | `02_code/plot_figure_5.py` |
| Supplementary Figure 1 | `4_results/supplementary_figures/figure_1/` | `02_code/plot_supplementary_figure_1.py` |
| Supplementary Figures 1–5 | `4_results/supplementary_figures/` | `scripts/plot_supplementary_figures.py` |
| Speed-to-power sensitivity | `4_results/supplementary_figures/speed_value_sensitivity/` | `run_speed_value_sensitivity.py`, `renewable_offgrid_value.py`, `make_speed_value_si.py` |

Run a plotting file with Python. The supplementary figure script requires
Supplementary Figure 1 to be generated first. The supplied figure inputs allow
plotting without rerunning the optimization models.

## Data and license

Source and processed data are in `1_source_data/`. Source references and model
assumptions are described in the manuscript and Supplementary Information.
To download source data, use the providers' sites and request your own API keys:

| Data | Source | Download script |
|---|---|---|
| Solar resource | [NSRDB](https://nsrdb.nlr.gov/) and [API documentation](https://developer.nlr.gov/docs/solar/nsrdb/) | `1_source_data/download_vre.py` |
| Wind resource | [Wind Toolkit API documentation](https://developer.nlr.gov/docs/wind/) | `1_source_data/download_vre.py` |
| Southern Company hourly demand, 2015–2025 | [EIA Open Data](https://www.eia.gov/opendata/) | `1_source_data/download_eia_demand.py` |

For weather downloads, request a key from the [NLR developer portal](https://developer.nlr.gov/signup/).
Set `NREL_API_KEY` to your key and `NREL_EMAIL` to your contact email. The variable
names follow the former NREL name; NSRDB requires the requesting user's email
in its [download requests](https://developer.nlr.gov/docs/solar/nsrdb/nsrdb-GOES-aggregated-v4-0-0-download/).
For EIA downloads, [register for an EIA key](https://www.eia.gov/opendata/register.php)
and set `EIA_API_KEY`. These credentials are only needed to download new data.

```bash
python 1_source_data/download_vre.py
python 1_source_data/download_eia_demand.py
```

Original project code is distributed under the [MIT License](LICENSE). HOPE
retains its [license](2_ongrid_code/HOPE/LICENSE) and
[citation](2_ongrid_code/HOPE/CITATION.cff). External datasets retain their
original terms.
