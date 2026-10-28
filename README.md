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

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the commands below from the repository root. Source data and solved results
are included; downloading new data is optional.

## Grid optimization

Grid optimization uses [HOPE](https://github.com/HOPE-Model-Project/HOPE.jl).
Refer to HOPE for installation and standard case execution. The supplied case
inputs and settings are in `2_ongrid_code/HOPE/ModelCases/`; derived cases are generated
by the builders in that directory. The included HOPE source and Julia environment
preserve the version used for this study.

Running a case directly through HOPE uses the planning reserve margin stored in
its inputs. `ReliabilityPRM.jl` and `ReliabilityPRM_GA.jl` add Monte Carlo
reliability evaluation and repeated HOPE solves to update that margin. They are
needed when recalculating the reliability-based reserve margin, rather than
running a case at its supplied margin.

## Off-grid and hybrid analyses

The following files are in `3_offgrid_code/`. Set the shared inputs once, then run
an analysis as `python 3_offgrid_code/<file> [arguments]` from the project root:

```bash
export OFFGRID_CASE="$PWD/2_ongrid_code/HOPE/ModelCases/TX8z_GTEP_dc_vre_case/Data_TX8z_GTEP_dc_vre"
export OFFGRID_LOAD="$PWD/1_source_data/DC_Load/load_profiles"
export OFFGRID_DESIGN="$PWD/3_offgrid_code/out/sensitivity_with_vre/offgrid_mip_final.json"
export OFFGRID_SENS=1
export OMP_NUM_THREADS=1
```

| File | Analysis | Arguments, where required |
|---|---|---|
| `offgrid_model.py` | Generator catalogue, operating costs and Monte Carlo reliability functions | Library; direct execution runs its model checks |
| `offgrid_mip.py` | Mixed-integer generation design | Optimizer used by the analysis scripts |
| `run_offgrid_mip_gasonly.py` | ERCOT gas-only off-grid supply | `all gasonly` for campuses; `pool gasonly` for pooled supply |
| `run_offgrid_mip_cases_gasonly.py` | Georgia and ERCOT 2035 gas-only supply | `georgia`, `2035 0.5` or `2035 3.0` |
| `run_free_siting.py` | Free-siting gas-only supply | None |
| `run_georgia_decomposition.py` | Georgia cost decomposition | None for all components; or `G1`, `G2`, `G4` |
| `run_offgrid_mip.py` | Off-grid supply with renewables allowed | Campus selection and output tag |
| `run_hybrid_diesel_first.py` | ERCOT diesel-bridge hybrid supply | `all` or a campus selection |
| `run_hybrid_diesel_first_cases.py` | Georgia and ERCOT 2035 hybrids | Scenario selection |
| `run_hybrid_2phase.py` | Connection-aware hybrid design | See the file's scenario options |
| `scenario_renewable_only/run_renewable_only.py` | Pure-renewable supply | `OFFGRID_BATT_HOURS=4` selects four-hour batteries |
| `scenario_two_unit_sizes/run_two_unit_sizes.py` | Two generation unit sizes | `sites` or `pool` |
| `scenario_workload_mix/` | Training, inference and colocation workloads | Profile builder, case builder and summary scripts |
| `scenario_ramp/ramp_hybrid.py` | Staged demand | None |
| `tools/emissions.py` | Off-grid and grid emissions | None |
| `tools/connected_util.py` | Campus gas utilization after grid connection | None |

Analysis scripts write to `3_offgrid_code/out/` or their scenario directories and
may reuse completed results. Hybrid connected-phase results also require the
corresponding HOPE case to be solved. The `finalize_*` and `summarize_*` files
assemble the designs and results for each hybrid analysis.

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
