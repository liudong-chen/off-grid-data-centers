"""
BYONG — ERCOT data-center load profile generator (final).

Implements the method in the Methods section §"Constructing DC load profile":

  1. u_train(t)  53 bootstrapped Monday-aligned weeks from the 136-week training library
                 (same-owner sites share a week with probability p = 0.5)
  2. u_inf(t)    53 weeks drawn independently from the 105-week inference library
  2b u_colo(t)   53 weeks drawn independently from the 104-week colocation library
  3. u(t)      = w_tr*u_train + w_inf*u_inf + w_colo*u_colo          (archetype mix)
  4. P_IT(t)   = P_GPU * [ s + (1-s) u(t) ],  P_GPU = P_ann / M_des   (M_des = M_0 + b(40-15))
  5. P_cool(t) = P_IT(t) * [ M(t) - 1 ],      M(t)  = M_0 + b*max(T(t)-15, 0)
     P_site(t) = P_IT(t) + P_cool(t)

Inputs  : raw_data/   
Outputs : results/
"""
import os, json, argparse
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(BASE, 'raw_data')
OUT  = os.path.join(BASE, 'load_profiles')
os.makedirs(OUT, exist_ok=True)

# =====================================================================
# PARAMETERS  (see draft.tex Tables "Utilization to power" and "Annual PUE")
# =====================================================================
YEAR      = 2025
N_WEEKS   = 53
N_DRAWS   = 20
SEED      = 20260824
T_REF     = 15.0        # deg C, reference temperature
T_DES     = 40.0        # deg C, design temperature
B_SLOPE   = 0.0044      # per deg C, fitted to Google Midlothian quarterly PUE vs ERA5 hourly
P_SHARED  = 0.50        # same-owner training-week sharing probability (Helios Earth-Venus lambda_0.95)

S_COMPONENT = {'train': 0.151, 'inference': 0.152, 'colo': 0.170}   # static share by component

ARCHETYPE = {                    # w_train, w_inference, w_colo
    'ai_training':      (0.8, 0.2, 0.0),
    'hyperscale_cloud': (0.5, 0.5, 0.0),
    'developer_campus': (0.5, 0.3, 0.2),
    'colo_enterprise':  (0.2, 0.4, 0.4),
}
M0 = {                           # baseline cooling multiplier at T <= T_REF
    'ai_training':      1.068,   # Google Midlothian TX annual 1.10
    'hyperscale_cloud': 1.123,   # AWS 1.15 / Microsoft 1.16 -> annual 1.155
    'developer_campus': 1.123,   # no published benchmark; hyperscale adopted
    'colo_enterprise':  1.408,   # Uptime Institute 2025, facilities >= 20 MW, annual 1.44
}

idx = pd.date_range(f'{YEAR}-01-01', periods=8760, freq='h')
first_monday_offset = (7 - idx[0].dayofweek) % 7


# =====================================================================
# 1. WEEK LIBRARIES
# =====================================================================
def weekly_blocks(s):
    """Full Monday-aligned weeks, each normalised to the trace's own peak."""
    x = s / s.max()
    ws = [g.values for _, g in x.groupby(pd.Grouper(freq='W-MON', label='left', closed='left'))
          if len(g) == 168 and g.notna().all()]
    if not ws:                                   # trace shorter than one aligned week
        how = x.groupby([x.index.dayofweek, x.index.hour]).mean()
        if len(how) == 168:
            ws = [how.values]
    return np.array(ws)


def build_libraries():
    LIB = {'train': {}, 'inference': {}, 'colo': {}}

    # -- training: SenseTime Helios x4 + Shanghai AI Lab Acme x2 (GPU allocated / capacity)
    occ = pd.read_csv(f'{RAW}/trace_helios_acme_gpu_allocated_hourly.csv',
                      index_col=0, parse_dates=True)
    cap  = {'helios_Earth': 1232, 'helios_Saturn': 2104, 'helios_Uranus': 2256,
            'helios_Venus': 1264, 'acme_seren': 2048, 'acme_kalos': 1152}
    warm = {'helios_Earth': 14, 'helios_Saturn': 14, 'helios_Uranus': 10,
            'helios_Venus': 14, 'acme_seren': 12, 'acme_kalos': 2}
    for k in occ:
        s = (occ[k] / cap[k]).clip(upper=1).dropna()
        s = s[(s.index >= s.index.min() + pd.Timedelta(days=warm[k])) &
              (s.index <= s.index.max() - pd.Timedelta(days=12))]
        LIB['train'][k] = weekly_blocks(s)

    # -- training: Microsoft Philly (mean GPU utilisation)
    ph = pd.read_csv(f'{RAW}/trace_philly_gpu_utilisation_hourly.csv', parse_dates=['t']).set_index('t')
    LIB['train']['ms_philly'] = weekly_blocks((ph.gpu_util_mean / 100)[ph.gpu_samples > 2000])

    # -- inference: Azure LLM code / conversation (tokens per hour, US Central)
    for f in ['code', 'conv']:
        a = pd.read_csv(f'{RAW}/trace_azure_llm_{f}_tokens_hourly.csv', index_col=0, parse_dates=True)
        s = a.ctx + a.gen
        s.index = s.index.tz_convert('US/Central').tz_localize(None)
        LIB['inference'][f'az_llm_{f}'] = weekly_blocks(s)

    # -- inference / colocation: NLR DIPLOEE simulated whole-facility utilisation
    for comp, tag in [('inference', 'inference_1MW_283nodes'), ('colo', 'colocation_10MW_2840nodes')]:
        parts = []
        for lvl in ['40u', '60u']:
            d = pd.read_csv(f'{RAW}/nlr_{tag}_{lvl}_power_1min.csv', parse_dates=['timestamp'])
            parts.append(d.set_index('timestamp').utilization.resample('1h').mean())
        v = pd.concat(parts).reset_index(drop=True)
        LIB[comp][f'nlr_{comp}_sim'] = weekly_blocks(
            pd.Series(v.values, index=pd.date_range('2018-01-01', periods=len(v), freq='h')))
    return LIB


# =====================================================================
# 2. WEATHER
# =====================================================================
ZONE_CITY = {'Coast': 'Houston', 'East': 'Tyler', 'North Central': 'Dallas',
             'North': 'Wichita_Falls', 'West': 'Abilene', 'Far West': 'Midland',
             'South Central': 'Austin', 'South': 'Laredo'}


def read_temp(city):
    d = pd.read_csv(f'{RAW}/weather_era5_{city}_2m_temperature_hourly.csv', skiprows=3)
    d.columns = ['time', 'T']
    t = pd.to_datetime(d.time, utc=True).dt.tz_convert('America/Chicago').dt.tz_localize(None)
    s = pd.Series(d['T'].values, index=t)
    s = s[~s.index.duplicated(keep='first')].reindex(idx).bfill().ffill()
    assert s.notna().all() and len(s) == 8760, city
    return s.values


# =====================================================================
# 3. GENERATOR
# =====================================================================
def draw_year(lib, rng, shared=None):
    keys = list(lib); picks, blocks = [], []
    for w in range(N_WEEKS):
        if shared is not None and shared[w] is not None:
            k, j = shared[w]
        else:
            k = keys[rng.integers(len(keys))]; j = rng.integers(len(lib[k]))
        picks.append((k, j)); blocks.append(lib[k][j])
    return np.roll(np.concatenate(blocks), 24 * first_monday_offset)[:8760], picks


def one_draw(sites, LIB, TEMP, rng):
    """Returns P_IT and P_cool, each 8760 x n_sites."""
    it, cool, picks_by_owner = {}, {}, {}
    for _, r in sites.iterrows():
        w_tr, w_inf, w_co = ARCHETYPE[r.archetype]
        s_i  = w_tr * S_COMPONENT['train'] + w_inf * S_COMPONENT['inference'] + w_co * S_COMPONENT['colo']
        m0_i = M0[r.archetype]
        m_des = m0_i + B_SLOPE * (T_DES - T_REF)

        own = r.owner_group
        shared = None
        if picks_by_owner.get(own) is not None:                       # same-owner dependence
            shared = [pk if rng.random() < P_SHARED else None for pk in picks_by_owner[own]]
        u_tr, picks = draw_year(LIB['train'], rng, shared)
        picks_by_owner.setdefault(own, picks)

        u_inf, _ = draw_year(LIB['inference'], rng)
        u_co, _  = draw_year(LIB['colo'], rng)
        u = w_tr * u_tr + w_inf * u_inf + w_co * u_co

        p_gpu = r.mw_published / m_des
        p_it  = p_gpu * (s_i + (1 - s_i) * u)
        M_t   = m0_i + B_SLOPE * np.maximum(TEMP[r.ercot_weather_zone] - T_REF, 0)
        it[r.site_id]   = p_it
        cool[r.site_id] = p_it * (M_t - 1.0)
    return pd.DataFrame(it, index=idx), pd.DataFrame(cool, index=idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=N_DRAWS)
    args = ap.parse_args()

    LIB = build_libraries()
    print('week libraries:', {c: {k: len(v) for k, v in d.items()} for c, d in LIB.items()})
    TEMP = {z: read_temp(c) for z, c in ZONE_CITY.items()}

    sites = pd.read_csv(f'{RAW}/ercot_datacenter_sites.csv')
    sites = sites[sites.include_default == 1].reset_index(drop=True)
    print(f'{len(sites)} sites, {sites.mw_published.sum():,.0f} MW announced')

    rng = np.random.default_rng(SEED)
    IT, CO = [], []
    for d in range(args.draws):
        a, b = one_draw(sites, LIB, TEMP, rng)
        IT.append(a); CO.append(b)
        if d == 0:
            it0, co0 = a, b
    arr_it = np.stack([x.values for x in IT])
    arr_co = np.stack([x.values for x in CO])
    arr    = arr_it + arr_co

    # ---------------- deliverables ----------------
    site0 = it0 + co0
    site0 = site0.round(2)
    site0.insert(len(site0.columns), 'TOTAL_MW', site0.sum(axis=1).round(2))
    site0.index.name = 'timestamp_local_central'
    site0.to_csv(f'{OUT}/dc_load_hourly_MW.csv')

    comp = pd.DataFrame({'P_IT_MW': it0.sum(axis=1).round(2),
                         'P_cool_MW': co0.sum(axis=1).round(2),
                         'P_site_MW': (it0 + co0).sum(axis=1).round(2)}, index=idx)
    comp.index.name = 'timestamp_local_central'
    comp.to_csv(f'{OUT}/dc_load_hourly_MW_IT_vs_cooling.csv')

    mean = pd.DataFrame(arr.mean(0), index=idx, columns=sites.site_id).round(2)
    p95  = pd.DataFrame(np.quantile(arr, .95, axis=0), index=idx, columns=sites.site_id).round(2)
    for df, nm in [(mean, 'ensemble_mean'), (p95, 'ensemble_p95')]:
        df.insert(len(df.columns), 'TOTAL_MW', df.sum(axis=1).round(2))
        df.index.name = 'timestamp_local_central'
        df.to_csv(f'{OUT}/dc_load_hourly_MW_{nm}.csv')

    byzone = site0.drop(columns='TOTAL_MW').T.groupby(sites.set_index('site_id').ercot_weather_zone).sum().T
    byzone.round(2).to_csv(f'{OUT}/dc_load_hourly_MW_by_weather_zone.csv')
    bybus = site0.drop(columns='TOTAL_MW').T.groupby(sites.set_index('site_id').tx2k_bus).sum().T
    bybus.round(2).to_csv(f'{OUT}/dc_load_hourly_MW_by_tx2k_bus.csv')

    # ---------------- per-site table ----------------
    pk, mn = arr_it.max(1) + arr_co.max(1), arr.mean(1)
    sp = arr.max(1)
    sites['w_train'], sites['w_inf'], sites['w_colo'] = zip(*[ARCHETYPE[a] for a in sites.archetype])
    sites['s'] = (sites.w_train * S_COMPONENT['train'] + sites.w_inf * S_COMPONENT['inference']
                  + sites.w_colo * S_COMPONENT['colo']).round(4)
    sites['M_0'] = sites.archetype.map(M0)
    sites['M_des'] = (sites.M_0 + B_SLOPE * (T_DES - T_REF)).round(4)
    sites['P_GPU_MW'] = (sites.mw_published / sites.M_des).round(1)
    sites['sim_peak_MW'] = sp.mean(0).round(1)
    sites['sim_mean_MW'] = mn.mean(0).round(1)
    sites['sim_load_factor'] = (mn / sp).mean(0).round(3)
    sites.to_csv(f'{OUT}/site_parameters_and_results.csv', index=False)

    # ---------------- summary ----------------
    tot = arr.sum(2)
    cf  = tot.max(1) / sp.sum(1)
    peak_h = [int(np.argmax(tot[d])) for d in range(len(tot))]
    summary = dict(
        year=YEAR, n_sites=int(len(sites)), announced_MW=float(sites.mw_published.sum()),
        draws=int(args.draws), seed=SEED,
        parameters=dict(s_component=S_COMPONENT, M_0=M0, b=B_SLOPE, T_ref=T_REF, T_des=T_DES,
                        p_shared_same_owner=P_SHARED, n_weeks=N_WEEKS, archetype=ARCHETYPE),
        fleet_coincident_peak_MW=dict(mean=float(tot.max(1).mean()),
                                      p05=float(np.quantile(tot.max(1), .05)),
                                      p95=float(np.quantile(tot.max(1), .95))),
        fleet_sum_of_site_peaks_MW=float(sp.sum(1).mean()),
        fleet_coincidence_factor=dict(mean=float(cf.mean()), p05=float(np.quantile(cf, .05)),
                                      p95=float(np.quantile(cf, .95))),
        fleet_mean_MW=float(tot.mean()), fleet_load_factor=float(tot.mean() / tot.max(1).mean()),
        fleet_energy_TWh=float(tot.mean(0).sum() / 1e6),
        IT_share_of_energy=float(arr_it.sum() / arr.sum()),
        cooling_share_of_energy=float(arr_co.sum() / arr.sum()),
        peak_month_by_draw=[int(idx[h].month) for h in peak_h],
        peak_timestamp_draw0=str(idx[peak_h[0]]))
    json.dump(summary, open(f'{OUT}/fleet_summary.json', 'w'), indent=1)

    print(f"\nfleet coincident peak  {summary['fleet_coincident_peak_MW']['mean']:,.0f} MW "
          f"[{summary['fleet_coincident_peak_MW']['p05']:,.0f}-{summary['fleet_coincident_peak_MW']['p95']:,.0f}]")
    print(f"coincidence factor     {summary['fleet_coincidence_factor']['mean']:.3f}")
    print(f"fleet mean             {summary['fleet_mean_MW']:,.0f} MW,  LF {summary['fleet_load_factor']:.3f}")
    print(f"annual energy          {summary['fleet_energy_TWh']:.1f} TWh   "
          f"(IT {100*summary['IT_share_of_energy']:.1f}%, cooling {100*summary['cooling_share_of_energy']:.1f}%)")
    print(f"draw-0 peak            {summary['peak_timestamp_draw0']}")
    print('\nwritten to', OUT)


if __name__ == '__main__':
    main()
