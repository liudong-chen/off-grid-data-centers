"""Shared machinery for the diesel-bridge sensitivity (ERCOT 2030, 17 campuses).

Adds a diesel genset to the off-grid machine catalogue at run time, so the existing
Monte Carlo (OffGridBYONG.lole_batch / lole_storage) and the exact merit-order fuel
integral (OffGridBYONG.merit_fuel_vec) score gas+diesel portfolios on exactly the same
random streams and the same load-duration machinery as the published designs.

Nothing in OffGridBYONG.py or offgrid_mip.py is edited: the catalogue is extended in
memory (og.CAT.append) and the module-level arrays that depend on it are rebuilt.

DIESEL PARAMETERS (base; every one overridable by environment variable)
    unit        3.0 MW      1,250-3,250 kW is the data-centre genset class
    capex       $800/kW     New_parameter.xlsx 'Diesel backup generator' (NREL/EDF/BofA)
                            alt: $1,374/kW 'Diesel peaker plant' (EIA/S&L TX 20-MW ICE)
    FOM         $35/kW-yr   same row
    VOM         $7.2/MWh    EIA/S&L engine-plant proxy (the backup row has 0; a unit that
                            RUNS for a year needs the engine-plant number)
    heat rate   9.5 MMBtu/MWh   (~70 gal/MWh)
    fuel        $20/MMBtu   ~ $2.77/gal delivered, off-road (no highway excise), from
                            Brent < $70/bbl real 2030 (AEO2026) + distillate crack +
                            delivery; STEO Sep-2026 wholesale diesel $3.00-3.74/gal
    FOR         6.32%       NERC GADS class-average EFORd for DIESEL units, 5-yr average
                            2020-2024 (as published in ISO-NE's GADS class averages)
    MTTR        37 h        mean time to repair of emergency/standby diesel gensets in two
                            large operational data sets (Marqusee & Jenket, Applied
                            Energy 2020, "Reliability of emergency and standby diesel
                            generators")
    planned     250 h/yr    derived from prime-duty service intervals (oil/filter every
                            250-500 h, valves 1,000-2,000 h, top-end overhaul
                            12,000-15,000 h): at ~3,000-8,000 run-h/yr that is
                            ~100-300 h/yr of scheduled downtime; the catalogue's gas
                            recip carries 300 h
    CRF         0.1019      20 yr at 8% real -- the paper's convention (the workbook's
                            7%/20 yr CRF 0.0944 is NOT used, to keep one convention)
    CO2         0.07413 t/MMBtu (163.45 lb/MMBtu, EIA)
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
H = 8760

_spec2 = importlib.util.spec_from_file_location('om', os.path.join(HERE, 'offgrid_mip.py'))
om = importlib.util.module_from_spec(_spec2); _spec2.loader.exec_module(om)
og = om.og                # the SAME module instance offgrid_mip anchors and scores with

CRF = og.CRF                                           # 0.1019
E = lambda k, d: float(os.environ.get(k, d))
DIESEL = dict(k='Diesel', name='Diesel genset, prime-rated', S=E('DSL_MW', 3.0),
              capex=E('DSL_CAPEX', 800.0), fom=E('DSL_FOM', 35.0), vom=E('DSL_VOM', 7.2),
              hr=E('DSL_HR', 9.5), q=E('DSL_FOR', 0.0632), mttr=int(E('DSL_MTTR', 37)),
              poh=int(E('DSL_POH', 250)), land='gas')
DIESEL['fixed_$MWyr'] = (DIESEL['capex']*CRF + DIESEL['fom'])*1000.0
DSL_FUEL = E('DSL_FUEL', 20.0)                         # $/MMBtu delivered
CO2_GAS, CO2_DSL = 0.05306, 0.07413                    # t/MMBtu
WIND_FIXED, SOLAR_FIXED, BATT_FIXED = om.WIND_FIXED, og.SOLAR_FIXED, og.BATT_FIXED
ACRES = om.ACRES
GASZ = json.load(open(os.path.join(HERE, 'zonal_gas_2030.json')))

_READY = False


def prepare():
    """Anchor gas capex to the on-grid candidates (paper basis), then append diesel."""
    global _READY
    if _READY:
        return
    om.anchor_capex()
    og.CAT.append(DIESEL)
    og.NC = len(og.CAT)
    og.S_MW = np.array([u['S'] for u in og.CAT])
    og.FX_MW = np.array([u['fixed_$MWyr'] for u in og.CAT])
    # The gas machines keep EXACTLY the published overhaul calendar (packed windows on the
    # five-type cycle); appending a sixth type to that cycle would move every gas window
    # and change the published designs' LOLE.  Diesel gensets get their own calendar:
    # unit j starts its 250-h service at hour j*250, so ~2.9% of a large genset fleet is
    # in service at any time and no two of the first 35 units overlap.
    _orig = og.maint_start
    def maint_start(i, j, _orig=_orig):
        if i < NG:
            return _orig(i, j)
        return (j*og.CAT[IDX_DSL]['poh']) % og.H
    og.maint_start = maint_start
    _READY = True


NG = 5                                   # gas types; diesel is index 5
IDX_DSL = 5


def var_costs(gas):
    """$/MWh for all six machines at this site's gas price and the diesel price."""
    v = [u['vom'] + u['hr']*gas for u in og.CAT[:NG]]
    v.append(DIESEL['vom'] + DIESEL['hr']*DSL_FUEL)
    return np.array(v)


# ------------------------------------------------------------------ site data
def load_sites(load_csv, meta_csv):
    ZM = {'Far West': 'FarWest', 'North Central': 'NorthCentral', 'South Central': 'SouthCentral'}
    d = pd.read_csv(load_csv, index_col=0, parse_dates=True)
    d = d[~((d.index.month == 2) & (d.index.day == 29))]
    meta = pd.read_csv(meta_csv)
    meta['zone'] = meta.ercot_weather_zone.map(lambda z: ZM.get(z, z))
    return d, meta


class Campus:
    def __init__(self, site_id, zone, load, design, cf_s, cf_w, mw_published):
        self.site, self.zone = site_id, zone
        self.load = np.asarray(load, float)[:H]
        self.peak, self.energy = float(self.load.max()), float(self.load.sum())
        self.gas = GASZ[zone]
        self.land = og.LAND[zone]
        self.cf_s, self.cf_w = cf_s, cf_w
        self.n0 = np.array(design['n'], int)          # 5 gas counts
        self.s, self.w, self.b = design['solar_MW'], design['wind_MW'], design['batt_MW']
        self.mw_published = float(mw_published)
        self.VR = var_costs(self.gas)
        self.design = design

    def net(self):
        x = self.load.copy()
        if self.s > 0 and self.cf_s is not None:
            x -= self.s*self.cf_s
        if self.w > 0 and self.cf_w is not None:
            x -= self.w*self.cf_w
        return x

    # ---- reliability
    def lole_many(self, N, reps, seed=20260826):
        """LOLE for a batch of 6-type portfolios (common random numbers)."""
        N = np.atleast_2d(np.asarray(N, np.int64))
        net = self.net()
        if self.b > 0:
            return np.array([og.lole_storage(net, n, reps, self.b, seed=seed) for n in N])
        return og.lole_batch(net, N, reps, seed=seed)

    # ---- cost, one year islanded
    def tiers(self, n6):
        """Energy served by each machine type, MWh/yr, exact merit order on net load."""
        n6 = np.asarray(n6, float)
        LD = og.ldc(np.maximum(self.net(), 0.0))
        order = np.argsort(self.VR)
        cum, Fc = 0.0, og.energy_above(np.array([0.0]), LD)[0]
        served = np.zeros(6)
        for i in order:
            cap = n6[i]*og.S_MW[i]
            Fn = og.energy_above(np.array([cum + cap]), LD)[0]
            served[i] = Fc - Fn
            cum, Fc = cum + cap, Fn
        unserved = Fc
        return served, unserved

    def cost(self, n6):
        n6 = np.asarray(n6, float)
        fixed_gas = float(n6[:NG] @ (og.S_MW[:NG]*og.FX_MW[:NG]))
        fixed_dsl = float(n6[IDX_DSL]*og.S_MW[IDX_DSL]*og.FX_MW[IDX_DSL])
        vre = self.s*SOLAR_FIXED + self.w*WIND_FIXED + self.b*BATT_FIXED
        acres = (float(n6 @ og.S_MW)*ACRES['gas'] + self.s*ACRES['solar']
                 + self.w*ACRES['wind'] + self.b*ACRES['storage'] + self.peak*ACRES['dc'])
        land = acres*self.land*og.LAND_CRF
        served, unserved = self.tiers(n6)
        fuel = float((served*self.VR).sum())
        fuel_gas = float((served[:NG]*self.VR[:NG]).sum())
        fuel_dsl = float(served[IDX_DSL]*self.VR[IDX_DSL])
        mmbtu_gas = float(sum(served[i]*og.CAT[i]['hr'] for i in range(NG)))
        mmbtu_dsl = float(served[IDX_DSL]*DIESEL['hr'])
        co2 = mmbtu_gas*CO2_GAS + mmbtu_dsl*CO2_DSL
        total = fixed_gas + fixed_dsl + vre + land + fuel
        return dict(fixed_gas=fixed_gas, fixed_diesel=fixed_dsl, vre_fixed=vre, land=land,
                    fuel=fuel, fuel_gas=fuel_gas, fuel_diesel=fuel_dsl, total=total,
                    lcoe=total/self.energy, served_by_type=served.tolist(),
                    unserved_MWh=unserved, co2_t=co2, gas_MW=float(n6[:NG] @ og.S_MW[:NG]),
                    diesel_MW=float(n6[IDX_DSL]*og.S_MW[IDX_DSL]))


RULE = os.environ.get('DSL_RULE', 'proportional')     # proportional | merit


def strip_gas_proportional(n0, share):
    """Remove the SAME fraction of every gas machine type (the sweep is a
    straight interpolation between the all-gas and the all-diesel fleet, so nothing in
    the replacement rule decides where the cost turns).

    Machines are whole, so x*n_j is rounded: floor first, then the largest fractional
    remainders get one more unit each, as long as that brings the removed MW closer to
    x of the gas MW.  A type with a single unit is therefore removed only once x >= 0.5
    (or earlier if that closes the MW gap).  Returns (n_remaining, removed_MW)."""
    n0 = np.array(n0, int)
    G = float(n0 @ og.S_MW[:NG])
    if share >= 0.999:
        return np.zeros(NG, int), G
    target = share*G
    want = share*n0
    rem = np.floor(want).astype(int)
    frac = want - rem
    removed = float(rem @ og.S_MW[:NG])
    for i in np.argsort(-frac):                    # largest remainder first
        if frac[i] <= 0 or rem[i] >= n0[i]:
            continue
        S = og.S_MW[i]
        if abs(removed + S - target) < abs(removed - target):
            rem[i] += 1; removed += S
    return n0 - rem, removed


def strip(n0, VR, share):
    return strip_gas_proportional(n0, share) if RULE == 'proportional' else strip_gas(n0, VR, share)


def removal_order(VR):
    """Gas types, most expensive to run first (aeros, recip, frame CT, then CC)."""
    return [int(i) for i in np.argsort(-VR[:NG])]


def strip_gas(n0, VR, share):
    """Remove whole gas machines, dearest-to-run first, until the removed MW is as close
    as possible to `share` of the gas fleet's MW.  Returns (n_remaining, removed_MW)."""
    n = np.array(n0, int)
    G = float(n @ og.S_MW[:NG])
    target = share*G
    if share >= 0.999:
        return np.zeros(NG, int), G
    removed = 0.0
    for i in removal_order(VR):
        while n[i] > 0:
            S = og.S_MW[i]
            if abs(removed + S - target) < abs(removed - target):
                n[i] -= 1; removed += S
            else:
                return n, removed
    return n, removed
