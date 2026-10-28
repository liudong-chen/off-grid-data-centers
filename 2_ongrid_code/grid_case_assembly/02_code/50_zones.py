"""Build the eight ERCOT weather-zone polygons from county boundaries.

The model is zonal, so the map has to be zonal. County geometry comes from the US Census
cartographic boundary file; counties are assigned to a weather zone by nearest anchor,
with the anchors placed at the population centre each zone is named for. That reproduces
ERCOT's zone map closely enough for a figure whose message is "two zones versus
seventeen sites", but the boundaries are approximate and the caption says so.

El Paso and Hudspeth are drawn but not coloured: they are in WECC, not ERCOT.

Writes out/tx_zones.json -- one dissolved polygon per zone, as plain lon/lat rings.
"""
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

SRC = '/tmp/counties.json'
OUT = 'build/out/tx_zones.json'

# anchor = the place each zone is named for, or its population centre
ANCHOR = {
    'North':        (-100.30, 34.90),   # Panhandle through Wichita Falls
    'West':         (-99.90, 32.10),    # Abilene / San Angelo
    'FarWest':      (-102.50, 31.30),   # Midland / Odessa, the Permian
    'NorthCentral': (-97.20, 32.75),    # Dallas-Fort Worth
    'East':         (-94.90, 31.90),    # Tyler / Longview
    'SouthCentral': (-98.10, 30.00),    # Austin / San Antonio
    'Coast':        (-95.50, 29.55),    # Houston / Galveston
    'South':        (-98.60, 27.40),    # Corpus Christi / Laredo
}
NOT_ERCOT = {'El Paso', 'Hudspeth'}

d = json.load(open(SRC))
feats = [f for f in d['features'] if f['id'].startswith('48')]
print(f'{len(feats)} Texas counties')

names = np.array([f['properties']['NAME'] for f in feats])
geoms = [shape(f['geometry']) for f in feats]
cent = np.array([[g.centroid.x, g.centroid.y] for g in geoms])

A = np.array(list(ANCHOR.values()))
keys = list(ANCHOR)
# longitude is compressed by cos(lat) at this latitude; scale so distance is physical
sc = np.array([np.cos(np.radians(31.0)), 1.0])
dist = np.linalg.norm((cent[:, None, :] - A[None, :, :])*sc, axis=2)
assign = np.array(keys)[dist.argmin(axis=1)]
assign[np.isin(names, list(NOT_ERCOT))] = 'outside'

out = {}
for z in keys + ['outside']:
    sel = [g for g, a in zip(geoms, assign) if a == z]
    if not sel:
        continue
    u = unary_union(sel).buffer(0.0)
    out[z] = mapping(u)
    n = sum(assign == z)
    print(f'  {z:13s} {n:3d} counties')

json.dump(out, open(OUT, 'w'))
print('written', OUT)
