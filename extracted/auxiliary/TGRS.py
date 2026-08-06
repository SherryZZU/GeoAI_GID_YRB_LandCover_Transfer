#!/usr/bin/env python3
"""Exact code-cell extraction from TGRS.ipynb.
Notebook magics/install commands are preserved as comments.
Secrets are redacted. This historical extraction is not the refactored CLI.
"""


# %% [notebook cell 0]
# NOTEBOOK_COMMAND: pip install cdsapi

# %% [notebook cell 1]
from pathlib import Path

API_KEY = '<REDACTED_CDS_API_KEY>'

creds_path = Path.home() / '.cdsapirc'

content = f"""url: https://cds.climate.copernicus.eu/api
key: {API_KEY}
"""

creds_path.write_text(content)
print(f"Saved to: {creds_path}")
print("\nFile contents:")
print(creds_path.read_text())

# %% [notebook cell 2]
from pathlib import Path

API_KEY = '<REDACTED_CDS_API_KEY>'

creds_path = Path('/root/.cdsapirc')
creds_path.write_text(f"url: https://cds.climate.copernicus.eu/api\nkey: {API_KEY}\n")

print("Credentials saved:")
print(creds_path.read_text())

# %% [notebook cell 3]
# Install
# NOTEBOOK_COMMAND: !pip install cdsapi -q

import cdsapi
from pathlib import Path

# Create output folder on Drive
OUT_DIR = Path('/content/drive/MyDrive/TGRS_Study/ERA5')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Test download — 1 variable, 1 year ────────────────────────────────────────
client = cdsapi.Client()

test_path = OUT_DIR / 'test_download.nc'

client.retrieve(
    'reanalysis-era5-land-monthly-means',
    {
        'product_type'   : ['monthly_averaged_reanalysis'],
        'variable'       : ['total_precipitation'],
        'year'           : ['2020'],
        'month'          : ['01'],
        'time'           : ['00:00'],
        'area'           : [42.5, 95.5, 32.0, 120.0],
        'data_format'    : 'netcdf',
        'download_format': 'unarchived',
    },
    str(test_path)
)

size_mb = test_path.stat().st_size / 1e6
print(f"\n✓ Test successful — {size_mb:.2f} MB downloaded")
print(f"  Saved to: {test_path}")

# %% [notebook cell 4]
import cdsapi
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = '<REDACTED_CDS_API_KEY>'
OUT_DIR = Path('/content/drive/MyDrive/TGRS_Study/ERA5')
OUT_DIR.mkdir(parents=True, exist_ok=True)

Path('/root/.cdsapirc').write_text(
    f"url: https://cds.climate.copernicus.eu/api\nkey: {API_KEY}\n"
)

YRB_BBOX = [42.5, 95.5, 32.0, 120.0]
client   = cdsapi.Client()

VARIABLES = [
    '2m_temperature',
    'maximum_2m_temperature_since_previous_post_processing',
    'minimum_2m_temperature_since_previous_post_processing',
    'total_precipitation',
    'potential_evaporation',
    'surface_solar_radiation_downwards',
    'surface_net_solar_radiation',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_dewpoint_temperature',
]

DECADES = [
    {'years': [str(y) for y in range(1958, 1970)], 'tag': '1958_1969'},
    {'years': [str(y) for y in range(1970, 1980)], 'tag': '1970_1979'},
    {'years': [str(y) for y in range(1980, 1990)], 'tag': '1980_1989'},
    {'years': [str(y) for y in range(1990, 2000)], 'tag': '1990_1999'},
    {'years': [str(y) for y in range(2000, 2010)], 'tag': '2000_2009'},
    {'years': [str(y) for y in range(2010, 2020)], 'tag': '2010_2019'},
    {'years': [str(y) for y in range(2020, 2024)], 'tag': '2020_2023'},
]

MONTHS = [f'{m:02d}' for m in range(1, 13)]

# ── Download ──────────────────────────────────────────────────────────────────
print("ERA5 full download started — 7 decades")
print(f"Output: {OUT_DIR}")
print("⚠ CDS maintenance on 15 June — complete today\n")

failed = []

for i, decade in enumerate(DECADES, 1):
    out_path = OUT_DIR / f'ERA5_YRB_{decade["tag"]}.nc'

    if out_path.exists() and out_path.stat().st_size > 1e6:
        print(f'[{i}/7] ✓ Already done — {out_path.name} '
              f'({out_path.stat().st_size/1e6:.0f} MB)')
        continue

    print(f'[{i}/7] Downloading {decade["tag"]} '
          f'({len(decade["years"])} years × 12 months) ...')

    try:
        client.retrieve(
            'reanalysis-era5-land-monthly-means',
            {
                'product_type'   : ['monthly_averaged_reanalysis'],
                'variable'       : VARIABLES,
                'year'           : decade['years'],
                'month'          : MONTHS,
                'time'           : ['00:00'],
                'area'           : YRB_BBOX,
                'data_format'    : 'netcdf',
                'download_format': 'unarchived',
            },
            str(out_path)
        )
        size_mb = out_path.stat().st_size / 1e6
        print(f'       ✓ Done — {out_path.name} ({size_mb:.0f} MB)\n')

    except Exception as e:
        print(f'       ✗ Failed — {decade["tag"]}: {e}\n')
        failed.append(decade['tag'])
        continue   # skip to next decade, don't abort entire run

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '═'*55)
print('DOWNLOAD SUMMARY')
print('═'*55)
files      = sorted(OUT_DIR.glob('ERA5_YRB_[0-9]*.nc'))
total_gb   = sum(f.stat().st_size for f in files) / 1e9
print(f'Completed : {len(files)}/7 files')
print(f'Total size: {total_gb:.2f} GB')
for f in files:
    print(f'  ✓ {f.name}  ({f.stat().st_size/1e6:.0f} MB)')
if failed:
    print(f'\nFailed ({len(failed)}) — re-run to retry:')
    for d in failed:
        print(f'  ✗ {d}')
else:
    print('\n✓ All 7 decades downloaded successfully')
print('═'*55)

# %% [notebook cell 5]
# NOTEBOOK_COMMAND: !pip install netcdf4 -q

# %% [notebook cell 6]
# ── Cell 1: Install and restart ───────────────────────────────────────────────
# NOTEBOOK_COMMAND: !pip install netcdf4 -q
print("Installation done — now go to Runtime → Restart session → then run Cell 2")

# %% [notebook cell 7]
# ── Cell 2: Run after restart ─────────────────────────────────────────────────
import xarray as xr
from pathlib import Path

# Verify netcdf4 is now available
import xarray.backends.plugins as p
print("Available engines:", list(p.list_engines().keys()))

OUT_DIR = Path('/content/drive/MyDrive/TGRS_Study/ERA5')
files   = sorted(OUT_DIR.glob('ERA5_YRB_[0-9]*.nc'))
print(f'\nFound {len(files)} decade files to merge')
for f in files:
    print(f'  {f.name}  ({f.stat().st_size/1e6:.0f} MB)')

# Merge
print('\nMerging...')
ds = xr.open_mfdataset(files, combine='by_coords', engine='netcdf4')

out_path = OUT_DIR / 'ERA5_YRB_1958_2023_merged.nc'
ds.to_netcdf(out_path)

size_mb = out_path.stat().st_size / 1e6
print(f'\n✓ Merged → {out_path.name} ({size_mb:.0f} MB)')
print(f'  Period     : {str(ds.time.values[0])[:7]} → {str(ds.time.values[-1])[:7]}')
print(f'  Time steps : {len(ds.time)} months (expect 792)')
print(f'  Variables  : {list(ds.data_vars)}')
print(f'  Grid       : {dict(ds.dims)}')

# %% [notebook cell 8]
import xarray as xr
from pathlib import Path

OUT_DIR  = Path('/content/drive/MyDrive/TGRS_Study/ERA5')
out_path = OUT_DIR / 'ERA5_YRB_1958_2023_merged.nc'

# Open and inspect
ds = xr.open_dataset(out_path, engine='netcdf4')

# Print whatever dimensions and variables exist
print(f'✓ ERA5_YRB_1958_2023_merged.nc — {out_path.stat().st_size/1e6:.0f} MB')
print(f'\nDimensions  : {dict(ds.dims)}')
print(f'Variables   : {list(ds.data_vars)}')
print(f'Coordinates : {list(ds.coords)}')

# Find time dimension dynamically
time_dim = [d for d in ds.dims if 'time' in d.lower() or 'valid' in d.lower()]
if time_dim:
    t = ds[time_dim[0]]
    print(f'\nTime dim    : {time_dim[0]}')
    print(f'Period      : {str(t.values[0])[:7]} → {str(t.values[-1])[:7]}')
    print(f'Time steps  : {len(t)} months (expect 792)')
else:
    print(f'\nAll coords  : {ds.coords}')

ds.close()
print('\n✓ All data downloads complete — ready for Phase 1')

# %% [notebook cell 9]
# NOTEBOOK_COMMAND: !pip install kaggle

# %% [notebook cell 10]
# ============================================================================
# Cross-sectional mediation test — does ET partitioning mediate the
# teleconnection -> drought-index-dominance relationship across 308 stations?
#
# Reframe being tested (sanity-check version, precursor to monthly PCMCI+):
#   Background dryness / vegetation  ->  ET partition (soil-evap fraction)  ->
#   which teleconnection dominates the station (PDO vs IOD)
#
# This is a SPATIAL cross-section: it tells us whether the spatial pattern is
# consistent with mediation. It is NOT a temporal causal claim — the monthly
# station-level PCMCI+ is the confirmatory analysis. Read the caveats at the end.
# ============================================================================

import glob, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

# ---- 1. Locate & load -------------------------------------------------------
hits = glob.glob('/content/drive/MyDrive/**/phase7_feature_table.csv', recursive=True)
assert hits, "phase7_feature_table.csv not found under /content/drive/MyDrive — set the path manually."
PATH = hits[0]
print(f"Loaded: {PATH}")
df = pd.read_csv(PATH)
df.columns = [c.strip() for c in df.columns]
print(f"Shape: {df.shape}")

# ---- 2. Build the outcome: PDO-vs-IOD dominance per station -----------------
# r_PDO / r_IOD are per-station teleconnection correlations. "Dominance" = the
# RELATIVE STRENGTH (magnitude), so D > 0 means PDO is the stronger control.
# >>> VERIFY what r_PDO/r_IOD were correlated against (which index? detrended?)
#     before over-interpreting — the printout below helps you check. <
for c in ['r_PDO', 'r_IOD', 'r_ENSO', 'Eb_frac', 'T_frac', 'NDVI_mean',
          'LAI_mean', 'Aridity', 'Ann_Precip_mm', 'Elev_proxy_m',
          'SMsurf_mean', 'SMroot_mean', 'latitude', 'longitude']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df['D_PDO_minus_IOD'] = df['r_PDO'].abs() - df['r_IOD'].abs()

# ---- 3. Explore BEFORE testing (verify the bivariate landscape) -------------
key = ['Eb_frac', 'T_frac', 'NDVI_mean', 'LAI_mean', 'Aridity',
       'SMsurf_mean', 'D_PDO_minus_IOD', 'r_PDO', 'r_IOD']
print("\n--- Correlation matrix (Pearson) among key variables ---")
print(df[key].corr().round(2).to_string())
print("\nExpected if the reframe holds: Eb_frac should correlate POSITIVELY with")
print("D_PDO_minus_IOD (more soil evaporation -> more PDO-dominant).")

# ---- 4. Mediation engine (standardized paths, bootstrap indirect CI) --------
def _beta(Xcols_df, y):
    """OLS coefficients via least squares, intercept added; returns dict col->coef."""
    Xm = np.column_stack([np.ones(len(Xcols_df)), Xcols_df.values])
    coef, *_ = np.linalg.lstsq(Xm, y.values, rcond=None)
    return dict(zip(['const'] + list(Xcols_df.columns), coef))

def mediation(df, X, M, Y, covars=(), n_boot=2000, seed=42):
    covars = list(covars)
    cols = [X, M, Y] + covars
    d = df[cols].apply(pd.to_numeric, errors='coerce').dropna()
    n = len(d)
    z = (d - d.mean()) / d.std(ddof=0)            # standardize -> comparable betas

    # point estimates + p-values via statsmodels
    m_c = sm.OLS(z[Y], sm.add_constant(z[[X] + covars])).fit()        # total effect c
    m_a = sm.OLS(z[M], sm.add_constant(z[[X] + covars])).fit()        # path a (X->M)
    m_b = sm.OLS(z[Y], sm.add_constant(z[[X, M] + covars])).fit()     # path b, direct c'
    a, b, c, cp = m_a.params[X], m_b.params[M], m_c.params[X], m_b.params[X]
    indirect = a * b

    # bootstrap the indirect effect a*b (fast: numpy lstsq)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        s = z.iloc[rng.integers(0, n, n)]
        a_b = _beta(s[[X] + covars], s[M])[X]
        b_b = _beta(s[[X, M] + covars], s[Y])[M]
        boot.append(a_b * b_b)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    prop = indirect / c if abs(c) > 1e-9 else np.nan
    return dict(n=n, a=a, b=b, c=c, cprime=cp, indirect=indirect,
                ci=(lo, hi), sig=(lo > 0) or (hi < 0),
                prop_mediated=prop, p_a=m_a.pvalues[X], p_b=m_b.pvalues[M],
                p_c=m_c.pvalues[X])

def report(tag, r):
    star = "SIGNIFICANT (CI excludes 0)" if r['sig'] else "not significant"
    print(f"\n=== {tag}  (n={r['n']}) ===")
    print(f"  a (X->M)            : {r['a']:+.3f}  (p={r['p_a']:.3g})")
    print(f"  b (M->Y | X)        : {r['b']:+.3f}  (p={r['p_b']:.3g})")
    print(f"  c  total effect     : {r['c']:+.3f}  (p={r['p_c']:.3g})")
    print(f"  c' direct effect    : {r['cprime']:+.3f}")
    print(f"  indirect (a*b)      : {r['indirect']:+.3f}  95% CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  -> {star}")
    print(f"  proportion mediated : {r['prop_mediated']:.0%}" if np.isfinite(r['prop_mediated']) else "  proportion mediated : n/a (weak total effect)")

# ---- 5. Primary mediation ---------------------------------------------------
# X = Aridity (exogenous background dryness), M = Eb_frac (realized partition),
# Y = PDO-minus-IOD dominance. Tests: does dryness act THROUGH the ET partition?
report("PRIMARY: Aridity -> Eb_frac -> PDO/IOD dominance",
       mediation(df, 'Aridity', 'Eb_frac', 'D_PDO_minus_IOD'))

# ---- 6. Robustness: swap the predictor and the mediator ---------------------
print("\n\n################  ROBUSTNESS SWEEP  ################")
for X in ['Aridity', 'NDVI_mean', 'LAI_mean']:
    for M in ['Eb_frac', 'T_frac']:
        r = mediation(df, X, M, 'D_PDO_minus_IOD', covars=['Elev_proxy_m'])
        flag = "***" if r['sig'] else "   "
        print(f"{flag} X={X:13s} M={M:8s}  indirect={r['indirect']:+.3f} "
              f"CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  c'={r['cprime']:+.3f}  n={r['n']}")
print("(*** = bootstrap CI excludes zero; covariate-adjusted for elevation)")

# ---- 7. Spatial-autocorrelation diagnostic (Moran's I on residuals) ---------
def morans_i(values, lat, lon, k=8, n_perm=999, seed=42):
    coords = np.radians(np.column_stack([lat, lon]))
    la, lo_ = coords[:, 0], coords[:, 1]
    dlat = la[:, None] - la[None, :]; dlon = lo_[:, None] - lo_[None, :]
    h = np.sin(dlat/2)**2 + np.cos(la)[:, None]*np.cos(la)[None, :]*np.sin(dlon/2)**2
    dist = 2*np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    np.fill_diagonal(dist, np.inf)
    n = len(values); W = np.zeros((n, n))
    for i in range(n):
        W[i, np.argsort(dist[i])[:k]] = 1.0
    W /= W.sum(axis=1, keepdims=True)
    zc = values - values.mean()
    I = (zc @ (W @ zc)) / (zc @ zc)
    rng = np.random.default_rng(seed)
    perm = [( (p := rng.permutation(zc)) @ (W @ p)) / (p @ p) for _ in range(n_perm)]
    p_val = (np.sum(np.array(perm) >= I) + 1) / (n_perm + 1)   # one-sided (clustering)
    return I, p_val

dd = df[['D_PDO_minus_IOD', 'Aridity', 'Eb_frac', 'latitude', 'longitude']].dropna()
res = sm.OLS(dd['D_PDO_minus_IOD'],
             sm.add_constant(dd[['Aridity', 'Eb_frac']])).fit().resid
I_raw, p_raw = morans_i(dd['D_PDO_minus_IOD'].values, dd['latitude'].values, dd['longitude'].values)
I_res, p_res = morans_i(res.values, dd['latitude'].values, dd['longitude'].values)
print("\n\n--- Spatial autocorrelation (Moran's I, k=8 neighbours) ---")
print(f"  Outcome D            : I={I_raw:+.3f}  p={p_raw:.3g}")
print(f"  Mediation residuals  : I={I_res:+.3f}  p={p_res:.3g}")
print("  If residual I is positive & significant, the bootstrap CIs above are")
print("  optimistic — spatial structure remains. The monthly PCMCI+ (time-domain,")
print("  per station) is the rigorous confirmatory test and avoids this.")

# ---- 8. Honest caveats ------------------------------------------------------
print("""
============================ READ THIS ============================
1. Cross-sectional / spatial: consistent-with-mediation, not proof of a
   temporal causal effect. Confirm with the monthly station-level PCMCI+.
2. Verify what r_PDO / r_IOD were computed against (which index, detrended?).
   The dominance outcome inherits that definition.
3. Eb_frac and Aridity are physically linked; the covariate-adjusted sweep
   guards against "it's just an aridity gradient," but interpret a*b with that
   in mind.
4. Only tests the PDO-vs-IOD strength contrast. The full SPEI<->ETdef REVERSAL
   needs both monthly indices together (next pipeline).
===================================================================
""")

# %% [notebook cell 11]
# ============================================================================
# Reversal-robustness audit — are the 121 "flip" stations real, and is the
# flip a change of OCEAN (PDO vs IOD) or a change of TIMESCALE (lag-0 vs lag 6-9)?
#
# Inputs (auto-located under /content/drive/MyDrive):
#   Station_SPEI3_Monthly.csv          ETdef_indices_308stations_monthly.csv
#   teleconnection_monthly.csv         phase7_feature_table.csv (for lat/lon)
#
# This is DESCRIPTIVE robustness checking, not the confirmatory causal test.
# It tells us which phenomenon the paper should explain before we build PCMCI+.
# ============================================================================
import glob, warnings, numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')

MAX_LAG   = 12          # months; positive lag = teleconnection LEADS the index
MIN_PAIRS = 80          # minimum overlapping months to attempt a correlation
FDR_Q     = 0.05

def find(name):
    h = glob.glob(f'/content/drive/MyDrive/**/{name}', recursive=True)
    assert h, f"{name} not found under /content/drive/MyDrive"
    return h[0]

# ---- robust loader v2: handles LONG (StationID + value col) and WIDE -------
def load_index_long_to_wide(path, value_candidates):
    """value_candidates: list of possible value column names, case-insensitive."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}

    # --- time axis ---
    if 'year' in low and 'month' in low:
        df['_t'] = pd.PeriodIndex(year=df[low['year']].astype(int),
                                  month=df[low['month']].astype(int), freq='M')
    else:
        dcol = next((low[c] for c in ('date','time','datetime') if c in low), None)
        assert dcol, f"No Year/Month or date column in {path}"
        df['_t'] = pd.PeriodIndex(pd.to_datetime(df[dcol]), freq='M')

    # --- station id column (now includes 'stationid') ---
    idcol = next((low[c] for c in
                  ('stationid','station_id','station','id','stn','site','code')
                  if c in low), None)
    # --- value column: first matching candidate ---
    valcol = None
    for cand in value_candidates:
        if cand.lower() in low:
            valcol = low[cand.lower()]; break
    if valcol is None:  # fuzzy contains-match fallback
        for cand in value_candidates:
            hit = next((c for c in df.columns if cand.lower() in c.lower()), None)
            if hit: valcol = hit; break

    assert idcol and valcol, (f"{path}: idcol={idcol} valcol={valcol}; "
                              f"columns={list(df.columns)}")
    w = df.pivot_table(index='_t', columns=idcol, values=valcol)
    w.columns = [str(int(c)) if isinstance(c,(int,float)) and float(c).is_integer()
                 else str(c) for c in w.columns]
    print(f"  loaded {path.split('/')[-1]}: {w.shape[1]} stations, "
          f"{w.shape[0]} months, value='{valcol}'")
    return w

print("Loading…")
spei = deseasonalize(load_index_long_to_wide(
        find('Station_SPEI3_Monthly.csv'), ['SPEI3','spei3','spei']))
etd  = deseasonalize(load_index_long_to_wide(
        find('ETdef_indices_308stations_monthly.csv'),
        ['ETdef3','ETdef_3','etdef3','etdef','ETdef-3']))
tc = pd.read_csv(find('teleconnection_monthly.csv')); tc.columns=[c.strip() for c in tc.columns]
tc['_t'] = pd.PeriodIndex(year=tc['Year'].astype(int), month=tc['Month'].astype(int), freq='M')
tc = tc.set_index('_t')[['ENSO','PDO','IOD']]
tc = tc.groupby(tc.index.month).transform(lambda x: x - x.mean())

# common monthly grid (gap-safe integer lags)
grid = pd.period_range(min(spei.index.min(), etd.index.min(), tc.index.min()),
                       max(spei.index.max(), etd.index.max(), tc.index.max()), freq='M')
spei, etd, tc = spei.reindex(grid), etd.reindex(grid), tc.reindex(grid)
stations = sorted(set(spei.columns) & set(etd.columns))
print(f"Stations in both indices: {len(stations)} | months on grid: {len(grid)} "
      f"({grid.min()}–{grid.max()})")

def lag1(a):
    a = a[~np.isnan(a)]
    return np.corrcoef(a[:-1], a[1:])[0,1] if len(a) > 3 else 0.0

def best_coupling(x, g):
    """Scan lags 0..MAX_LAG (g leads x). Return (best_r, best_lag, p_neff, n_eff)."""
    best = (0.0, 0, 1.0, np.nan)
    for L in range(0, MAX_LAG+1):
        xi = x[L:] if L else x
        gi = g[:len(g)-L] if L else g
        mask = ~(np.isnan(xi) | np.isnan(gi))
        n = mask.sum()
        if n < MIN_PAIRS: continue
        xv, gv = xi[mask], gi[mask]
        if xv.std()==0 or gv.std()==0: continue
        r = np.corrcoef(xv, gv)[0,1]
        neff = n * (1 - lag1(xv)*lag1(gv)) / (1 + lag1(xv)*lag1(gv))
        neff = max(3.0, min(neff, n))
        t = r*np.sqrt((neff-2)/max(1-r**2,1e-9))
        p = 2*stats.t.sf(abs(t), neff-2)
        if abs(r) > abs(best[0]): best = (r, L, p, neff)
    return best

# ---- per-station couplings for both indices --------------------------------
rows = []
for s in stations:
    rec = {'station': s}
    for tag, W in (('SPEI', spei), ('ETd', etd)):
        x = W[s].values.astype(float)
        for oc in ('PDO','IOD','ENSO'):
            r,L,p,ne = best_coupling(x, tc[oc].values.astype(float))
            rec[f'{tag}_{oc}_r']=r; rec[f'{tag}_{oc}_lag']=L
            rec[f'{tag}_{oc}_p']=p; rec[f'{tag}_{oc}_neff']=ne
        # dominant ocean = larger |r| between PDO and IOD
        dom = 'PDO' if abs(rec[f'{tag}_PDO_r'])>=abs(rec[f'{tag}_IOD_r']) else 'IOD'
        rec[f'{tag}_dom']  = dom
        rec[f'{tag}_dom_r']= rec[f'{tag}_{dom}_r']
        rec[f'{tag}_dom_lag']= rec[f'{tag}_{dom}_lag']
        rec[f'{tag}_dom_p']= rec[f'{tag}_{dom}_p']
        rec[f'{tag}_margin']= abs(rec[f'{tag}_PDO_r'])-abs(rec[f'{tag}_IOD_r'])  # decision margin
    rows.append(rec)
res = pd.DataFrame(rows)

# ---- FDR across stations on the dominant-coupling p (per index) ------------
def bh(pvals, q=FDR_Q):
    p = np.asarray(pvals); n=len(p); order=np.argsort(p); ranked=p[order]
    thresh = q*(np.arange(1,n+1))/n
    passed = ranked <= thresh
    cut = np.max(np.where(passed)[0]) if passed.any() else -1
    out = np.zeros(n, bool)
    if cut>=0: out[order[:cut+1]] = True
    return out
res['SPEI_dom_sig'] = bh(res['SPEI_dom_p'].values)
res['ETd_dom_sig']  = bh(res['ETd_dom_p'].values)
res['both_sig']     = res['SPEI_dom_sig'] & res['ETd_dom_sig']

# ---- flips ------------------------------------------------------------------
res['flip_general'] = res['SPEI_dom'] != res['ETd_dom']
res['flip_headline']= (res['SPEI_dom']=='IOD') & (res['ETd_dom']=='PDO')

# ---- attach lat/lon if available -------------------------------------------
try:
    ft = pd.read_csv(find('phase7_feature_table.csv')); ft.columns=[c.strip() for c in ft.columns]
    ft['station']=ft['station_id'].astype(str)
    res = res.merge(ft[['station','longitude','latitude','sub_basin_id']], on='station', how='left')
except Exception as e:
    print("lat/lon merge skipped:", e)

# ============================ REPORT ========================================
n = len(res)
print("\n"+"="*70)
print("REVERSAL ROBUSTNESS AUDIT")
print("="*70)

print(f"\n[1] Coupling strength reality check")
for tag in ('SPEI','ETd'):
    anysig = res[f'{tag}_dom_sig'].mean()
    medr   = res[f'{tag}_dom_r'].abs().median()
    print(f"  {tag}: median |dominant r| = {medr:.3f} | stations FDR-significant = "
          f"{res[f'{tag}_dom_sig'].sum()}/{n} ({anysig:.0%})")

print(f"\n[2] Flip counts (your published figure: 121)")
print(f"  General flip (dominant ocean differs SPEI vs ETdef): {res['flip_general'].sum()}")
print(f"  Headline flip (IOD in SPEI -> PDO in ETdef)        : {res['flip_headline'].sum()}")

fl = res[res['flip_headline']]
print(f"\n[3] Are the headline flips robust?  (n={len(fl)})")
if len(fl):
    print(f"  both couplings FDR-significant : {fl['both_sig'].sum()}/{len(fl)} "
          f"({fl['both_sig'].mean():.0%})")
    print(f"  median decision margin |r_PDO|-|r_IOD|:  SPEI={fl['SPEI_margin'].median():+.3f}  "
          f"ETd={fl['ETd_margin'].median():+.3f}")
    print(f"    (margins near 0 => 'dominance' is a near-tie / coin-flip)")

print(f"\n[4] Ocean shift vs TIMESCALE shift")
print(f"  Median best-lag of dominant coupling:  SPEI={res['SPEI_dom_lag'].median():.0f} mo   "
      f"ETd={res['ETd_dom_lag'].median():.0f} mo")
paired = res.dropna(subset=['SPEI_dom_lag','ETd_dom_lag'])
try:
    w_stat,w_p = stats.wilcoxon(paired['SPEI_dom_lag'], paired['ETd_dom_lag'])
    direction = "ETdef SHORTER" if paired['ETd_dom_lag'].median()<paired['SPEI_dom_lag'].median() else "ETdef longer/equal"
    print(f"  Wilcoxon paired lag test: p={w_p:.3g}  ({direction})")
except Exception as e:
    print("  Wilcoxon failed:", e)

print("\n"+"-"*70)
print("ADAPTIVE READ")
print("-"*70)
frac_both = fl['both_sig'].mean() if len(fl) else 0
if frac_both < 0.5:
    print(f"• Only {frac_both:.0%} of headline flips have BOTH couplings significant ->")
    print("  the 'ocean reversal' framing is FRAGILE; many flips sit between two")
    print("  near-zero correlations. Do NOT anchor the paper on ocean dominance alone.")
else:
    print(f"• {frac_both:.0%} of headline flips are doubly-significant -> ocean reversal")
    print("  has real support; worth carrying into the causal model.")
try:
    if w_p < 0.05 and paired['ETd_dom_lag'].median() < paired['SPEI_dom_lag'].median():
        print("• ETdef couplings peak at SHORTER lags than SPEI (significant) -> the")
        print("  reframe as a RESPONSE-TIMESCALE shift (storage- vs demand-limited) is")
        print("  supported and is more robust than the ocean-identity contrast.")
except: pass
print("• Confirmatory step remains the monthly per-station PCMCI+ (time-domain).")

out = '/content/drive/MyDrive/TGRS_Study/reversal_audit_perstation.csv'
import os; os.makedirs('/content/drive/MyDrive/TGRS_Study', exist_ok=True)
res.to_csv(out, index=False)
print(f"\nSaved per-station audit -> {out}")

# %% [notebook cell 12]
# ============================================================================
# Sub-basin & cluster reversal audit  (decision-grade)
# Bar locked in advance: lag-0, Bretherton p<0.05, block-bootstrap CI excl. 0,
# + artifact gate (>=60% member sign agreement & survives dropping top-2 stns).
# Scales: 8 sub-basins | 4 reaches | 3 reaches.  Comparable to station audit.
# ============================================================================
import glob, warnings, numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')

MAX_LAG, MIN_PAIRS, N_BOOT, SEED = 12, 60, 3000, 42
SIGN_AGREE_MIN = 0.60
rng = np.random.default_rng(SEED)

def find(name):
    h = glob.glob(f'/content/drive/MyDrive/**/{name}', recursive=True)
    assert h, f"{name} not found under /content/drive/MyDrive"
    return h[0]

# ---------- robust long->wide loader (StationID + value col) ----------------
def load_wide(path, value_candidates):
    df = pd.read_csv(path); df.columns=[c.strip() for c in df.columns]
    low={c.lower():c for c in df.columns}
    df['_t']=pd.PeriodIndex(year=df[low['year']].astype(int),
                            month=df[low['month']].astype(int), freq='M')
    idcol=next((low[c] for c in ('stationid','station_id','station','id') if c in low),None)
    val=None
    for cand in value_candidates:
        if cand.lower() in low: val=low[cand.lower()]; break
    if val is None:
        for cand in value_candidates:
            hit=next((c for c in df.columns if cand.lower() in c.lower()),None)
            if hit: val=hit; break
    assert idcol and val, f"{path}: idcol={idcol} val={val}; cols={list(df.columns)}"
    w=df.pivot_table(index='_t', columns=idcol, values=val)
    w.columns=[str(int(c)) for c in w.columns]
    print(f"  {path.split('/')[-1]}: {w.shape[1]} stns x {w.shape[0]} mo (val='{val}')")
    return w

def deseason(w):
    return w.groupby(w.index.month).transform(lambda x: x-x.mean())

print("Loading…")
SPEI_raw = load_wide(find('Station_SPEI3_Monthly.csv'), ['SPEI3','spei3','spei'])
ETD_raw  = load_wide(find('ETdef_indices_308stations_monthly.csv'),
                     ['ETdef_3','ETdef3','etdef'])
tc=pd.read_csv(find('teleconnection_monthly.csv')); tc.columns=[c.strip() for c in tc.columns]
tc['_t']=pd.PeriodIndex(year=tc['Year'].astype(int),month=tc['Month'].astype(int),freq='M')
tc=tc.set_index('_t')[['ENSO','PDO','IOD']]
tc=tc.groupby(tc.index.month).transform(lambda x:x-x.mean())

# ---------- station -> sub-basin map ----------------------------------------
def load_map():
    try:
        m=pd.read_csv(find('station_subbasin_map.csv')); m.columns=[c.strip() for c in m.columns]
        low={c.lower():c for c in m.columns}
        sid=next(low[c] for c in low if 'station' in c or c=='id')
        bid=next(low[c] for c in low if 'sub' in c or 'basin' in c)
        return {str(int(s)):int(b) for s,b in zip(m[sid],m[bid]) if pd.notna(b)}
    except Exception as e:
        print("  station_subbasin_map missing, using phase7_feature_table:", e)
        ft=pd.read_csv(find('phase7_feature_table.csv')); ft.columns=[c.strip() for c in ft.columns]
        return {str(int(s)):int(b) for s,b in zip(ft['station_id'],ft['sub_basin_id']) if pd.notna(b)}
S2B=load_map()
print(f"  station->subbasin map: {len(S2B)} stations, basins={sorted(set(S2B.values()))}")

# common grid
grid=pd.period_range('1981-01','2020-12',freq='M')
SPEI_raw,ETD_raw,tc=SPEI_raw.reindex(grid),ETD_raw.reindex(grid),tc.reindex(grid)

# ---------- aggregation unit definitions ------------------------------------
REACH4={'Upper':[109,78,89],'Middle':[59,75,100],'Lower':[108],'Endorheic':[60]}
REACH3={'Upper+Endo':[109,78,89,60],'Middle':[59,75,100],'Lower':[108]}
def stns_for(basin_ids):
    bset=set(basin_ids); return [s for s,b in S2B.items() if b in bset]

UNITS=[]
for b in sorted(set(S2B.values())):
    UNITS.append(('subbasin', f'SB{b}', stns_for([b])))
for name,bl in REACH4.items(): UNITS.append(('reach4', name, stns_for(bl)))
for name,bl in REACH3.items(): UNITS.append(('reach3', name, stns_for(bl)))

# ---------- stats helpers ----------------------------------------------------
def lag1(a):
    a=a[~np.isnan(a)]; return np.corrcoef(a[:-1],a[1:])[0,1] if len(a)>3 else 0.0

def align(x,g,L):
    xi=x[L:] if L else x; gi=g[:len(g)-L] if L else g
    m=~(np.isnan(xi)|np.isnan(gi)); return xi[m],gi[m]

def neff_p(r,xv,gv):
    n=len(xv); ne=n*(1-lag1(xv)*lag1(gv))/(1+lag1(xv)*lag1(gv)); ne=max(3.0,min(ne,n))
    t=r*np.sqrt((ne-2)/max(1-r**2,1e-9)); return 2*stats.t.sf(abs(t),ne-2), ne

def block_boot_ci(xv,gv):
    n=len(xv); L=max(3,int(round(n**(1/3)*1.5))); nb=int(np.ceil(n/L)); out=[]
    for _ in range(N_BOOT):
        starts=rng.integers(0,n,nb); idx=np.concatenate([np.arange(s,s+L) for s in starts])%n
        idx=idx[:n]; xb,gb=xv[idx],gv[idx]
        if xb.std()==0 or gb.std()==0: continue
        out.append(np.corrcoef(xb,gb)[0,1])
    return np.percentile(out,[2.5,97.5]) if out else (np.nan,np.nan)

def couple(xfull,gfull,lag):
    xv,gv=align(xfull,gfull,lag)
    if len(xv)<MIN_PAIRS or xv.std()==0 or gv.std()==0: return None
    r=np.corrcoef(xv,gv)[0,1]; p,ne=neff_p(r,xv,gv); lo,hi=block_boot_ci(xv,gv)
    return dict(r=r,p=p,neff=ne,ci=(lo,hi),n=len(xv),
                ci_excl0=(lo>0 or hi<0), passes=(p<0.05 and (lo>0 or hi<0)))

def best_lag_scan(xfull,gfull):
    best=None
    for L in range(0,MAX_LAG+1):
        c=couple(xfull,gfull,L)
        if c and (best is None or abs(c['r'])>abs(best['r'])): best=c; best['lag']=L
    return best

# ---------- run audit --------------------------------------------------------
rows=[]
for scale,name,stns in UNITS:
    stns=[s for s in stns if s in SPEI_raw.columns and s in ETD_raw.columns]
    if not stns: continue
    agg_spei=deseason(SPEI_raw[stns].mean(axis=1).to_frame('v'))['v'].values
    agg_etd =deseason(ETD_raw[stns].mean(axis=1).to_frame('v'))['v'].values
    for idx_tag,agg in (('SPEI',agg_spei),('ETd',agg_etd)):
        for oc in ('PDO','IOD','ENSO'):
            g=tc[oc].values.astype(float)
            c0=couple(agg,g,0)                 # locked: lag-0
            cb=best_lag_scan(agg,g)            # exploratory: best lag
            rec=dict(scale=scale,unit=name,n_stns=len(stns),index=idx_tag,ocean=oc)
            if c0:
                rec.update(r0=c0['r'],p0=c0['p'],neff0=c0['neff'],
                           ci0=c0['ci'],pass0=c0['passes'])
            if cb:
                rec.update(r_best=cb['r'],lag_best=cb['lag'],p_best=cb['p'],pass_best=cb['passes'])
            rows.append(rec)
A=pd.DataFrame(rows)

# ---------- artifact gate for any lag-0 PASS (PDO/IOD only) ------------------
def member_signdecomp(stns, idx_raw, oc, lag, agg_sign):
    g=tc[oc].values.astype(float); members=[]
    for s in stns:
        x=deseason(idx_raw[[s]])[s].values.astype(float)
        xv,gv=align(x,g,lag)
        if len(xv)<MIN_PAIRS or xv.std()==0: continue
        members.append((s,np.corrcoef(xv,gv)[0,1]))
    if not members: return None
    rs=np.array([m[1] for m in members])
    frac=np.mean(np.sign(rs)==agg_sign)
    # drop 2 stations with largest |r| in agg sign direction, recompute aggregate
    contrib=sorted(members,key=lambda m:-(m[1]*agg_sign))
    drop={contrib[0][0],contrib[1][0]} if len(contrib)>=2 else set()
    keep=[s for s in stns if s not in drop]
    raw=idx_raw[keep].mean(axis=1).to_frame('v'); aggk=deseason(raw)['v'].values
    ck=couple(aggk,g,lag)
    return dict(n_members=len(members),frac_same_sign=frac,
                survives_drop2=(ck['passes'] if ck else False),
                r_after_drop=(ck['r'] if ck else np.nan))

print("\n"+"="*72); print("LOCKED BAR — lag-0, Bretherton p<0.05, block-bootstrap CI excludes 0")
print("="*72)
show=A[A.ocean.isin(['PDO','IOD'])].copy()
def fmt(r):
    ci=r.get('ci0',(np.nan,np.nan))
    return (f"  [{r['scale']:8s}] {r['unit']:11s} n={r['n_stns']:3d} {r['index']:4s}~{r['ocean']:3s} "
            f"r0={r.get('r0',np.nan):+.3f} p={r.get('p0',np.nan):.3g} "
            f"CI[{ci[0]:+.3f},{ci[1]:+.3f}] neff={r.get('neff0',np.nan):.0f} "
            f"{'PASS' if r.get('pass0') else '·'}")
for _,r in show.iterrows(): print(fmt(r))

passed=show[show.pass0==True]
print("\n"+"-"*72)
print(f"Units clearing the locked bar (PDO/IOD, lag-0): {len(passed)}")
print("-"*72)
if len(passed)==0:
    print("None. -> Per the pre-registered decision: teleconnection dominance is")
    print("a RULED-OUT NULL at sub-basin and cluster scale. Write the eco-")
    print("hydrological partitioning paper; report this audit as the null result.")
else:
    raw_lookup={'SPEI':SPEI_raw,'ETd':ETD_raw}
    for _,r in passed.iterrows():
        stns=[s for s in stns_for(REACH4.get(r['unit'],REACH3.get(r['unit'],
              [int(r['unit'][2:])] if r['unit'].startswith('SB') else [])))
              if s in raw_lookup[r['index']].columns]
        dec=member_signdecomp(stns, raw_lookup[r['index']], r['ocean'], 0, np.sign(r['r0']))
        verdict="REAL regional signal" if (dec and dec['frac_same_sign']>=SIGN_AGREE_MIN
                 and dec['survives_drop2']) else "OUTLIER-DRIVEN / artifact"
        print(f"\n  {r['unit']} {r['index']}~{r['ocean']}: r0={r['r0']:+.3f}")
        if dec:
            print(f"    member sign agreement : {dec['frac_same_sign']:.0%} "
                  f"({'>=' if dec['frac_same_sign']>=SIGN_AGREE_MIN else '<'}{SIGN_AGREE_MIN:.0%})")
            print(f"    survives dropping top-2: {dec['survives_drop2']} "
                  f"(r after drop={dec['r_after_drop']:+.3f})")
        print(f"    --> {verdict}")

# exploratory best-lag view (flagged)
print("\n"+"-"*72)
print("EXPLORATORY (best-lag scan, selection-inflated — NOT the locked bar):")
eb=show[show.get('pass_best')==True]
print(f"  units 'passing' under best-lag scan: {len(eb)} "
      f"(treat as hypothesis-generating only)")

import os; os.makedirs('/content/drive/MyDrive/TGRS_Study',exist_ok=True)
A.to_csv('/content/drive/MyDrive/TGRS_Study/subbasin_cluster_audit.csv',index=False)
print("\nSaved -> /content/drive/MyDrive/TGRS_Study/subbasin_cluster_audit.csv")

# %% [notebook cell 13]
# ============================================================================
# ETdef–PDO decomposition: does the SB89/SB100 coupling enter via atmospheric
# DEMAND (PET) or via ACTUAL ET / water SUPPLY (management-sensitive)?
#
# Identity (on 3-month-accumulated, deseasonalised anomalies):
#   ETdef = PET - AET   =>   Cov(ETdef,PDO) = Cov(PET,PDO) - Cov(AET,PDO)
#   AET   = Et + Eb (+Ei...) => Cov(AET,PDO) = Cov(Et,PDO)+Cov(Eb,PDO)+...
# Supply-driven (AET term dominant, esp. Eb) supports management-fingerprint.
# Demand-driven (PET term dominant) supports a climate pathway.
#
# Inputs auto-located under /content/drive/MyDrive:
#   ETdef_indices_308stations_monthly.csv   PET_BMA_308stations_monthly.csv
#   GLEAM_v4.3a_components_308stations_monthly.csv   teleconnection_monthly.csv
#   station_subbasin_map.csv (or phase7_feature_table.csv)
# ============================================================================
import glob, warnings, numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')
rng = np.random.default_rng(42)
FOCUS = [89, 100]                      # the two surviving sub-basins
ACCUM = 3                              # months, to match ETdef-3

def find(name):
    h = glob.glob(f'/content/drive/MyDrive/**/{name}', recursive=True)
    assert h, f"{name} not found"; return h[0]

def load_long(path, value_candidates, return_all=False):
    df = pd.read_csv(path); df.columns=[c.strip() for c in df.columns]
    low={c.lower():c for c in df.columns}
    df['_t']=pd.PeriodIndex(year=df[low['year']].astype(int),
                            month=df[low['month']].astype(int), freq='M')
    idcol=next((low[c] for c in ('stationid','station_id','station','id') if c in low),None)
    if return_all:                     # return df + idcol for multi-column (GLEAM)
        return df, idcol
    val=None
    for cand in value_candidates:
        if cand.lower() in low: val=low[cand.lower()]; break
    if val is None:
        for cand in value_candidates:
            hit=next((c for c in df.columns if cand.lower() in c.lower()),None)
            if hit: val=hit; break
    assert idcol and val, f"{path}: id={idcol} val={val}; cols={list(df.columns)}"
    w=df.pivot_table(index='_t', columns=idcol, values=val)
    w.columns=[str(int(c)) for c in w.columns]
    print(f"  {path.split('/')[-1]}: val='{val}', {w.shape[1]} stns x {w.shape[0]} mo")
    return w

def pivot_col(df, idcol, col):
    w=df.pivot_table(index='_t', columns=idcol, values=col)
    w.columns=[str(int(c)) for c in w.columns]; return w

# ---- load ------------------------------------------------------------------
print("Loading…")
ETD = load_long(find('ETdef_indices_308stations_monthly.csv'), ['ETdef_3','ETdef3','etdef'])
PET = load_long(find('PET_BMA_308stations_monthly.csv'),
                ['PET_BMA','PET_BMA_mm','PET_mm','PET','pet'])

tc = pd.read_csv(find('teleconnection_monthly.csv')); tc.columns=[c.strip() for c in tc.columns]
tc['_t']=pd.PeriodIndex(year=tc['Year'].astype(int),month=tc['Month'].astype(int),freq='M')
PDO = tc.set_index('_t')['PDO']

# ---- GLEAM components: print columns, map components -----------------------
gdf, gid = load_long(find('GLEAM_v4.3a_components_308stations_monthly.csv'), [], return_all=True)
meta = {'year','month','stationid','station_id','station','id','latitude','longitude',
        'lat','lon','elevation_m','_t','sub_basin_id','subbasin_id'}
gcols=[c for c in gdf.columns if c.lower() not in meta]
print(f"\n  >>> GLEAM component columns found: {gcols}")
print("  >>> CHECK this mapping; edit GMAP if names differ <<<")

def pick(cols, *keys):
    for k in keys:
        hit=next((c for c in cols if k.lower() in c.lower()), None)
        if hit: return hit
    return None
GMAP = {
 'AET_total': pick(gcols,'E_a','et_total','aet','actualet','_e_','evap_total') or pick(gcols,'E'),
 'Et':        pick(gcols,'transp','_et','et_t','e_t'),     # transpiration
 'Eb':        pick(gcols,'bare','soil_evap','e_b','_eb','es_'),  # bare-soil evaporation
 'Ei':        pick(gcols,'intercept','e_i','_ei'),
}
print("  >>> GMAP =", GMAP)
GLEAM = {k: pivot_col(gdf, gid, v) for k,v in GMAP.items() if v is not None}
# AET total: use mapped total if present, else sum available flux components
if 'AET_total' not in GLEAM or GLEAM.get('AET_total') is None:
    parts=[GLEAM[k] for k in ('Et','Eb','Ei') if k in GLEAM]
    assert parts, "No AET total and no components to sum — paste GLEAM columns."
    GLEAM['AET_total']=sum(parts); print("  AET_total = sum of components")

# ---- station -> subbasin ----------------------------------------------------
try:
    m=pd.read_csv(find('station_subbasin_map.csv')); m.columns=[c.strip() for c in m.columns]
    low={c.lower():c for c in m.columns}
    sid=next(low[c] for c in low if 'station' in c or c=='id')
    bid=next(low[c] for c in low if 'sub' in c or 'basin' in c)
    S2B={str(int(s)):int(b) for s,b in zip(m[sid],m[bid]) if pd.notna(b)}
except Exception:
    ft=pd.read_csv(find('phase7_feature_table.csv')); ft.columns=[c.strip() for c in ft.columns]
    S2B={str(int(s)):int(b) for s,b in zip(ft['station_id'],ft['sub_basin_id']) if pd.notna(b)}
print(f"  map: {len(S2B)} stations, basins {sorted(set(S2B.values()))}")

# ---- helpers: subbasin mean, 3-mo accumulate, deseasonalise ----------------
grid=pd.period_range('1981-01','2020-12',freq='M')
def sb_mean(wide, basin):
    stns=[s for s,b in S2B.items() if b==basin and s in wide.columns]
    return wide[stns].reindex(grid).mean(axis=1) if stns else None
def accum(s):  return s.rolling(ACCUM, min_periods=ACCUM).sum()
def deseason(s):
    s=s.copy(); idx=s.index.month
    return s.groupby(idx).transform(lambda x: x-x.mean())
def lag1(a): a=a[~np.isnan(a)]; return np.corrcoef(a[:-1],a[1:])[0,1] if len(a)>3 else 0.0

def stat(x, g):
    m=~(np.isnan(x)|np.isnan(g)); x,g=x[m],g[m]
    if len(x)<60 or x.std()==0 or g.std()==0: return None
    r=np.corrcoef(x,g)[0,1]
    ne=len(x)*(1-lag1(x)*lag1(g))/(1+lag1(x)*lag1(g)); ne=max(3,min(ne,len(x)))
    t=r*np.sqrt((ne-2)/max(1-r**2,1e-9)); p=2*stats.t.sf(abs(t),ne-2)
    L=max(3,int(round(len(x)**(1/3)*1.5))); nb=int(np.ceil(len(x)/L)); bs=[]
    for _ in range(3000):
        st=rng.integers(0,len(x),nb); idx=np.concatenate([np.arange(s,s+L) for s in st])%len(x); idx=idx[:len(x)]
        xb,gb=x[idx],g[idx]
        if xb.std()>0 and gb.std()>0: bs.append(np.corrcoef(xb,gb)[0,1])
    lo,hi=np.percentile(bs,[2.5,97.5]) if bs else (np.nan,np.nan)
    cov=np.cov(x,g)[0,1]
    return dict(r=r,p=p,neff=ne,lo=lo,hi=hi,cov=cov,n=len(x))

pdo_anom = deseason(PDO.reindex(grid)).values

# ---- run per sub-basin ------------------------------------------------------
def series_anoms(basin):
    out={}
    etd_obs = deseason(sb_mean(ETD,basin)).values           # observed ETdef-3 (already an index)
    out['ETdef3_obs']=etd_obs
    pet = deseason(accum(sb_mean(PET,basin))).values
    aet = deseason(accum(sb_mean(GLEAM['AET_total'],basin))).values
    out['PET']=pet; out['AET']=aet; out['Deficit(PET-AET)']=pet-aet
    for k in ('Et','Eb'):
        if k in GLEAM:
            out[k]=deseason(accum(sb_mean(GLEAM[k],basin))).values
    return out

print("\n"+"="*74); print("ETdef–PDO DECOMPOSITION (lag-0, 3-mo accum, deseasonalised)")
print("="*74)
all_basins=sorted(set(S2B.values()))
for basin in FOCUS + [b for b in all_basins if b not in FOCUS]:
    A=series_anoms(basin)
    tag = "  <<< SURVIVOR" if basin in FOCUS else ""
    print(f"\n--- SB{basin}{tag} ---")
    print(f"  {'term':18s} {'r(PDO)':>8} {'p':>8} {'95% CI':>20} {'cov':>10}")
    refs={}
    for name in ['ETdef3_obs','Deficit(PET-AET)','PET','AET','Et','Eb']:
        if name not in A: continue
        s=stat(A[name], pdo_anom)
        if s is None: print(f"  {name:18s}   (insufficient)"); continue
        refs[name]=s
        sig='*' if s['p']<0.05 and (s['lo']>0 or s['hi']<0) else ' '
        print(f"  {name:18s} {s['r']:+8.3f} {s['p']:8.3g} "
              f"[{s['lo']:+.3f},{s['hi']:+.3f}]{sig} {s['cov']:+10.4f}")
    # covariance decomposition of the deficit
    if 'Deficit(PET-AET)' in refs:
        cD=refs['Deficit(PET-AET)']['cov']
        cPET=refs.get('PET',{}).get('cov',np.nan)
        cAET=refs.get('AET',{}).get('cov',np.nan)
        if np.isfinite(cPET) and np.isfinite(cAET) and abs(cD)>1e-9:
            print(f"    Cov(Deficit,PDO)={cD:+.4f}  =  +Cov(PET,PDO)={cPET:+.4f}  "
                  f"-Cov(AET,PDO)={-cAET:+.4f}")
            demand_share = cPET/cD; supply_share = (-cAET)/cD
            print(f"    -> demand(PET) share = {demand_share:+.0%} | "
                  f"supply(AET) share = {supply_share:+.0%}")
            verdict=("SUPPLY/management-leaning" if abs(cAET)>abs(cPET)
                     else "DEMAND/climate-leaning")
            if 'Eb' in refs and 'Et' in refs and abs(cAET)>abs(cPET):
                eb_part=refs['Eb']['cov']; et_part=refs['Et']['cov']
                dom='soil-evaporation' if abs(eb_part)>abs(et_part) else 'transpiration'
                verdict+=f" (AET dominated by {dom})"
            print(f"    => {verdict}")

print("""
========================= READ THIS =========================
- 'ETdef3_obs' vs 'Deficit(PET-AET)': if their r(PDO) values are close, the
  PET/AET reconstruction faithfully represents your ETdef -> decomposition valid.
  If they diverge, the GMAP mapping or your ETdef definition differs: paste the
  GLEAM columns + how ETdef was built, and I'll fix the mapping.
- Negative ETdef~PDO arises from PET FALLING or AET RISING with PDO. The
  covariance split says which. AET-dominated (esp. soil-evaporation) in the two
  most-managed reaches supports the water-management-fingerprint reading.
- Still correlational and small-effect; this characterises the surviving signal,
  it does not resurrect a basin-wide teleconnection mechanism.
=============================================================
""")

# %% [notebook cell 14]
# ============================================================================
# Per-sub-basin irrigated-fraction ranking (LGRIP30, 30 m) — adjudicates the
# management-fingerprint reading: are SB89/SB100 the MOST IRRIGATED units,
# distinct from merely the most ARID (which would be eco-hydrology, not mgmt)?
# ============================================================================
import ee, pandas as pd, numpy as np
ee.Authenticate()                       # first run only
ee.Initialize(project='zz-sheheryarkhan00')   # <-- set your project id

# ---- 1. Sub-basin polygons as a GEE asset ----------------------------------
# One-time upload: in the GEE Code Editor -> Assets -> New -> Shapefile,
# upload your sub-basin .shp (+.shx/.dbf/.prj), then paste its asset id here.
SUBBASINS = ee.FeatureCollection('projects/zz-sheheryarkhan00/assets/YRB_Subbasins')
SB_ID_PROP = 'SubBasin_ID'              # <-- the polygon attribute holding 59,60,...
# (if your shapefile uses WRRNM or another field, set it here)

# ---- 2. LGRIP30 -------------------------------------------------------------
lgrip = ee.ImageCollection('projects/sat-io/open-datasets/GFSAD/LGRIP30').mosaic()

# >>> VERIFY class codes. Per LGRIP30 docs the layer encodes non-cropland,
#     irrigated cropland, rainfed cropland, and water. Confirm the integers
#     from the histogram printed below (and the sat-io catalog page) and set:
IRRIGATED_CLASSES = [2]      # <-- adjust after checking histogram
RAINFED_CLASSES   = [3]      # <-- adjust after checking histogram
# (catalog: gee-community-catalog.org/projects/lgrip30/ has the authoritative table)

# ---- 3. Class histogram per sub-basin (run once to verify codes) -----------
hist = lgrip.reduceRegions(
    collection=SUBBASINS, reducer=ee.Reducer.frequencyHistogram(), scale=30)
print("Class histograms per sub-basin (verify irrigated/rainfed codes):")
for f in hist.getInfo()['features']:
    p = f['properties']
    print(f"  SB{p.get(SB_ID_PROP)}: {p.get('histogram')}")

# ---- 4. Area accounting -----------------------------------------------------
px_area = ee.Image.pixelArea()                       # m^2 per pixel
irr  = lgrip.remap(IRRIGATED_CLASSES, [1]*len(IRRIGATED_CLASSES), 0).rename('irr')
rain = lgrip.remap(RAINFED_CLASSES,   [1]*len(RAINFED_CLASSES),   0).rename('rain')
area_stack = (px_area.rename('total')
              .addBands(px_area.updateMask(irr).rename('irr_area'))
              .addBands(px_area.updateMask(rain).rename('rain_area')))

stats = area_stack.reduceRegions(
    collection=SUBBASINS, reducer=ee.Reducer.sum(), scale=30, crs='EPSG:4326')
feats = stats.getInfo()['features']
basin_order = [59, 60, 75, 78, 89, 100, 108, 109]   # same order as histogram output
rows = []
for sb, f in zip(basin_order, feats):
    p = f['properties']
    tot  = p.get('total', 0) or 0
    irra = p.get('irr_area', 0) or 0
    rana = p.get('rain_area', 0) or 0
    crop = irra + rana
    rows.append(dict(
        SubBasin = sb,
        irrig_frac_of_basin = irra/tot if tot else np.nan,
        crop_frac_of_basin  = crop/tot if tot else np.nan,
        irrig_share_of_crop = irra/crop if crop else np.nan,
    ))
df = pd.DataFrame(rows).set_index('SubBasin').sort_index()

# ---- 5. Bring in aridity (per-station -> sub-basin mean) -------------------
import glob
ft = pd.read_csv(glob.glob('/content/drive/MyDrive/**/phase7_feature_table.csv',
                           recursive=True)[0]); ft.columns=[c.strip() for c in ft.columns]
arid = ft.groupby('sub_basin_id')['Aridity'].mean()
df['Aridity_mean'] = arid

# ---- 6. Juxtapose with the ETdef~PDO audit result --------------------------
# (lag-0 r0 and verdict from the locked-bar sub-basin audit)
audit = {59:(-0.101,'ns'),60:(-0.214,'artifact'),75:(-0.031,'ns'),78:(+0.136,'ns'),
         89:(-0.244,'REAL'),100:(-0.257,'REAL'),108:(-0.158,'ns'),109:(+0.206,'ns')}
df['ETd_PDO_r0']  = [audit.get(i,(np.nan,''))[0] for i in df.index]
df['ETd_PDO_flag']= [audit.get(i,(np.nan,''))[1] for i in df.index]

# ---- 7. Ranked output -------------------------------------------------------
df['irrig_rank'] = df['irrig_frac_of_basin'].rank(ascending=False).astype('Int64')
df['arid_rank']  = df['Aridity_mean'].rank(ascending=False).astype('Int64')  # check direction!
df = df.sort_values('irrig_frac_of_basin', ascending=False)
pd.set_option('display.width', 140, 'display.max_columns', 20)
print("\n"+"="*78)
print("IRRIGATION vs ARIDITY vs ETdef-PDO COUPLING  (8 sub-basins)")
print("="*78)
print(df.round(3).to_string())

print(f"\nSpearman(irrig_frac, |ETd_PDO_r0|): "
      f"{df['irrig_frac_of_basin'].corr(df['ETd_PDO_r0'].abs(), method='spearman'):+.2f}")
print(f"Spearman(aridity,    |ETd_PDO_r0|): "
      f"{df['Aridity_mean'].corr(df['ETd_PDO_r0'].abs(), method='spearman'):+.2f}")

df.to_csv('/content/drive/MyDrive/TGRS_Study/subbasin_irrigation_ranking.csv')
print("\nSaved -> /content/drive/MyDrive/TGRS_Study/subbasin_irrigation_ranking.csv")

# %% [notebook cell 15]
# ============================================================================
# Step: WorldCover weak labels (6 functional classes) + per-sub-basin
# composition + stratified validation points (Olofsson good-practice).
# Foundation for both the LULC map and the dual-backbone comparison.
# ============================================================================
import ee, pandas as pd, numpy as np
ee.Authenticate()                       # first run only
ee.Initialize(project='zz-sheheryarkhan00')   # <-- set your project id

# ---- 1. Sub-basin polygons as a GEE asset ----------------------------------
# One-time upload: in the GEE Code Editor -> Assets -> New -> Shapefile,
# upload your sub-basin .shp (+.shx/.dbf/.prj), then paste its asset id here.
SUBBASINS = ee.FeatureCollection('projects/zz-sheheryarkhan00/assets/YRB_Subbasins')
SB_ID_PROP = 'SubBasin_ID'              # <-- the polygon attribute holding 59,60,...

basin_order = [59, 60, 75, 78, 89, 100, 108, 109]   # collection order (from prior run)
region = SUBBASINS.geometry()

# ---- 1. WorldCover v200 (2021) -> 6 functional classes ---------------------
# WorldCover codes: 10 tree,20 shrub,30 grass,40 crop,50 built,60 bare,
#                   70 snow,80 water,90 wetland,95 mangrove,100 moss
# Functional: 1 Cropland, 2 Forest, 3 Grassland, 4 Bare/sparse, 5 Water, 6 Built
wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
FROM = [40, 10, 20, 30, 100, 60, 70, 80, 90, 95, 50]
TO   = [ 1,  2,  3,  3,   3,  4,  4,  5,  5,  5,  6]   # shrub+moss->Grassland; snow->Bare
func = wc.remap(FROM, TO, 0).rename('func').clip(region)
CLASS_NAMES = {1:'Cropland',2:'Forest',3:'Grassland',4:'Bare/sparse',5:'Water',6:'Built'}
# (shrub grouped with grassland for eco-hydrology; adjust TO if you prefer shrub->forest)

# ---- 2. Per-sub-basin functional composition (paper table + sanity check) --
px = ee.Image.pixelArea()
stack = px.rename('total')
for k in CLASS_NAMES:
    stack = stack.addBands(px.updateMask(func.eq(k)).rename(f'c{k}'))
comp = stack.reduceRegions(collection=SUBBASINS, reducer=ee.Reducer.sum(), scale=100,tileScale=4)

rows = []
for sb, f in zip(basin_order, comp.getInfo()['features']):
    p = f['properties']; tot = p.get('total', 0) or 1
    rows.append({'SubBasin': sb, **{CLASS_NAMES[k]: (p.get(f'c{k}',0) or 0)/tot
                                    for k in CLASS_NAMES}})
df = pd.DataFrame(rows).set_index('SubBasin').sort_index()
pd.set_option('display.width', 140, 'display.max_columns', 12)
print("Functional land-cover fraction per sub-basin (WorldCover 2021):")
print(df.round(3).to_string())
df.to_csv('/content/drive/MyDrive/TGRS_Study/subbasin_functional_composition.csv')

# ---- 3. Stratified validation points (independent accuracy assessment) -----
PER_CLASS = 75          # ~450 total; raise rare-class floor if needed
pts = func.stratifiedSample(
    numPoints=PER_CLASS, classBand='func', region=region, scale=10,
    seed=42, geometries=True, classValues=list(CLASS_NAMES),
    classPoints=[PER_CLASS]*len(CLASS_NAMES))
pts = pts.map(lambda ft: ft.set({
    'lon': ft.geometry().coordinates().get(0),
    'lat': ft.geometry().coordinates().get(1),
    'wc_class': ee.Number(ft.get('func')),
    'manual_label': ''}))          # <-- you fill this in by eye, blind to wc_class

ee.batch.Export.table.toDrive(
    collection=pts.select(['lon','lat','wc_class','manual_label']),
    description='YRB_validation_points', folder='TGRS_Study',
    fileNamePrefix='YRB_validation_points', fileFormat='CSV').start()
print(f"\nExporting ~{PER_CLASS*len(CLASS_NAMES)} stratified validation points "
      f"-> Drive/TGRS_Study/YRB_validation_points.csv (check GEE Tasks tab)")

# ---- 4. (LATER, when fine-tuning) export the weak-label raster -------------
# ee.batch.Export.image.toDrive(image=func, description='YRB_worldcover_func',
#   folder='TGRS_Study', region=region, scale=10, maxPixels=1e13).start()

# %% [notebook cell 16]
# ============================================================================
# Data-prep export: co-registered 30 m Sentinel-2 (HLS band order) + 6-class
# WorldCover weak-label raster, per sub-basin. Foundation for the dual-backbone
# training. Runs in PARALLEL with your manual validation-point interpretation.
# ============================================================================
import ee, pandas as pd, numpy as np
ee.Authenticate()
ee.Initialize(project='zz-sheheryarkhan00')

SUBBASINS = ee.FeatureCollection('projects/zz-sheheryarkhan00/assets/YRB_Subbasins')
basin_order = [60]
YEAR = 2021                      # match WorldCover 2021
EXPORT_CRS = 'EPSG:32649'   # WGS84 / UTM 48N — metric, so scale=30 = true 30 m
SCALE = 30

# ---- Sentinel-2 SR, cloud-masked median, HLS band order --------------------
def s2_composite(geom):
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(geom)
            .filterDate(f'{YEAR}-01-01', f'{YEAR}-12-31'))
    csp = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    s2 = s2.linkCollection(csp, ['cs']).map(
        lambda im: im.updateMask(im.select('cs').gte(0.6)))   # Cloud Score+ mask
    comp = s2.median()
    # B2,B3,B4,B8A,B11,B12 -> Blue,Green,Red,NIR_narrow,SWIR1,SWIR2  (Prithvi/HLS order)
    return (comp.select(['B2','B3','B4','B8A','B11','B12'],
                        ['BLUE','GREEN','RED','NIR_NARROW','SWIR1','SWIR2'])
                .toInt16().clip(geom))                        # int16 reflectance*1; scale in loader

# ---- WorldCover -> 6 functional classes (same mapping as composition step) -
wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
FROM = [40, 10, 20, 30, 100, 60, 70, 80, 90, 95, 50]
TO   = [ 1,  2,  3,  3,   3,  4,  4,  5,  5,  5,  6]
func = wc.remap(FROM, TO, 0).rename('func').toByte()

# ---- per-sub-basin export (S2 + label, identical grid) ---------------------
basin_order = [59, 60, 75, 78, 89, 100, 108, 109]   # keep full list
TARGET = ee.Projection('EPSG:32649').atScale(30)
feats = SUBBASINS.toList(SUBBASINS.size())

for i, sb in enumerate(basin_order):
    if sb != 60:        # ← test only SB60, skip the rest
        continue
    geom = ee.Feature(feats.get(i)).geometry()
    img  = s2_composite(geom).reproject(TARGET)
    lbl  = func.clip(geom).reproject(TARGET)
    common = dict(region=geom, scale=30, crs='EPSG:32649',
                  maxPixels=1e13, folder='TGRS_LULC')
    ee.batch.Export.image.toDrive(image=img, description=f'S2_30m_SB{sb}',
        fileNamePrefix=f'S2_30m_SB{sb}', **common).start()
    ee.batch.Export.image.toDrive(image=lbl, description=f'LABEL_30m_SB{sb}',
        fileNamePrefix=f'LABEL_30m_SB{sb}', **common).start()
    print(f'SB{sb}: queued at true 30 m')

print("\n16 export tasks queued -> Drive/TGRS_LULC/ (watch the GEE Tasks tab).")
print("S2 and LABEL for each basin share crs+scale+region, so they co-register.")

# %% [notebook cell 17]
import ee
ee.Initialize(project='zz-sheheryarkhan00')

SUBBASINS = ee.FeatureCollection('projects/zz-sheheryarkhan00/assets/YRB_Subbasins')
feats = SUBBASINS.toList(SUBBASINS.size())
geom = ee.Feature(feats.get(1)).geometry()     # SB60 is at position 1

# ---- LABEL: WorldCover -> functional, majority-aggregate 9.3 m -> 30 m -----
wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
FROM=[40,10,20,30,100,60,70,80,90,95,50]; TO=[1,2,3,3,3,4,4,5,5,5,6]
func = wc.remap(FROM, TO, 0).rename('func')
lbl30 = (func
    .reduceResolution(reducer=ee.Reducer.mode(), maxPixels=1500)   # majority vote
    .reproject(crs='EPSG:32649', scale=30)
    .toByte())

# ---- S2: cloud-masked median, reproject to 30 m ----------------------------
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(geom).filterDate('2021-01-01','2021-12-31'))
csp = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
s2 = s2.linkCollection(csp,['cs']).map(lambda im: im.updateMask(im.select('cs').gte(0.6)))
comp = (s2.median().select(['B2','B3','B4','B8A','B11','B12'],
                           ['BLUE','GREEN','RED','NIR_NARROW','SWIR1','SWIR2']))
img30 = comp.reproject(crs='EPSG:32649', scale=30).toInt16()

# ---- THE KEY CHECK: pixel size server-side, no file needed -----------------
print("LABEL nominal scale (m):", lbl30.projection().nominalScale().getInfo())
print("S2    nominal scale (m):", img30.select(0).projection().nominalScale().getInfo())
print("--> both must read ~30. If so, resolution is fixed regardless of Drive.")

# %% [notebook cell 18]
for t in ee.batch.Task.list()[:16]:
    if t.status()['state'] in ('READY','RUNNING'):
        t.cancel()
print("cancelled pending tasks")

# %% [notebook cell 19]
import ee
ee.Initialize(project='zz-sheheryarkhan00')

SUBBASINS = ee.FeatureCollection('projects/zz-sheheryarkhan00/assets/YRB_Subbasins')
basin_order = [59, 60, 75, 78, 89, 100, 108, 109]
feats = SUBBASINS.toList(SUBBASINS.size())

wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
FROM=[40,10,20,30,100,60,70,80,90,95,50]; TO=[1,2,3,3,3,4,4,5,5,5,6]
func = wc.remap(FROM, TO, 0).rename('func')

# 30 m affine transform in EPSG:32649: [xscale, 0, xmin, 0, yscale, ymax]
CRS = 'EPSG:32649'
CRS_TRANSFORM = [30, 0, 0, 0, -30, 0]      # 30 m pixels, origin snapped to grid

def s2_30(geom):
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(geom).filterDate('2021-01-01','2021-12-31'))
    csp = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    s2 = s2.linkCollection(csp,['cs']).map(
        lambda im: im.updateMask(im.select('cs').gte(0.6)))
    return (s2.median().select(['B2','B3','B4','B8A','B11','B12'],
                ['BLUE','GREEN','RED','NIR_NARROW','SWIR1','SWIR2']).toInt16())

lbl = (func.reduceResolution(reducer=ee.Reducer.mode(), maxPixels=1500).toByte())

for i, sb in enumerate(basin_order):
    if sb != 60:            # ← test SB60 only
        continue
    geom = ee.Feature(feats.get(i)).geometry()
    common = dict(region=geom, crs=CRS, crsTransform=CRS_TRANSFORM,
                  maxPixels=1e13, folder='TGRS_LULC')
    ee.batch.Export.image.toDrive(image=s2_30(geom).clip(geom),
        description=f'S2_30m_SB{sb}', fileNamePrefix=f'S2_30m_SB{sb}', **common).start()
    ee.batch.Export.image.toDrive(image=lbl.clip(geom),
        description=f'LABEL_30m_SB{sb}', fileNamePrefix=f'LABEL_30m_SB{sb}', **common).start()
    print(f'SB{sb}: queued with forced 30 m crsTransform')

print("\n16 tasks queued. crsTransform forces the grid — scale cannot be overridden now.")

# %% [notebook cell 20]
import ee
ee.Initialize(project='zz-sheheryarkhan00')

SUBBASINS = ee.FeatureCollection('projects/zz-sheheryarkhan00/assets/YRB_Subbasins')
print("feature count:", SUBBASINS.size().getInfo())
print("property names:", SUBBASINS.first().propertyNames().getInfo())

# area of each feature in km^2, with whatever id-like properties exist
def add_area(f):
    return f.set('area_km2', f.geometry().area(1).divide(1e6).round())
fc = SUBBASINS.map(add_area)
print("\nper-feature areas (km^2):")
for f in fc.getInfo()['features']:
    p = f['properties']
    print({k: p[k] for k in p if k != 'system:index'})

# %% [notebook cell 21]
import ee
ee.Initialize(project='zz-sheheryarkhan00')

SB = ee.FeatureCollection('projects/zz-sheheryarkhan00/assets/YRB_Subbasins')

wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
FROM=[40,10,20,30,100,60,70,80,90,95,50]; TO=[1,2,3,3,3,4,4,5,5,5,6]
func = wc.remap(FROM, TO, 0).rename('func')
lbl  = func.reduceResolution(ee.Reducer.mode(), maxPixels=1500).reproject('EPSG:32649',None,30).toByte()

def s2_30(geom):
    s2=(ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(geom).filterDate('2021-01-01','2021-12-31'))
    csp=ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    s2=s2.linkCollection(csp,['cs']).map(lambda im: im.updateMask(im.select('cs').gte(0.6)))
    return (s2.median().select(['B2','B3','B4','B8A','B11','B12'],
            ['BLUE','GREEN','RED','NIR_NARROW','SWIR1','SWIR2'])
            .reproject('EPSG:32649',None,30).toInt16())

# stack S2(6) + label(1) = 7 bands, export random 224x224 patches per basin
stack = s2_30(SB.geometry()).addBands(lbl).clip(SB.geometry())

PATCH, N_PER_BASIN = 224, 60     # 60 patches/basin x 8 = 480 tiles
for sb_id in [59,60,75,78,89,100,108,109]:
    geom = SB.filter(ee.Filter.eq('ID', sb_id)).geometry()      # ← select by real ID
    pts  = ee.FeatureCollection.randomPoints(geom, N_PER_BASIN, seed=sb_id)
    # export the stacked image, sampled at patch tiles around each point
    task = ee.batch.Export.image.toDrive(
        image=stack.float(),
        description=f'patches_SB{sb_id}',
        fileNamePrefix=f'patches_SB{sb_id}',
        folder='TGRS_LULC_patches',
        region=geom, scale=30, crs='EPSG:32649',
        maxPixels=1e13)
    # NOTE: see explanation below — patch export needs a different call
    print(f'SB{sb_id}: {geom.area(1).divide(1e6).round().getInfo()} km^2 selected')

# %% [notebook cell 22]
import ee
ee.Initialize(project='zz-sheheryarkhan00')

SB = ee.FeatureCollection('projects/zz-sheheryarkhan00/assets/YRB_Subbasins')
TEST_ONLY = True          # ← True = SB100 only; flip to False to export all 8

# ---- label: WorldCover -> 6 functional classes, majority-aggregate to 30 m -
wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
FROM=[40,10,20,30,100,60,70,80,90,95,50]; TO=[1,2,3,3,3,4,4,5,5,5,6]
lbl = (wc.remap(FROM, TO, 0).rename('func')
         .reduceResolution(ee.Reducer.mode(), maxPixels=1500)
         .reproject('EPSG:32649', None, 30).toByte())

# ---- S2: cloud-masked median, HLS band order, 30 m -------------------------
def s2_30(geom):
    s2=(ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(geom).filterDate('2021-01-01','2021-12-31'))
    csp=ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    s2=s2.linkCollection(csp,['cs']).map(lambda im: im.updateMask(im.select('cs').gte(0.6)))
    return (s2.median().select(['B2','B3','B4','B8A','B11','B12'],
            ['BLUE','GREEN','RED','NIR_NARROW','SWIR1','SWIR2'])
            .reproject('EPSG:32649', None, 30).toInt16())

ids = [100] if TEST_ONLY else [59,60,75,78,89,100,108,109]
for sb_id in ids:
    geom = SB.filter(ee.Filter.eq('ID', sb_id)).geometry()       # ← real ID field
    km2  = geom.area(1).divide(1e6).round().getInfo()
    common = dict(region=geom, scale=30, crs='EPSG:32649',
                  maxPixels=1e13, folder='TGRS_LULC')
    ee.batch.Export.image.toDrive(image=s2_30(geom).clip(geom),
        description=f'S2_30m_SB{sb_id}', fileNamePrefix=f'S2_30m_SB{sb_id}', **common).start()
    ee.batch.Export.image.toDrive(image=lbl.clip(geom),
        description=f'LABEL_30m_SB{sb_id}', fileNamePrefix=f'LABEL_30m_SB{sb_id}', **common).start()
    print(f'SB{sb_id}: {km2} km^2 queued at 30 m')

print("\nDone. With TEST_ONLY=True this queued only SB100 (2 tasks).")

# %% [notebook cell 23]
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
import rasterio, glob
for f in sorted(glob.glob('/content/drive/MyDrive/TGRS_LULC/*SB100*.tif')):
    with rasterio.open(f) as src:
        print(f.split('/')[-1], '|', src.count, 'x', src.height, 'x', src.width, '|', src.dtypes[0])

# %% [notebook cell 24]
import rasterio
for f in ['LABEL_30m_SB100','S2_30m_SB100']:
    with rasterio.open(f'/content/drive/MyDrive/TGRS_LULC/{f}.tif') as src:
        print(f, '| shape:', src.count, 'x', src.height, 'x', src.width,
              '| dtype:', src.dtypes[0], '| crs:', src.crs)

# %% [notebook cell 25]
import rasterio, numpy as np
with rasterio.open('/content/drive/MyDrive/TGRS_LULC/LABEL_30m_SB100.tif') as src:
    a = src.read(1)
    valid = int((a != 0).sum())
    print(f"grid pixels : {a.size:,}")
    print(f"valid pixels: {valid:,}  ({100*valid/a.size:.0f}%)")
    print(f"implied area: {valid*900/1e6:,.0f} km^2   (SB100 known = 22,701)")
    u,c = np.unique(a, return_counts=True)
    print("class histogram:", dict(zip(u.tolist(), c.tolist())))

# %% [notebook cell 26]
import ee, pandas as pd, numpy as np
ee.Authenticate()
ee.Initialize(project='zz-sheheryarkhan00')

wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
print("WorldCover native scale (m):", wc.projection().nominalScale().getInfo())
print("WorldCover CRS:", wc.projection().crs().getInfo())

# %% [notebook cell 27]
# ============================================================================
# Stage 1 — Patch extractor: basin GeoTIFFs -> 224x224 training patches
# Run on KAGGLE after uploading the TGRS_LULC exports as a dataset.
# Produces patches.npz per spatial fold (sub-basin), rejecting nodata tiles.
# ============================================================================
import numpy as np, rasterio, glob, os, re
from rasterio.windows import Window
from collections import defaultdict

SRC = '/content/drive/MyDrive/TGRS_LULC'        # ← Drive instead of Kaggle
OUT = '/content/patches'; os.makedirs(OUT, exist_ok=True)
PATCH = 224
STRIDE = 224                  # non-overlapping; lower to 112 for 2x more patches
MIN_VALID = 0.30              # reject patches <30% labeled pixels
NODATA = 0

# group S2 + LABEL tiles by sub-basin (handles GEE's auto-split suffixes)
def sb_of(path):
    m = re.search(r'SB(\d+)', os.path.basename(path)); return int(m.group(1)) if m else None

s2_files  = sorted(glob.glob(f'{SRC}/*S2_30m_SB*.tif'))
lbl_files = sorted(glob.glob(f'{SRC}/*LABEL_30m_SB*.tif'))
print(f"found {len(s2_files)} S2 tiles, {len(lbl_files)} label tiles")

# pair S2<->label by matching sub-basin AND tile suffix
def key(path):  # SB id + tile offset suffix if present
    b = os.path.basename(path)
    sb = sb_of(b); suf = re.search(r'(\d{10}-\d{10})', b)
    return (sb, suf.group(1) if suf else '')

lbl_by_key = {key(f): f for f in lbl_files}

counts = defaultdict(int)
for s2f in s2_files:
    k = key(s2f)
    if k not in lbl_by_key:
        print(f"  no label match for {os.path.basename(s2f)} — skipping"); continue
    sb = k[0]
    with rasterio.open(s2f) as s2src, rasterio.open(lbl_by_key[k]) as lsrc:
        assert (s2src.height, s2src.width) == (lsrc.height, lsrc.width), \
            f"shape mismatch {k}"
        H, W = s2src.height, s2src.width
        X_list, Y_list = [], []
        for top in range(0, H - PATCH + 1, STRIDE):
            for left in range(0, W - PATCH + 1, STRIDE):
                win = Window(left, top, PATCH, PATCH)
                y = lsrc.read(1, window=win)
                if (y != NODATA).mean() < MIN_VALID:     # reject mostly-nodata
                    continue
                x = s2src.read(window=win).astype(np.int16)   # (6,224,224)
                X_list.append(x); Y_list.append(y.astype(np.uint8))
        if X_list:
            X = np.stack(X_list); Y = np.stack(Y_list)
            np.savez_compressed(f'{OUT}/patches_SB{sb}_{k[1] or "0"}.npz', X=X, y=Y)
            counts[sb] += len(X_list)
            print(f"  SB{sb} [{k[1] or 'single'}]: {len(X_list)} patches  X={X.shape}")

print("\nper-basin patch counts:", dict(sorted(counts.items())))
print("total patches:", sum(counts.values()))
print("Saved -> /kaggle/working/patches/  (one .npz per basin/tile)")

# %% [notebook cell 28]
import ee
ee.Authenticate()                          # ← add this; opens a sign-in link
ee.Initialize(project='zz-sheheryarkhan00')

SB = ee.FeatureCollection('projects/zz-sheheryarkhan00/assets/YRB_Subbasins')
TEST_ONLY = False          # ← now exports all 8

wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
FROM=[40,10,20,30,100,60,70,80,90,95,50]; TO=[1,2,3,3,3,4,4,5,5,5,6]
lbl = (wc.remap(FROM, TO, 0).rename('func')
         .reduceResolution(ee.Reducer.mode(), maxPixels=1500)
         .reproject('EPSG:32649', None, 30).toByte())

def s2_30(geom):
    s2=(ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(geom).filterDate('2021-01-01','2021-12-31'))
    csp=ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    s2=s2.linkCollection(csp,['cs']).map(lambda im: im.updateMask(im.select('cs').gte(0.6)))
    return (s2.median().select(['B2','B3','B4','B8A','B11','B12'],
            ['BLUE','GREEN','RED','NIR_NARROW','SWIR1','SWIR2'])
            .reproject('EPSG:32649', None, 30).toInt16())

ids = [100] if TEST_ONLY else [59,60,75,78,89,100,108,109]
for sb_id in ids:
    geom = SB.filter(ee.Filter.eq('ID', sb_id)).geometry()
    km2  = geom.area(1).divide(1e6).round().getInfo()
    common = dict(region=geom, scale=30, crs='EPSG:32649', maxPixels=1e13, folder='TGRS_LULC')
    ee.batch.Export.image.toDrive(image=s2_30(geom).clip(geom),
        description=f'S2_30m_SB{sb_id}', fileNamePrefix=f'S2_30m_SB{sb_id}', **common).start()
    ee.batch.Export.image.toDrive(image=lbl.clip(geom),
        description=f'LABEL_30m_SB{sb_id}', fileNamePrefix=f'LABEL_30m_SB{sb_id}', **common).start()
    print(f'SB{sb_id}: {km2} km^2 queued')

print("\nAll 8 queued. SB100 already done — it will just re-export.")

# %% [notebook cell 29]
import ee
for t in ee.batch.Task.list()[:16]:
    s = t.status()
    if 'SB75' in s['description'] or 'SB109' in s['description']:
        print(s['description'], '->', s['state'],
              '|', s.get('error_message', ''))

# %% [notebook cell 30]
import ee
ee.Authenticate(); ee.Initialize(project='zz-sheheryarkhan00')
SB = ee.FeatureCollection('projects/zz-sheheryarkhan00/assets/YRB_Subbasins')

def s2_30(geom):
    s2=(ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(geom).filterDate('2021-01-01','2021-12-31'))
    csp=ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    s2=s2.linkCollection(csp,['cs']).map(lambda im: im.updateMask(im.select('cs').gte(0.6)))
    return (s2.median().select(['B2','B3','B4','B8A','B11','B12'],
            ['BLUE','GREEN','RED','NIR_NARROW','SWIR1','SWIR2'])
            .reproject('EPSG:32649', None, 30).toInt16())

for sb_id in [75, 109]:                      # only the two that failed
    geom = SB.filter(ee.Filter.eq('ID', sb_id)).geometry()
    ee.batch.Export.image.toDrive(image=s2_30(geom).clip(geom),
        description=f'S2_30m_SB{sb_id}', fileNamePrefix=f'S2_30m_SB{sb_id}',
        folder='TGRS_LULC', region=geom, scale=30, crs='EPSG:32649',
        maxPixels=1e13).start()
    print(f'SB{sb_id} S2 re-queued')

# %% [notebook cell 31]
# ============================================================================
# COLAB: shrink the 4GB prediction-raster zip -> small zip for figure rendering
# Mounts Drive, unzips in place, downsamples ~8x (keeps georeferencing), re-zips.
# ============================================================================
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','rasterio'], check=False)
import glob, os, zipfile, numpy as np, rasterio
from rasterio.enums import Resampling

# ---- 1. locate the uploaded zip in the TGSR_Study folder -------------------
# adjust this glob if your Drive path differs (My Drive vs a shared drive)
cands = glob.glob('/content/drive/MyDrive/**/full prediction-raster set.zip', recursive=True)
assert cands, "zip not found - check the folder name/path under MyDrive"
ZIP = cands[0]
print("found zip:", ZIP, f"({os.path.getsize(ZIP)/1e9:.2f} GB)")

# ---- 2. unzip into local Colab disk (fast, not Drive) ----------------------
RAW = '/content/preds_raw'; os.makedirs(RAW, exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(RAW)
tifs = sorted(glob.glob(f'{RAW}/**/*.tif', recursive=True))
print(f"extracted {len(tifs)} rasters")
for t in tifs[:6]: print("   ", os.path.basename(t))

# ---- 3. downsample ~8x (mode for classes, average for entropy) -------------
DST = '/content/preds_small'; os.makedirs(DST, exist_ok=True)
F = 8
for f in tifs:
    name = os.path.basename(f); is_ent = 'entropy' in name
    with rasterio.open(f) as s:
        H, W = max(1, s.height//F), max(1, s.width//F)
        resamp = Resampling.average if is_ent else Resampling.mode
        a = s.read(1, out_shape=(H, W), resampling=resamp)
        if is_ent:
            a = np.clip(a/2*200, 0, 255).astype('uint8')   # 0..2 nats -> 0..200
        else:
            a = a.astype('uint8')
        tr = s.transform * s.transform.scale(s.width/W, s.height/H)
        prof = dict(driver='GTiff', height=H, width=W, count=1, dtype='uint8',
                    crs=s.crs, transform=tr, nodata=0, compress='lzw')
    with rasterio.open(f'{DST}/{name}', 'w', **prof) as d:
        d.write(a, 1)
    print('  ->', name, f'{W}x{H}')

# ---- 4. zip the small set back to Drive ------------------------------------
OUT_ZIP = '/content/drive/MyDrive/TGSR_Study/preds_small.zip'
os.makedirs(os.path.dirname(OUT_ZIP), exist_ok=True)
with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob(f'{DST}/*.tif')):
        z.write(f, os.path.basename(f))
print(f"\nDONE -> {OUT_ZIP}  ({os.path.getsize(OUT_ZIP)/1e6:.0f} MB)")
print("Upload confirmed in Drive; tell Claude the filename: preds_small.zip")

# %% [notebook cell 32]
# COLAB: produce validation_predictions.csv from the FULL-RES predictions already on Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
import subprocess, sys; subprocess.run([sys.executable,'-m','pip','install','-q','rasterio'],check=False)
import zipfile, glob, os, numpy as np, pandas as pd, rasterio
from rasterio.windows import Window
from pyproj import Transformer

# 1. locate + unzip the full prediction set (4 GB) into Colab local disk
ZIP=glob.glob('/content/drive/MyDrive/**/full prediction-raster set.zip',recursive=True)[0]
print('unzipping', ZIP); RAW='/content/preds_full'; os.makedirs(RAW,exist_ok=True)
with zipfile.ZipFile(ZIP) as z: z.extractall(RAW)

# 2. validation points — upload YRB_validation_points_clean_land_type.csv to Colab first,
#    or point this at its location on your Drive
df=pd.read_csv('/content/drive/MyDrive/TGRS_Study/YRB_validation_points_clean_land_type.csv')
LMAP={'Cropland':1,'Forest':2,'Grassland':3,'Bare land':4,'Water':5,'Built-up':6}
df['true']=df['land_type'].map(LMAP)

basins=[59,60,75,78,89,100,108,109]
def tiles(b,kind):
    return sorted(set(glob.glob(f'{RAW}/**/S2_30m_SB{b}_{kind}.tif',recursive=True)+
                      glob.glob(f'{RAW}/**/S2_30m_SB{b}-*_{kind}.tif',recursive=True)))
def sample(kind):
    out=np.zeros(len(df),int); bas=np.zeros(len(df),int)
    for b in basins:
        for tp in tiles(b,kind):
            with rasterio.open(tp) as s:
                tr=Transformer.from_crs('EPSG:4326',s.crs,always_xy=True)
                for i,r in df.iterrows():
                    if out[i]!=0: continue
                    x,y=tr.transform(r['lon'],r['lat']); bb=s.bounds
                    if not(bb.left<=x<=bb.right and bb.bottom<=y<=bb.top): continue
                    row,col=s.index(x,y)
                    try: v=int(s.read(1,window=Window(col,row,1,1))[0,0])
                    except: v=0
                    if v>0: out[i]=v; bas[i]=b
    return out,bas
rn,bas=sample('resnet'); pr,_=sample('prithvi')
df['pred_resnet']=rn; df['pred_prithvi']=pr; df['basin']=bas
out='/content/drive/MyDrive/TGSR_Study/validation_predictions.csv'
df.to_csv(out,index=False)
m=df[df['basin']>0]
print('saved -> validation_predictions.csv | matched',len(m),'points')
print('ResNet full-res OA:',round((m.pred_resnet==m.true).mean(),3),
      '| Prithvi:',round((m.pred_prithvi==m.true).mean(),3))

# %% [notebook cell 33]
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

df = pd.read_csv("/content/drive/MyDrive/TGSR_Study/validation_predictions.csv")   # adjust path if needed
df = df[df["true"].isin([1,2,3,4,5,6])].copy()
N = len(df)
NAMES = {1:"Cropland",2:"Forest",3:"Grassland",4:"Bare",5:"Water",6:"Built"}
C = [1,2,3,4,5,6]

def confusion(pred_col):
    M = np.zeros((6,6), int)
    for _, r in df.iterrows():
        M[int(r["true"])-1, int(r[pred_col])-1] += 1
    return M

def oa(pred_col):
    return (df[pred_col] == df["true"]).mean()

M_res, M_pri = confusion("pred_resnet"), confusion("pred_prithvi")
oa_res, oa_pri = oa("pred_resnet"), oa("pred_prithvi")

# row-normalized for color, raw counts as labels
def rownorm(M):
    s = M.sum(1, keepdims=True); s[s==0] = 1
    return M / s

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
labels = [NAMES[c] for c in C]
for ax, (M, oa_v, title) in zip(
        axes, [(M_pri, oa_pri, f"Prithvi-EO-2.0  (OA={oa_pri:.3f})"),
               (M_res, oa_res, f"GID ResNet-50  (OA={oa_res:.3f})")]):
    im = ax.imshow(rownorm(M), cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            if M[i,j] > 0:
                ax.text(j, i, str(M[i,j]), ha="center", va="center",
                        color="white" if rownorm(M)[i,j] > 0.5 else "#333", fontsize=10)
    # Forest-column highlight (orchard confusion)
    ax.add_patch(Rectangle((0.5, -0.5), 1, 6, fill=False, ec="#c0392b", lw=2, ls="--"))
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Reference (true)")
    ax.set_title(title, fontsize=13, fontweight="bold")
fig.colorbar(im, ax=axes, fraction=0.025, label="Row-normalized (producer's) proportion")
fig.suptitle(f"Confusion matrices on {N} validation points (wall-to-wall; dashed: Forest column — orchard confusion)\n"
             f"ResNet area-adjusted OA 33.4% (Olofsson); sample-count OA shown above",
             fontsize=12, y=1.02)
fig.savefig("figS10_confusion_corrected.png", dpi=300, bbox_inches="tight")
fig.savefig("figS10_confusion_corrected.pdf", bbox_inches="tight")
print(f"saved. ResNet OA {oa_res:.3f}, Prithvi OA {oa_pri:.3f}, n={N}")

# %% [notebook cell 34]
import pandas as pd, numpy as np, glob, rasterio
from rasterio.merge import merge

# --- 1. area weights from FULL-RES maps (the fix: native 30 m, not 8x-downsampled) ---
PRED = "/content/drive/MyDrive/preds_full/preds"   # the native-resolution _resnet.tif files
CLASSES = [1,2,3,4,5,6]
NAMES = {1:"Cropland",2:"Forest",3:"Grassland",4:"Bare",5:"Water",6:"Built"}
print("folder exists:", os.path.isdir(PRED))
all_tif = glob.glob(f"{PRED}/**/*.tif", recursive=True)
print(f"total .tif found: {len(all_tif)}")
for f in all_tif[:10]:
    print("  ", os.path.basename(f))
counts = {c: 0 for c in CLASSES}
for b in [59,60,75,78,89,100,108,109]:
    tiles = sorted(set(glob.glob(f"{PRED}/*SB{b}_resnet.tif") +
                       glob.glob(f"{PRED}/*SB{b}-*_resnet.tif")))
    srcs = [rasterio.open(t) for t in tiles]
    arr, _ = merge(srcs); [s.close() for s in srcs]
    for c in CLASSES: counts[c] += int((arr[0] == c).sum())

Ntot = sum(counts.values())
W = {c: counts[c]/Ntot for c in CLASSES}        # area weights
PIX_KM2 = 0.03**2                                # native 30 m pixel = 0.0009 km^2
A = {c: counts[c]*PIX_KM2 for c in CLASSES}; A_tot = sum(A.values())

# --- 2. confusion from the 450 points ---
df = pd.read_csv("/content/drive/MyDrive/TGSR_Study/validation_predictions.csv")
df = df[df["true"].isin(CLASSES)]
n = {(i,j): 0 for i in CLASSES for j in CLASSES}
for _, r in df.iterrows():
    n[(int(r["pred_resnet"]), int(r["true"]))] += 1
ni = {i: sum(n[(i,j)] for j in CLASSES) for i in CLASSES}

# --- 3. Olofsson area-adjusted estimators (eqns 1-6) ---
p = {(i,j): (W[i]*n[(i,j)]/ni[i] if ni[i] else 0) for i in CLASSES for j in CLASSES}
OA = sum(p[(c,c)] for c in CLASSES)
V_OA = sum((W[i]**2)*( (p[(i,i)]/W[i] if W[i] else 0) )*(1-(p[(i,i)]/W[i] if W[i] else 0))/(ni[i]-1)
           for i in CLASSES if ni[i] > 1)
print(f"area-adjusted OA = {OA:.4f} +/- {1.96*np.sqrt(V_OA):.4f}  (95% CI)")
for c in CLASSES:
    pdi = sum(p[(c,j)] for j in CLASSES); pdj = sum(p[(i,c)] for i in CLASSES)
    U = p[(c,c)]/pdi if pdi else float("nan")
    P = p[(c,c)]/pdj if pdj else float("nan")
    print(f"  {NAMES[c]:>10}: UA={U:.3f}  PA={P:.3f}  map_area={A[c]:,.0f} km2")
