# README 02 — Preprocessing Notebook & `gnn_preprocessed.pkl`

## Purpose
This notebook takes `gnn_base.pkl` as input and produces `gnn_preprocessed.pkl`.
Its only job is to identify which days have too many missing station readings
and save a `flagged` boolean array. Everything else (normalisation, NaN filling)
happens later in the training notebook.

---

## Input
| File | Description |
|------|-------------|
| `gnn_base.pkl` | Raw data, graph, station metadata |

---

## Config (set at top of notebook)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `WINDOW_SIZE` | `7` | Days in the input window |
| `SKIP_THRESHOLD` | `0.30` | Fraction of core features missing per station to flag it |
| `MIN_BAD_STATIONS` | `3` | Number of stations that must be flagged to mark a day as bad |
| `CORE_FEATURES` | `['TG','TN','TX','UG','RH','FG','Q']` | Features used for flagging only |

---

## Steps

### Step 1 — Load gnn_base.pkl
- Loads `data_array`, `dates`, `stations`, `feature_cols`, `rh_idx`
- Data is raw and un-normalised at this stage

### Step 2 — Compute NaN mask
- `nan_mask = np.isnan(data_array)` — shape `[T, N, F]`
- Overall NaN rate: **16.7%**

### Step 3 — Identify core features
Initial attempt used `['TG', 'TN', 'TX', 'UG', 'RH', 'FG', 'PG', 'Q']` but
`PG` (pressure) had **36.4% missing** — structurally sparse, not a sensor fault.
Including it would flag too many valid days.

Final core features: `['TG', 'TN', 'TX', 'UG', 'RH', 'FG', 'Q']`

### Step 4 — Compute missing fraction per station per day
- For each day and each station, compute what fraction of core features is missing
- Shape: `[T, N]`

### Step 5 — Flag bad days
- A station is "bad" on a given day if more than 30% of its core features are missing
- A day is "flagged" if **3 or more stations** are bad on that day
- This avoids flagging days where only 1-2 stations have minor gaps

### Step 6 — Results

| Category | Count | Percentage |
|----------|-------|------------|
| Total days | 10,897 | 100% |
| Flagged days | 2,151 | 19.7% |
| Usable days | 8,746 | 80.3% |

---

## Why this approach

### Why not flag any day with a single bad station?
77.6% of days would be flagged — almost no usable data.

### Why core features only?
Many features (NG, W1-W6, PG etc.) are structurally sparse — entire stations
never report them. Using all 46 features would flag valid days just because
an optional sensor was never installed.

### Why MIN_BAD_STATIONS = 3?
- 1 bad station → 77.6% flagged (too aggressive)
- 2 bad stations → 52.5% flagged (too aggressive)
- 3 bad stations → 19.7% flagged ✅ (reasonable)
- 5 bad stations → 2.0% flagged (too permissive)

---

## What is NOT in this file
- No data values
- No scalers
- No station metadata (already in gnn_base.pkl — no point duplicating)
- No NaN filling (must happen after normalisation in training notebook)

---

## Output — `gnn_preprocessed.pkl`

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `flagged` | `np.bool_` | `(10897,)` | True = day has >=3 bad stations |
| `config` | `dict` | — | All config parameters used |

### Config saved inside the pkl
```python
{
    "window_size"      : 7,
    "skip_threshold"   : 0.3,
    "min_bad_stations" : 3,
    "core_features"    : ['TG', 'TN', 'TX', 'UG', 'RH', 'FG', 'Q']
}
```

---

## NaN rates per feature (for reference)

| Feature | NaN % | Notes |
|---------|-------|-------|
| TG, TN, TX, UG, UN, UX, EEG | ~4.5% | Core met variables, reliable |
| RH, RHX, DR, FG, DDVEC etc. | ~6.9% | Slightly higher, still good |
| Q, SQ, SP, EV24 | ~7.4–7.7% | Solar/radiation variables |
| VVN, VVNH, VVX, VVXH | ~32.5% | Visibility — sparse |
| PG, PN, PNH, PX, PXH | ~36.4% | Pressure — structurally sparse |
| W1, W2, W3, W5 | ~42.8% | Weather type codes — very sparse |
| W6 | ~45.7% | Weather type codes — very sparse |
| NG | ~48.5% | Cloud cover — very sparse |

---

## When to recreate this file
| Change | Recreate? |
|--------|-----------|
| Different `MIN_BAD_STATIONS` | ✅ Yes |
| Different `SKIP_THRESHOLD` | ✅ Yes |
| Different `CORE_FEATURES` | ✅ Yes |
| New raw data in gnn_base.pkl | ✅ Yes |
| Different normalisation | ❌ No |
| Different loss function | ❌ No |
| Different model architecture | ❌ No |
