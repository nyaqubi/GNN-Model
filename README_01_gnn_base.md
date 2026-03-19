# README 01 — Creating `gnn_base.pkl`

## Purpose
`gnn_base.pkl` is the **stable foundation file** for all experiments. It stores raw,
un-normalised data alongside the graph structure and station metadata. It never needs
to be recreated unless the raw CSV data changes.

---

## Input
| File | Description |
|------|-------------|
| `all_stations_combined.csv` | Raw KNMI gauge data, semicolon-separated, all stations |

---

## Steps

### Step 1 — Load CSV
- Loaded with `sep=";"`, `parse_dates=["date"]`, `dayfirst=True`
- Sorted by `["date", "station_name"]`

### Step 1b — Filter to Netherlands only
- Latitude range : `(50.75, 53.75)`
- Longitude range: `(3.20, 7.25)`
- Result: 123 → 114 stations

### Step 1c — Deduplicate stations
- Identified 51 duplicate groups (same physical station, different name over time)
- For each group, kept the station with the **most RH observations**
- Additional check: removed stations within 5km of each other using Haversine distance
- Result: 114 → 62 stations

### Step 1d — Trim time range
- Kept records from `1995-01-01` onwards
- Reason: pre-1995 data is sparse and inconsistent across stations

### Step 1e — Drop low-coverage stations
- Removed stations with less than **20% RH coverage** over the full time range
- Result: 62 → 34 stations

### Step 2 — Auto-detect feature columns
- Excluded: `date`, `station_name`, `lat`, `lon`, `height_m`
- Detected: **47 features**

### Step 3 — Station metadata
- Built a `stations` dataframe with columns: `node_id`, `station_name`, `lat`, `lon`, `height_m`
- `node_id` is the graph node index (0–33)

### Step 4 — Build KNN graph edges
- Used `K=5` nearest neighbours per station based on `(lat, lon)`
- Edge weights: inverse distance `1 / (d + 1e-6)`
- Result: 170 directed edges

### Step 5 — Pivot to `[T, N, F]` array
- Pivoted each feature into a `[T, N]` matrix then stacked
- Final shape: `(10897, 34, 47)` — `[timesteps, stations, features]`

### Step 6 — Drop high-NaN features
- Dropped features with more than **80% missing values**
- Dropped: `TZG`
- Final shape: `(10897, 34, 46)`

### Step 7 — Record RH index
- `rh_idx = feature_cols.index("RH")` → **13**
- This is the target variable index used throughout training

---

## What is NOT in this file
- No normalisation / scaling
- No NaN filling
- No scalers
- No flagged days

All of those are computed at training time from this file.

---

## Output — `gnn_base.pkl`

| Key | Type | Shape / Value | Description |
|-----|------|---------------|-------------|
| `data_array` | `np.float32` | `(10897, 34, 46)` | Raw un-normalised values |
| `dates` | `DatetimeIndex` | length 10897 | 1995-01-01 → 2024-10-31 |
| `stations` | `pd.DataFrame` | `(34, 5)` | node_id, name, lat, lon, height_m |
| `feature_cols` | `list[str]` | length 46 | Feature names in column order |
| `rh_idx` | `int` | `13` | Index of RH (target) in feature_cols |
| `edge_index` | `np.int64` | `(2, 170)` | Graph edge pairs |
| `edge_attr` | `np.float64` | `(170,)` | Edge weights (inverse distance) |
| `K` | `int` | `5` | Number of KNN neighbours used |

---

## 34 Stations Retained

| node_id | Station | lat | lon |
|---------|---------|-----|-----|
| 0 | AMSTERDAM/SCHIPHOL AP | 52.3172 | 4.7897 |
| 1 | ARCEN AWS | 51.4972 | 6.1961 |
| 2 | BERKHOUT AWS | 52.6428 | 4.9789 |
| 3 | CABAUW TOWER AWS | 51.9692 | 4.9258 |
| 4 | DE BILT AWS | 52.0989 | 5.1797 |
| 5 | DE KOOY VK | 52.9269 | 4.7811 |
| 6 | DEELEN | 52.0547 | 5.8722 |
| 7 | EINDHOVEN AP | 51.4497 | 5.3769 |
| 8 | ELL AWS | 51.1967 | 5.7625 |
| 9 | GILZE RIJEN | 51.5650 | 4.9353 |
| 10 | GRONINGEN AP EELDE | 53.1236 | 6.5847 |
| 11 | HEINO AWS | 52.4344 | 6.2589 |
| 12 | HERWIJNEN AWS | 51.8578 | 5.1453 |
| 13 | HOEK VAN HOLLAND AWS | 51.9911 | 4.1217 |
| 14 | HOOGEVEEN AWS | 52.7489 | 6.5731 |
| 15 | HUPSEL AWS | 52.0678 | 6.6567 |
| 16 | LAUWERSOOG AWS | 53.4117 | 6.1992 |
| 17 | LEEUWARDEN | 53.2231 | 5.7517 |
| 18 | LELYSTAD AP | 52.4483 | 5.5081 |
| 19 | MAASTRICHT AACHEN AP | 50.9053 | 5.7619 |
| 20 | MARKNESSE AWS | 52.7019 | 5.8875 |
| 21 | NIEUW BEERTA AWS | 53.1944 | 7.1492 |
| 22 | ROTTERDAM THE HAGUE AP | 51.9606 | 4.4469 |
| 23 | SOESTERBERG | 52.1289 | 5.2731 |
| 24 | STAVOREN AWS | 52.8967 | 5.3833 |
| 25 | TERSCHELLING HOORN AWS | 53.3911 | 5.3458 |
| 26 | TWENTHE AWS | 52.2731 | 6.8908 |
| 27 | VALKENBURG VK | 52.1703 | 4.4294 |
| 28 | VLISSINGEN AWS | 51.4414 | 3.5958 |
| 29 | VOLKEL | 51.6586 | 5.7067 |
| 30 | WESTDORPE AWS | 51.2247 | 3.8611 |
| 31 | WIJK AAN ZEE AWS | 52.5053 | 4.6031 |
| 32 | WILHELMINADORP AWS | 51.5258 | 3.8836 |
| 33 | WOENSDRECHT | 51.4478 | 4.3419 |

---

## When to recreate this file
| Change | Recreate? |
|--------|-----------|
| New raw CSV data added | ✅ Yes |
| Different station filter | ✅ Yes |
| Different K for KNN graph | ✅ Yes |
| Different normalisation | ❌ No — handled in training |
| Different loss function | ❌ No |
| Different window size | ❌ No |
