# README 03 — Model Architecture & Training Choices

## Overview
A Graph Neural Network (GNN) that predicts next-day rainfall (RH) across
34 Dutch weather stations simultaneously. Each station is a node in a graph;
edges connect the 5 nearest neighbours. The model reads a 7-day history window
per station and outputs one prediction per station.

---

## Files

| File | Description |
|------|-------------|
| `gnn_base.pkl` | Raw data, graph, station metadata |
| `gnn_preprocessed.pkl` | Flagged days + config |
| `gnn_split.pkl` | Train / val / test index splits |
| `best_model_run03.pt` | Best model weights (state dict only) |
| `best_model_run03_full_checkpoint.pt` | Full checkpoint including optimizer state |

---

## Transform Config (run03)

All normalisation choices live in `TransformConfig` — change one block to
run a new experiment without touching any other code.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `target_transform` | `log1p` | Apply log1p to RH before z-score scaling |
| `rain_weight` | `5.0` | Rainy samples penalised 5x in loss |
| `feature_transform` | `zscore` | StandardScaler on all input features |

### Why log1p on the target?
RH (rainfall) is heavily zero-inflated — roughly 70-80% of readings are 0mm
(dry days). Standard z-score normalisation assumes a roughly normal distribution.
Applied to a zero-spike distribution it distorts the target scale and makes the
loss signal misleading during training.

`log1p(x) = log(1 + x)` compresses the long right tail of rainfall values into
a more symmetric distribution. The StandardScaler is then fitted on these
log-transformed values, which better matches its normality assumption.

### Transform pipeline (target RH only)
```
raw mm  →  log1p  →  StandardScaler  →  normalised value used in training
                                         (reverse: inverse_scaler → expm1 → mm)
```

### dry_day_norm
After fitting the scaler on log1p(RH), the value `0.0mm → log1p(0) = 0.0`
is pushed through the scaler to compute `dry_day_norm = -0.7448`. Any
normalised target value below `-0.7448` represents a dry day. This threshold
is used by `WeightedMSELoss`.

---

## NaN Handling

After `build_transforms()` applies normalisation, NaNs are filled with `0.0`.
In normalised space, `0.0 = the feature mean` — a neutral, non-informative value.

**Order matters:** filling must happen after normalisation.
Filling raw values with 0 would inject incorrect signal (0mm is not the mean
of rainfall in raw space).

---

## Dataset — `RainfallWindowDataset`

| Property | Value |
|----------|-------|
| Window size | 7 days |
| Input shape per sample | `[N=34, W=7, F=46]` |
| Target shape per sample | `[N=34, 1]` |
| Total possible samples | 10,890 |
| Valid samples (after flagging) | 8,700 |
| Skipped (flagged days) | 2,190 |

A sample is skipped if **any day** in its 7-day window or its target day
is in the `flagged` array.

---

## Train / Val / Test Split

| Split | Samples | % | Date range |
|-------|---------|---|------------|
| Train | 6,090 | 70% | 1999-03-31 → 2015-12-01 |
| Val | 1,305 | 15% | 2015-12-02 → 2021-02-15 |
| Test | 1,305 | 15% | 2021-02-16 → 2024-10-31 |

Split is **chronological** (not random) to prevent data leakage — the model
is never trained on future data.

The split is saved to `gnn_split.pkl` and reused across runs. It only needs
to be recreated if `WINDOW_SIZE` or `MIN_BAD_STATIONS` changes.

---

## Model Architecture — `RainfallGNN`

```
Input: [N, W, F] = [34 stations, 7 days, 46 features]
         │
         ▼
┌─────────────────┐
│   GRU           │  input_size=46, hidden_size=64, batch_first=True
│                 │  Reads 7-day history per station
│  Output: [N,64] │  (last hidden state only)
└────────┬────────┘
         │  Dropout(0.2)
         ▼
┌─────────────────┐
│   GATConv       │  in=64, out=32, heads=4, concat=True
│                 │  Stations share information with 5 neighbours
│  Output:[N,128] │  (32 per head × 4 heads = 128)
└────────┬────────┘
         │  ReLU + Dropout(0.2)
         ▼
┌─────────────────┐
│   Linear        │  in=128, out=1
│                 │  One prediction per station
│  Output: [N, 1] │
└─────────────────┘
```

| Component | Parameters |
|-----------|------------|
| GRU | `input=46, hidden=64` |
| GAT | `in=64, out=32, heads=4, concat=True` → output 128 |
| Linear | `128 → 1` |
| Dropout | `p=0.2` after GRU and after GAT |
| Total trainable params | **30,209** |

### Why GRU?
Captures temporal dependencies across the 7-day input window per station.
Lighter than LSTM (no cell state) — sufficient for a 7-step sequence.

### Why GAT?
Allows stations to share information with their nearest neighbours using
learned attention weights. A station with a missing reading can borrow
signal from nearby stations. GAT is preferred over GCN because it learns
which neighbours matter more via attention.

### Why K=5 neighbours?
Balances local spatial context (nearby stations are most correlated) against
noise from distant stations. The Netherlands is small — 5 neighbours covers
the immediate region well.

---

## Loss Function — `WeightedMSELoss`

### Problem with standard MSELoss
With ~70-80% dry days in the target, standard MSE rewards the model for
always predicting near zero. It gets penalised less by ignoring rain events
than by attempting to predict them. This leads to negative R² — worse than
predicting the mean.

### Solution
```python
loss = mean( weights * (pred - target)² )

where:
  weights = rain_weight (5.0)  if target > dry_day_norm (-0.7448)
  weights = 1.0                if target <= dry_day_norm
```

Rainy samples contribute 5× more to the loss, forcing the model to attend
to rain events rather than collapsing to predicting zero.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `dry_threshold` | `-0.7448` | Computed from scaler: log1p(0mm) normalised |
| `rain_weight` | `5.0` | Starting value — increase to 10.0 if model still ignores rain |

---

## Optimizer

| Parameter | Value |
|-----------|-------|
| Algorithm | Adam |
| Learning rate | `0.001` |
| Weight decay | None (default) |

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 32 |
| Shuffle train | True |
| Shuffle val/test | False |
| Best model criterion | Lowest `val_loss` |
| Device | CPU |

---

## Run History

| Run | Transform | Loss | Best val_loss | Best epoch | Notes |
|-----|-----------|------|---------------|------------|-------|
| run02 | zscore only | MSELoss | 0.7749 | 32 | R² = -0.026 — baseline |
| run03 | log1p + zscore | WeightedMSELoss (w=5) | TBD | TBD | Current run |

---

## Evaluation Pipeline

After training, predictions are inverse-transformed back to mm:

```
normalised prediction
    → inverse StandardScaler  (back to log space)
    → expm1()                 (inverse of log1p, back to mm)
    → clip at 0               (rainfall cannot be negative)
    → mm
```

Metrics reported in both normalised space and mm for comparability.

---

## MLflow Tracking

| Setting | Value |
|---------|-------|
| Tracking URI | `file:///media/user/DataDisk/Python/Dordrecht/mlruns` |
| Experiment | `rainfall_gnn` |
| UI command | `mlflow ui --backend-store-uri file:///media/user/DataDisk/Python/Dordrecht/mlruns --port 5000` |

Logged per epoch: `train_loss`, `train_mae`, `val_loss`, `val_mae`, `epoch_time_s`

Logged at end: `best_val_loss`, `best_epoch`, `best_val_mae`, `total_time_s`

---

## Changing Experiment Settings

To run a new experiment, change only `active_config` in Cell 1 of the training notebook:

```python
# Try higher rain weight
active_config = TransformConfig(
    target_transform = "log1p",
    rain_weight      = 10.0,    # ← only change needed
)

# Try plain zscore (reproduce run02 behaviour)
active_config = TransformConfig(
    target_transform = "zscore",
    rain_weight      = 1.0,
)
```

Also increment the run name and checkpoint filenames:
```python
RUN_NAME             = "model_04_log1p_weight10"
best_model_path      = "...best_model_run04.pt"
full_checkpoint_path = "...best_model_run04_full_checkpoint.pt"
```
