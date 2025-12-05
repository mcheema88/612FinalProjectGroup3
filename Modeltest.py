import os
import time
import joblib
from pathlib import Path
from math import sqrt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import tensorflow as tf
from tensorflow import keras

# CONFIG
DATA_DIR = Path("../data")
MODEL_DIR = Path("../models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    DATA_DIR / "combined_all_areas.csv",
    DATA_DIR / "urban_areas_combined.csv",
    DATA_DIR / "rural_areas_combined.csv"
]

# LOAD DATA 
print("Loading dataset...")
df = None
for path in CANDIDATES:
    if path.exists():
        df = pd.read_csv(path, parse_dates=['datetime'] if 'datetime' in pd.read_csv(path, nrows=1).columns else None)
        print(f"Loaded: {path.name} ({df.shape})")
        break
if df is None:
    raise FileNotFoundError("No data file found in ../data/")

# PREPROCESSING 
# Target column detection
target_candidates = ['load_mw', 'Load_MW', 'load', 'Load', 'actual_load']
target_col = next((col for col in target_candidates if col in df.columns), None)
if not target_col:
    raise ValueError("Target load column not found!")
df = df.rename(columns={target_col: "load_mw"})

# Ensure datetime
if 'datetime' not in df.columns:
    for col in ['timestamp', 'date', 'DateTime']:
        if col in df.columns:
            df = df.rename(columns={col: 'datetime'})
            break
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Weather column normalization
if 'temperature_c' in df.columns:
    df = df.rename(columns={'temperature_c': 'temp_c'})
if 'precipitation' in df.columns:
    df = df.rename(columns={'precipitation': 'precip_mm'})

# Fill missing weather
for col in ['temp_c', 'precip_mm']:
    if col in df.columns:
        df[col] = df[col].ffill().bfill()

# Outlier removal (3×IQR)
Q1 = df['load_mw'].quantile(0.25)
Q3 = df['load_mw'].quantile(0.75)
IQR = Q3 - Q1
df = df[df['load_mw'].between(Q1 - 3*IQR, Q3 + 3*IQR)]

# Train/test split (80/20 chronological)
split_idx = int(0.8 * len(df))
train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
print(f"Train: {train_df.shape} | Test: {test_df.shape}")

# FEATURE ENGINEERING
def add_time_features(df):
    d = df.copy()
    d['hour'] = d['datetime'].dt.hour
    d['dayofweek'] = d['datetime'].dt.dayofweek
    d['month'] = d['datetime'].dt.month

    d['hour_sin'] = np.sin(2 * np.pi * d['hour']/24)
    d['hour_cos'] = np.cos(2 * np.pi * d['hour']/24)
    d['dow_sin'] = np.sin(2 * np.pi * d['dayofweek']/7)
    d['dow_cos'] = np.cos(2 * np.pi * d['dayofweek']/7)
    d['month_sin'] = np.sin(2 * np.pi * (d['month']-1)/12)
    d['month_cos'] = np.cos(2 * np.pi * (d['month']-1)/12)
    return d

def add_lags_rolls(df, lags=[1, 24, 168], roll=24):
    d = df.copy().sort_values('datetime')
    for lag in lags:
        d[f'load_lag_{lag}h'] = d['load_mw'].shift(lag)
    d[f'load_roll_mean_{roll}h'] = d['load_mw'].rolling(roll, min_periods=1).mean()
    d[f'load_roll_std_{roll}h'] = d['load_mw'].rolling(roll, min_periods=1).std().fillna(0)
    return d

# Apply
train_fe = train_df.pipe(add_time_features).pipe(add_lags_rolls)
test_fe = test_df.pipe(add_time_features).pipe(add_lags_rolls)

train_fe = train_fe.dropna().reset_index(drop=True)
test_fe = test_fe.dropna().reset_index(drop=True)

# Categorical encoding
cat_cols = [c for c in ['area_code', 'region', 'location_name'] if c in train_fe.columns]
if cat_cols:
    train_fe = pd.get_dummies(train_fe, columns=cat_cols, drop_first=True)
    test_fe = pd.get_dummies(test_fe, columns=cat_cols, drop_first=True)
    train_fe, test_fe = train_fe.align(test_fe, join='left', axis=1, fill_value=0)

# Final X/y
features = [c for c in train_fe.columns if c not in ['datetime', 'load_mw']]
X_train = train_fe[features].values
y_train = train_fe['load_mw'].values
X_test = test_fe[features].values
y_test = test_fe['load_mw'].values

# Scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')
joblib.dump(features, MODEL_DIR / 'features.pkl')
print(f"{len(features)} features engineered")

# MODEL TRAINING
def evaluate(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred)
    }

results = {}

models = {}

# 1–5: Individual models
print("Training models...")
lr = LinearRegression()
models['Linear'] = lr.fit(X_train_s, y_train)
results['Linear'] = evaluate(y_test, lr.predict(X_test_s))

rf = RandomForestRegressor(n_estimators=400, max_depth=25, n_jobs=-1, random_state=42)
models['RF'] = rf.fit(X_train_s, y_train)
results['RF'] = evaluate(y_test, rf.predict(X_test_s))

gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42)
models['GBR'] = gbr.fit(X_train_s, y_train)
results['GBR'] = evaluate(y_test, gbr.predict(X_test_s))

xgb_m = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, n_jobs=-1, random_state=42)
models['XGB'] = xgb_m.fit(X_train_s, y_train)
results['XGB'] = evaluate(y_test, xgb_m.predict(X_test_s))

# Neural Net
def build_nn():
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_dim=X_train_s.shape[1]),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

nn = build_nn()
nn.fit(X_train_s, y_train, epochs=150, batch_size=128, validation_split=0.2,
       callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)], verbose=0)
models['NN'] = nn
results['NN'] = evaluate(y_test, nn.predict(X_test_s, verbose=0).flatten())

# Ensembles
voting = VotingRegressor([('rf', rf), ('xgb', xgb_m), ('gbr', gbr)])
models['Voting'] = voting.fit(X_train_s, y_train)
results['Voting'] = evaluate(y_test, voting.predict(X_test_s))

stack = StackingRegressor(
    estimators=[('rf', rf), ('xgb', xgb_m), ('gbr', gbr)],
    final_estimator=Ridge(), cv=5, n_jobs=-1
)
models['Stacking'] = stack.fit(X_train_s, y_train)
results['Stacking'] = evaluate(y_test, stack.predict(X_test_s))

# RESULTS & SAVE
comp = pd.DataFrame(results).T.sort_values('MAE')
print("\nFINAL RANKING BY MAE:")
print(comp.round(3))

# Plot
plt.figure(figsize=(10,5))
sns.barplot(x=comp.index, y='MAE', data=comp.reset_index(), palette="coolwarm")
plt.title("Model Comparison - MAE (Lower = Better)")
plt.xticks(rotation=45)
plt.ylabel("MAE (MW)")
plt.tight_layout()
plt.show()

# Save best
best_name = comp.index[0]
best_model = models[best_name]

if best_name == "NN":
    nn.save(MODEL_DIR / "best_model_nn.keras")
else:
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")

comp.to_csv(MODEL_DIR / "comparison.csv")
print(f"\nBest model: {best_name} → saved to {MODEL_DIR}")

# PREDICTION FUNCTION
def predict_load(datetime_str: str, temp_c: float = 10.0, precip_mm: float = 0.0, area_code=None):
    """Predict single future point using deployed model"""
    features_list = joblib.load(MODEL_DIR / 'features.pkl')
    scaler = joblib.load(MODEL_DIR / 'scaler.pkl')

    future = pd.DataFrame([{
        'datetime': pd.to_datetime(datetime_str),
        'temp_c': temp_c,
        'precip_mm': precip_mm
    }])
    if area_code is not None:
        future['area_code'] = area_code

    future = add_time_features(future)
    history = train_df.tail(200).copy()
    full = pd.concat([history, future]).reset_index(drop=True)
    full = add_lags_rolls(full)
    row = full.iloc[-1:]

    if cat_cols:
        row = pd.get_dummies(row, columns=cat_cols, drop_first=True)

    X = row.reindex(columns=features_list, fill_value=0).values
    X_s = scaler.transform(X)

    if best_name == "NN":
        model = keras.models.load_model(MODEL_DIR / "best_model_nn.keras")
        return float(model.predict(X_s, verbose=0)[0][0])
    else:
        model = joblib.load(MODEL_DIR / "best_model.pkl")
        return float(model.predict(X_s)[0])

# Example
if __name__ == "__main__":
    pred = predict_load("2025-07-15 14:00", temp_c=28.0, precip_mm=0.0)
    print(f"\nPrediction example (Jul 15 2025, 2PM, 28°C): {pred:.1f} MW")