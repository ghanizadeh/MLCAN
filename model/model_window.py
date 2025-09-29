# rf_window_page.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from itertools import combinations
from scipy.signal import welch
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================
# Utility functions
# ================================
def a20_index(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    pct_err = np.abs((y_pred - y_true) / np.where(y_true == 0, np.nan, y_true))
    return np.mean(np.nan_to_num(pct_err) <= 0.20)

def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    a20 = a20_index(y_true, y_pred)
    return {"R²": round(r2, 3), "RMSE": round(rmse, 3),
            "MAE": round(mae, 3), "a20": round(a20, 3)}

def plot_pred_vs_true(y_true, y_pred, target_name, dataset_type):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolor="k")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{target_name} ({dataset_type})")
    st.pyplot(plt.gcf())
    plt.close()

# ================================
# Feature extractors
# ================================
def extract_time_features(x, opts, prefix=""):
    feats = {}
    if "Mean" in opts: feats[f"{prefix}_mean"] = np.mean(x)
    if "Median" in opts: feats[f"{prefix}_median"] = np.median(x)
    if "Std" in opts: feats[f"{prefix}_std"] = np.std(x)
    if "Min" in opts: feats[f"{prefix}_min"] = np.min(x)
    if "Max" in opts: feats[f"{prefix}_max"] = np.max(x)
    if "Range" in opts: feats[f"{prefix}_range"] = np.max(x) - np.min(x)
    if "RMS" in opts: feats[f"{prefix}_rms"] = np.sqrt(np.mean(x**2))
    if "Skew" in opts: feats[f"{prefix}_skew"] = pd.Series(x).skew()
    if "Kurtosis" in opts: feats[f"{prefix}_kurt"] = pd.Series(x).kurt()
    return feats

def extract_freq_features(x, fs, opts, prefix=""):
    feats = {}
    if fs is None or fs <= 0: return feats
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    if "Spectral Centroid" in opts:
        feats[f"{prefix}_spec_centroid"] = np.sum(f * Pxx) / np.sum(Pxx)
    if "Band Power" in opts:
        bands = [(0, 5), (5, 50), (50, fs/2)]
        for (lo, hi) in bands:
            idx = np.logical_and(f >= lo, f < hi)
            feats[f"{prefix}_band_{lo}_{hi}"] = np.trapz(Pxx[idx], f[idx]) if idx.any() else 0
    if "Peak Frequency" in opts:
        feats[f"{prefix}_peak_freq"] = f[np.argmax(Pxx)]
    return feats

def extract_cross_features(window_df, opts):
    feats = {}
    sensors = [c for c in window_df.columns if c.startswith("Sensor")]
    if "Correlation" in opts:
        for i, j in combinations(sensors, 2):
            feats[f"corr_{i}_{j}"] = window_df[i].corr(window_df[j])
    if "Diff" in opts:
        for i, j in combinations(sensors, 2):
            feats[f"diff_{i}_{j}"] = window_df[i].mean() - window_df[j].mean()
    return feats

# ================================
# Sliding window
# ================================
def window_indices(n, win, hop):
    idx = []
    start = 0
    while start + win <= n:
        idx.append((start, start + win))
        start += hop
    return idx

def build_dataset(df, features_time, features_freq, features_cross,
                  window_sec, overlap, fs, target, indicator_choice):
    rows, targets = [], []
    sensors = [c for c in df.columns if c.startswith("Sensor")]

    # Indicator filter
    if indicator_choice != "both" and "indicator" in df.columns:
        df = df[df["indicator"] == int(indicator_choice)]

    win = int(window_sec * fs)
    hop = int(win * (1 - overlap))
    idxs = window_indices(len(df), win, hop)

    for (i0, i1) in idxs:
        wdf = df.iloc[i0:i1]
        feats = {}
        for s in sensors:
            x = wdf[s].values
            feats.update(extract_time_features(x, features_time, prefix=f"{s}_time"))
            feats.update(extract_freq_features(x, fs, features_freq, prefix=f"{s}_freq"))
        feats.update(extract_cross_features(wdf, features_cross))
        rows.append(feats)
        targets.append(wdf[target].median())

    return pd.DataFrame(rows).fillna(0), np.array(targets)

# ================================
# Streamlit page
# ================================
def show_RF_window_page():
    st.title("🌲 Random Forest with Sliding Windows")

    # --- File input ---
    option = st.radio("Choose input method:", ("Local folder", "Upload CSVs"))
    dfs = []
    if option == "Local folder":
        folder_path = st.text_input("📂 Enter folder path with CSVs:")
        if folder_path and os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
            for f in files:
                try:
                    dfs.append(pd.read_csv(os.path.join(folder_path, f)))
                except Exception as e:
                    st.error(f"❌ {f}: {e}")
    else:
        uploaded_files = st.file_uploader("Upload CSVs", type="csv", accept_multiple_files=True)
        if uploaded_files:
            for f in uploaded_files:
                try:
                    dfs.append(pd.read_csv(f))
                except Exception as e:
                    st.error(f"❌ {f.name}: {e}")
    if not dfs: return
    df = pd.concat(dfs, ignore_index=True)

    # --- Config ---
    target = st.selectbox("🎯 Select target", ["CRL", "AliCat"])
    fs = st.number_input("Sampling frequency (Hz)", value=1065, step=1)
    window_sec = st.slider("Window size (s)", 0.5, 10.0, 2.0, 0.5)
    overlap = st.slider("Overlap (%)", 0, 90, 50, 5) / 100.0
    indicator_choice = st.radio("Indicator filter", ["0", "1", "both"], index=1)

    with st.expander("⚙️ Feature selection", expanded=True):
        features_time = st.multiselect("Time-domain", ["Mean","Median","Std","Min","Max","Range","RMS","Skew","Kurtosis"], default=["Mean","Std"])
        features_freq = st.multiselect("Frequency-domain", ["Spectral Centroid","Band Power","Peak Frequency"], default=["Spectral Centroid"])
        features_cross = st.multiselect("Cross-sensor", ["Correlation","Diff"], default=[])

    # --- Train ---
    if st.button("🚀 Train Random Forest"):
        X, y = build_dataset(df, features_time, features_freq, features_cross,
                             window_sec, overlap, fs, target, indicator_choice)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
        ])
        pipe.fit(X, y)
        y_pred = pipe.predict(X)

        # Metrics
        st.subheader("📊 Metrics (in-sample)")
        st.write(evaluate_metrics(y, y_pred))

        # Plot
        st.subheader("📈 Predicted vs True")
        plot_pred_vs_true(y, y_pred, target, "Train (all windows)")

        # Feature importance
        importances = pipe.named_steps["rf"].feature_importances_
        fi_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values("Importance", ascending=False)
        st.subheader("🔍 Feature Importances")
        st.dataframe(fi_df)

        # Save
        model_path = f"{target}_rf_window.pkl"
        joblib.dump(pipe, model_path)
        st.success(f"✅ Model saved as {model_path}")
