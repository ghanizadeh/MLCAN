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
from sklearn.model_selection import train_test_split

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
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
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
# Sliding window helpers
# ================================
def window_indices(n, win, hop):
    idx = []
    start = 0
    while start + win <= n:
        idx.append((start, start + win))
        start += hop
    return idx

def build_dataset_from_files(file_list, features_time, features_freq, features_cross,
                             window_sec, overlap, fs, target, indicator_choice):
    """
    Build dataset from multiple CSVs.
    - If window_sec == 0: one feature vector per file.
    - Else: sliding windows per file.
    """
    all_rows, all_targets = [], []

    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")
            continue

        sensors = [c for c in df.columns if c.startswith("Sensor")]

        # Indicator filter
        if indicator_choice != "both" and "indicator" in df.columns:
            df = df[df["indicator"] == int(indicator_choice)]
        if df.empty:
            continue

        # --- Case 1: Full-signal mode ---
        if window_sec == 0:
            feats = {}
            for s in sensors:
                x = df[s].values
                feats.update(extract_time_features(x, features_time, prefix=f"{s}_time"))
                feats.update(extract_freq_features(x, fs, features_freq, prefix=f"{s}_freq"))
            feats.update(extract_cross_features(df, features_cross))
            all_rows.append(feats)
            all_targets.append(df[target].median())
            continue

        # --- Case 2: Sliding-window mode ---
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
            all_rows.append(feats)
            all_targets.append(wdf[target].median())

    return pd.DataFrame(all_rows).fillna(0), np.array(all_targets)

# ================================
# Streamlit Page
# ================================
def show_RF_window_page():
    st.title("🌲 Random Forest with Sliding or Full-Signal Features")

    # --- File Input ---
    folder_path = st.text_input("📂 Enter folder path with CSV signals:")
    if not folder_path or not os.path.isdir(folder_path):
        st.info("Please provide a valid folder path containing CSV signal files.")
        return

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not files:
        st.error("No CSV files found in the folder.")
        return

    # --- Configuration ---
    target = st.selectbox("🎯 Select target", ["CRL", "AliCat"])
    fs = st.number_input("Sampling frequency (Hz)", value=1065, step=1)
    window_sec = st.slider("Window size (s)", 0.0, 60.0, 0.0, 0.5)  # allow 0
    overlap = st.slider("Overlap (%)", 0, 90, 50, 5) / 100.0
    indicator_choice = st.radio("Indicator filter", ["0", "1", "both"], index=2)
    train_size = st.slider("Training set size (%)", 50, 90, 70, 5) / 100.0

    with st.expander("⚙️ Feature selection", expanded=True):
        features_time = st.multiselect(
            "Time-domain features",
            ["Mean","Median","Std","Min","Max","Range","RMS","Skew","Kurtosis"],
            default=["Mean","Std"]
        )
        features_freq = st.multiselect(
            "Frequency-domain features",
            ["Spectral Centroid","Band Power","Peak Frequency"],
            default=["Spectral Centroid"]
        )
        features_cross = st.multiselect(
            "Cross-sensor features", ["Correlation","Diff"], default=[]
        )

    # --- Train Model ---
    if st.button("🚀 Train Random Forest"):
        with st.spinner("Extracting features..."):
            X, y = build_dataset_from_files(
                files, features_time, features_freq, features_cross,
                window_sec, overlap, fs, target, indicator_choice
            )

        if X.empty:
            st.error("No features were extracted — check your input settings.")
            return

        st.subheader("🧾 Extracted Dataset Preview")
        st.write(X.head())
        st.write(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
        ])
        pipe.fit(X_train, y_train)

        # Predict
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)

        # Evaluate
        metrics_train = evaluate_metrics(y_train, y_pred_train)
        metrics_test = evaluate_metrics(y_test, y_pred_test)
        st.subheader("📊 Model Performance")
        metrics_df = pd.DataFrame([metrics_train, metrics_test], index=["Train", "Test"])
        st.table(metrics_df)

        # Plots
        st.subheader("📈 Predicted vs True")
        col1, col2 = st.columns(2)
        with col1:
            plot_pred_vs_true(y_train, y_pred_train, target, "Train")
        with col2:
            plot_pred_vs_true(y_test, y_pred_test, target, "Test")

        # Feature importances
        importances = pipe.named_steps["rf"].feature_importances_
        fi_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values("Importance", ascending=False)
        st.subheader("🔍 Feature Importances")
        st.dataframe(fi_df)

        # Save model
        model_path = f"{target}_rf_window.pkl"
        joblib.dump(pipe, model_path)
        st.success(f"✅ Model saved as {model_path}")
