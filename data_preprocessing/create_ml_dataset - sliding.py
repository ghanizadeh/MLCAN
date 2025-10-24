# utility/create_ml_dataset.py
import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch, find_peaks
from numpy.fft import rfft, rfftfreq
import streamlit as st

# ======================
# Feature extractor
# ======================
def extract_features(df_window, selected_time, selected_freq, selected_cross, selected_slug, fs: float):
    feats = {}
    cols = ["Sensor5","Sensor3","Sensor1","Sensor6","Sensor4","Sensor2"]

    # --- TIME DOMAIN ---
    for col in cols:
        x = df_window[col].values.astype(float)
        if x.size == 0:  # guard
            continue
        if "mean" in selected_time:   feats[f"TIME_{col}_mean"] = float(np.mean(x))
        if "std" in selected_time:    feats[f"TIME_{col}_std"] = float(np.std(x))
        if "min" in selected_time:    feats[f"TIME_{col}_min"] = float(np.min(x))
        if "max" in selected_time:    feats[f"TIME_{col}_max"] = float(np.max(x))
        if "median" in selected_time: feats[f"TIME_{col}_median"] = float(np.median(x))
        if "range" in selected_time:  feats[f"TIME_{col}_range"] = float(np.ptp(x))
        if "skew" in selected_time:   feats[f"TIME_{col}_skew"] = float(skew(x, bias=False))
        if "kurtosis" in selected_time: feats[f"TIME_{col}_kurtosis"] = float(kurtosis(x, bias=False))
        if "rms" in selected_time:    feats[f"TIME_{col}_rms"] = float(np.sqrt(np.mean(x**2)))
        if "p25" in selected_time:    feats[f"TIME_{col}_p25"] = float(np.percentile(x, 25))
        if "p75" in selected_time:    feats[f"TIME_{col}_p75"] = float(np.percentile(x, 75))
        if "iqr" in selected_time:    feats[f"TIME_{col}_iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))
        if "cv" in selected_time:
            m = np.mean(x)
            feats[f"TIME_{col}_cv"] = float(np.std(x) / (m + 1e-8))
        if "entropy" in selected_time:
            hist, _ = np.histogram(x, bins=20, density=True)
            feats[f"TIME_{col}_entropy"] = float(entropy(hist + 1e-12))

    # --- FREQ DOMAIN ---
    for col in cols:
        x = df_window[col].values.astype(float)
        if x.size == 0:
            continue

        # Welch PSD with correct sampling frequency
        nperseg = min(256, len(x))
        f, Pxx = welch(x, fs=fs, nperseg=nperseg)
        Pxx_sum = np.sum(Pxx) + 1e-12
        centroid = np.sum(f * Pxx) / Pxx_sum

        if "dom_freq" in selected_freq: feats[f"FREQ_{col}_dom_freq"] = float(f[np.argmax(Pxx)])
        if "spec_centroid" in selected_freq: feats[f"FREQ_{col}_spec_centroid"] = float(centroid)
        if "spec_bw" in selected_freq:
            feats[f"FREQ_{col}_spec_bw"] = float(np.sqrt(np.sum(((f - centroid) ** 2) * Pxx) / Pxx_sum))
        if "total_power" in selected_freq: feats[f"FREQ_{col}_total_power"] = float(np.sum(Pxx))
        if "low_power" in selected_freq: feats[f"FREQ_{col}_low_power"] = float(np.sum(Pxx[(f >= 0) & (f < 0.1)]))
        if "mid_power" in selected_freq: feats[f"FREQ_{col}_mid_power"] = float(np.sum(Pxx[(f >= 0.1) & (f < 1.0)]))
        if "high_power" in selected_freq: feats[f"FREQ_{col}_high_power"] = float(np.sum(Pxx[f >= 1.0]))
        if "spectral_entropy" in selected_freq:
            p = Pxx / Pxx_sum
            feats[f"FREQ_{col}_spectral_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))

        # FFT on half-spectrum with frequency axis in Hz
        X = rfft(x)
        mag = np.abs(X)
        power = mag ** 2
        power_sum = np.sum(power) + 1e-12
        freqs = rfftfreq(len(x), d=1.0 / fs)

        if "fft_energy" in selected_freq: feats[f"FREQ_{col}_fft_energy"] = float(np.sum(power))
        if "fft_entropy" in selected_freq:
            pwr = power / power_sum
            feats[f"FREQ_{col}_fft_entropy"] = float(-np.sum(pwr * np.log(pwr + 1e-12)))
        if "fft_peak_mag" in selected_freq: feats[f"FREQ_{col}_fft_peak_mag"] = float(np.max(mag))
        if "fft_peak_freq" in selected_freq:
            feats[f"FREQ_{col}_fft_peak_freq"] = float(freqs[np.argmax(mag)])

    # --- CROSS SENSOR ---
    if "dp" in selected_cross:
        feats["CROSS_dp_5_3"] = float(np.mean(df_window["Sensor5"] - df_window["Sensor3"]))
        feats["CROSS_dp_3_1"] = float(np.mean(df_window["Sensor3"] - df_window["Sensor1"]))
        feats["CROSS_dp_6_4"] = float(np.mean(df_window["Sensor6"] - df_window["Sensor4"]))
        feats["CROSS_dp_4_2"] = float(np.mean(df_window["Sensor4"] - df_window["Sensor2"]))

    if "corr" in selected_cross:
        # Fill NA to avoid NaNs in corr
        corr = df_window[cols].fillna(method="ffill").fillna(method="bfill").corr().values
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                feats[f"CROSS_corr_{cols[i]}_{cols[j]}"] = float(corr[i, j])

    if "phase" in selected_cross:
        # Phase difference at dominant frequency of Sensor5 (reference)
        ref = df_window["Sensor5"].values.astype(float)
        if ref.size > 0:
            X_ref = rfft(ref)
            k = int(np.argmax(np.abs(X_ref)))
            for other in cols[1:]:
                X_o = rfft(df_window[other].values.astype(float))
                phase_diff = float(np.angle(X_o[k]) - np.angle(X_ref[k])) if k < len(X_o) else np.nan
                feats[f"CROSS_phase_Sensor5_{other}"] = phase_diff

    # --- SLUGGING ---
    for col in cols:
        x = df_window[col].values.astype(float)
        if x.size == 0:
            continue
        thr = np.mean(x) + np.std(x)
        peaks, _ = find_peaks(x, height=thr)
        if "peak_count" in selected_slug:  feats[f"SLUG_{col}_peak_count"] = int(len(peaks))
        if "peak_rate" in selected_slug:   feats[f"SLUG_{col}_peak_rate"]  = float(len(peaks) / len(x))
        if "crest_factor" in selected_slug: feats[f"SLUG_{col}_crest_factor"] = float(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-8))
        if "peak_to_mean" in selected_slug: feats[f"SLUG_{col}_peak_to_mean"] = float(np.max(x) / (np.mean(x) + 1e-8))
        if "zero_cross" in selected_slug:   feats[f"SLUG_{col}_zero_cross"] = float(np.mean(np.diff(np.sign(x)) != 0))

    return feats


# ======================
# Helper: Window Generator
# ======================
def generate_windows(df, win_size_samples, step_size):
    for start in range(0, len(df) - win_size_samples + 1, step_size):
        yield start, df.iloc[start:start + win_size_samples]


# ======================
# Streamlit Page
# ======================
def show_create_ML_dataset():
    st.markdown("#### 🔧 Select Features ####")

    selected_time = st.multiselect(
        "Time-domain features:",
        ["mean","std","min","max","median","range","skew","kurtosis","rms","p25","p75","iqr","cv","entropy"],
        default=["mean","std"]
    )

    selected_freq = st.multiselect(
        "Frequency-domain features:",
        ["dom_freq","spec_centroid","spec_bw","total_power","low_power","mid_power","high_power","spectral_entropy",
         "fft_energy","fft_entropy","fft_peak_mag","fft_peak_freq"],
        default=["dom_freq","spec_centroid"]
    )

    selected_cross = st.multiselect(
        "Cross-sensor features:",
        ["dp","corr","phase"],
        default=["dp"]
    )

    selected_slug = st.multiselect(
        "Slugging features:",
        ["peak_count","peak_rate","crest_factor","peak_to_mean","zero_cross"],
        default=["peak_count","peak_rate"]
    )

    # ======================
    # Sliding Window Params
    # ======================
    st.markdown("#### 🪟 Sliding Window Parameters ####")
    win_size_sec  = st.number_input("Window size (seconds)", min_value=1, max_value=60, value=5, step=1)
    overlap_pct   = st.slider("Overlap (%)", min_value=0, max_value=90, value=50, step=10)
    sampling_freq = st.number_input("Sampling frequency (Hz)", min_value=1, value=1000, step=100)

    win_size_samples = int(max(1, win_size_sec * sampling_freq))
    step_size = max(1, int(win_size_samples * (1 - overlap_pct / 100.0)))

    # ======================
    # Indicator Selection
    # ======================
    st.markdown("#### 🎚️ Indicator Filter ####")
    indicator_choice = st.radio("Select indicator:", ["1", "0", "both"], index=0)

    # ======================
    # File Source
    # ======================
    st.markdown("#### 📁 Select source ####")
    option = st.radio("Enter the source folder:", ["Local folder", "Upload CSVs (Google Drive / Manual)"])

    required_cols = ["Sensor5","Sensor3","Sensor1","Sensor6","Sensor4","Sensor2"]  # minimal required for features
    results = []
    feature_cols = None  # will be captured from the first computed window

    def process_file(df, filename):
        nonlocal feature_cols

        # Validate required columns
        if not all(col in df.columns for col in required_cols):
            st.warning(f"⚠️ Skipping {filename} - missing required columns {set(required_cols) - set(df.columns)}")
            return

        # Apply indicator filter (if present)
        if "indicator" in df.columns and indicator_choice != "both":
            df = df[df["indicator"] == int(indicator_choice)]

        if len(df) < win_size_samples:
            st.warning(f"⚠️ {filename}: not enough rows ({len(df)}) for window size {win_size_samples}. Skipping.")
            return

        # Generate windowed features
        count_windows = 0
        for win_idx, df_window in generate_windows(df, win_size_samples, step_size):
            feats = extract_features(
                df_window,
                selected_time, selected_freq, selected_cross, selected_slug,
                fs=float(sampling_freq)
            )

            # Capture feature column order once
            if feature_cols is None:
                feature_cols = list(feats.keys())

            # Keep row values aligned to feature_cols (stable order)
            row_vals = [feats.get(k, np.nan) for k in feature_cols]
            results.append([filename, win_idx, indicator_choice] + row_vals)
            count_windows += 1

        st.success(f"✅ File processed: {filename} — windows: {count_windows}")

    # --- Local folder ---
    if option == "Local folder":
        folder_path = st.text_input("Enter the LOCAL folder path containing CSV files:")
        if st.button("Generate from Local Folder"):
            if not folder_path or not os.path.exists(folder_path):
                st.error("⚠️ Please enter a valid folder path.")
            else:
                for file in os.listdir(folder_path):
                    if file.lower().endswith(".csv"):
                        try:
                            df = pd.read_csv(os.path.join(folder_path, file))
                            process_file(df, file)
                        except Exception as e:
                            st.error(f"⚠️ Skipping {file} due to error: {e}")

    # --- Uploaded files ---
    else:
        uploaded_files = st.file_uploader("Upload one or more CSV files", type="csv", accept_multiple_files=True)
        if uploaded_files and st.button("Generate from Uploaded Files"):
            for uploaded_file in uploaded_files:
                try:
                    df = pd.read_csv(uploaded_file)
                    process_file(df, uploaded_file.name)
                except Exception as e:
                    st.error(f"⚠️ Skipping {uploaded_file.name} due to error: {e}")

    # --- Save / Show results ---
    if results:
        if feature_cols is None:
            feature_cols = []  # in case no features were selected
        columns = ["Filename", "WindowStartIdx", "Indicator"] + feature_cols

        output_df = pd.DataFrame(results, columns=columns)
        st.dataframe(output_df)
        csv_bytes = output_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Dataset", csv_bytes, "dataset_with_features.csv", "text/csv")
