# utility/create_ml_dataset.py
import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch, find_peaks
from scipy.fft import fft
import streamlit as st

# ======================
# Feature extractor
# ======================
def extract_features(df_window, selected_time, selected_freq, selected_cross, selected_slug, selected_sensors):
    feats = {}
    cols = selected_sensors   # use user-selected sensors

    # --- TIME DOMAIN ---
    for col in cols:
        x = df_window[col].values
        if "mean" in selected_time:   feats[f"TIME_{col}_mean"] = np.mean(x)
        if "std" in selected_time:    feats[f"TIME_{col}_std"] = np.std(x)
        if "min" in selected_time:    feats[f"TIME_{col}_min"] = np.min(x)
        if "max" in selected_time:    feats[f"TIME_{col}_max"] = np.max(x)
        if "median" in selected_time: feats[f"TIME_{col}_median"] = np.median(x)
        if "range" in selected_time:  feats[f"TIME_{col}_range"] = np.ptp(x)
        if "skew" in selected_time:   feats[f"TIME_{col}_skew"] = skew(x)
        if "kurtosis" in selected_time: feats[f"TIME_{col}_kurtosis"] = kurtosis(x)
        if "rms" in selected_time:    feats[f"TIME_{col}_rms"] = np.sqrt(np.mean(x**2))
        if "p25" in selected_time:    feats[f"TIME_{col}_p25"] = np.percentile(x, 25)
        if "p75" in selected_time:    feats[f"TIME_{col}_p75"] = np.percentile(x, 75)
        if "iqr" in selected_time:    feats[f"TIME_{col}_iqr"] = np.percentile(x, 75) - np.percentile(x, 25)
        if "cv" in selected_time:     feats[f"TIME_{col}_cv"] = np.std(x) / (np.mean(x)+1e-8)
        if "entropy" in selected_time:
            hist, _ = np.histogram(x, bins=20, density=True)
            feats[f"TIME_{col}_entropy"] = entropy(hist + 1e-12)

    # --- FREQ DOMAIN ---
    for col in cols:
        x = df_window[col].values
        f, Pxx = welch(x)
        if "dom_freq" in selected_freq: feats[f"FREQ_{col}_dom_freq"] = f[np.argmax(Pxx)]
        if "spec_centroid" in selected_freq: feats[f"FREQ_{col}_spec_centroid"] = np.sum(f * Pxx) / np.sum(Pxx)
        if "spec_bw" in selected_freq: feats[f"FREQ_{col}_spec_bw"] = np.sqrt(np.sum(((f - np.sum(f*Pxx)/np.sum(Pxx))**2) * Pxx) / np.sum(Pxx))
        if "total_power" in selected_freq: feats[f"FREQ_{col}_total_power"] = np.sum(Pxx)
        if "low_power" in selected_freq: feats[f"FREQ_{col}_low_power"] = np.sum(Pxx[(f>=0) & (f<0.1)])
        if "mid_power" in selected_freq: feats[f"FREQ_{col}_mid_power"] = np.sum(Pxx[(f>=0.1) & (f<1.0)])
        if "high_power" in selected_freq: feats[f"FREQ_{col}_high_power"] = np.sum(Pxx[f>=1.0])
        if "spectral_entropy" in selected_freq:
            feats[f"FREQ_{col}_spectral_entropy"] = -np.sum((Pxx/np.sum(Pxx)+1e-12)*np.log(Pxx/np.sum(Pxx)+1e-12))

        fft_vals = fft(x)
        mag = np.abs(fft_vals)
        power = mag**2
        norm_power = power/np.sum(power)

        if "fft_energy" in selected_freq: feats[f"FREQ_{col}_fft_energy"] = np.sum(power)
        if "fft_entropy" in selected_freq: feats[f"FREQ_{col}_fft_entropy"] = -np.sum(norm_power*np.log(norm_power+1e-12))
        if "fft_peak_mag" in selected_freq: feats[f"FREQ_{col}_fft_peak_mag"] = np.max(mag)
        if "fft_peak_freq" in selected_freq: feats[f"FREQ_{col}_fft_peak_freq"] = np.argmax(mag) * (0.5/len(mag))

    # --- CROSS SENSOR ---
    if "corr" in selected_cross and len(cols) > 1:
        corr = df_window[cols].corr().values
        for i in range(len(cols)):
            for j in range(i+1,len(cols)):
                feats[f"CROSS_corr_{cols[i]}_{cols[j]}"] = corr[i,j]

    return feats

# ======================
# Sliding Window Helper
# ======================
def sliding_window(df, window_size, overlap, fs):
    step = int(window_size * fs * (1 - overlap))
    size = int(window_size * fs)
    windows = []
    for start in range(0, len(df) - size + 1, step):
        windows.append(df.iloc[start:start+size])
    return windows

# ======================
# Find Signal Zero Helper
# ======================
def get_signal_zero_means(folder_path, indicator_choice, auto_trim_size=None):
    zero_means = {}
    for file in os.listdir(folder_path):
        if file.lower().endswith(".csv") and "alicat0.0" in file.lower() and "vfd0.0" in file.lower():
            try:
                df = pd.read_csv(os.path.join(folder_path, file))
                if indicator_choice != "both":
                    df = df[df["indicator"] == int(indicator_choice)]
                if auto_trim_size is not None and len(df) > auto_trim_size:
                    df = df.iloc[:auto_trim_size]
                zero_means["Sensor1"] = df["Sensor1"].mean()
                zero_means["Sensor2"] = df["Sensor2"].mean()
                st.info(f"📏 Signal Zero detected from `{file}` (S1={zero_means['Sensor1']:.4f}, S2={zero_means['Sensor2']:.4f})")
                return zero_means
            except Exception as e:
                st.error(f"⚠️ Could not read {file}: {e}")
    st.warning("⚠️ No Signal Zero file found (AliCat0.0 & VFD0.0).")
    return None

# ======================
# Main Streamlit Page
# ======================
def show_create_ML_dataset():
    #st.markdown("### 🧮 Create ML Dataset from Processed CSVs")

    with st.container(border=True):
        st.markdown("#### 📡 Select Sensors ####")
        all_sensors = ["Sensor1","Sensor2","Sensor3","Sensor4","Sensor5","Sensor6"]
        selected_sensors = st.multiselect("Choose sensors to include:", all_sensors, default=all_sensors)

    # --- Feature selection ---
    with st.container(border=True):
        st.markdown("#### 🔧 Select Features ####")
        selected_time = st.multiselect("Time-domain features:",
            ["mean","std","min","max","median","range","skew","kurtosis","rms","p25","p75","iqr","cv","entropy"],
            default=["mean","std"])
        selected_freq = st.multiselect("Frequency-domain features:",
            ["dom_freq","spec_centroid","spec_bw","total_power","low_power","mid_power","high_power","spectral_entropy",
            "fft_energy","fft_entropy","fft_peak_mag","fft_peak_freq"],
            default=["dom_freq","spec_centroid"])
        selected_cross = st.multiselect("Cross-sensor features:", ["dp","corr","phase"], default=["dp"])
        selected_slug = st.multiselect("Slugging features:",
            ["peak_count","peak_rate","crest_factor","peak_to_mean","zero_cross"],
            default=["peak_count","peak_rate"])

    # --- Options ---
    # --- Options ---
    with st.container(border=True):
        st.markdown("#### ⚙️ Options ####")
        use_sliding = st.radio("**Sliding Window** method:", ["No - Use Entire Signal", "Yes - Apply Sliding Windows"], index=0)
        if use_sliding == "Yes - Apply Sliding Windows":
            fs = st.number_input("Sampling frequency (Hz)", min_value=1, value=100)
            window_size = st.number_input("Window size (seconds)", min_value=1, value=2)
            overlap_pct = st.slider("Overlap (%)", 0, 90, 50)
            overlap = overlap_pct/100
        else:
            fs, window_size, overlap = None, None, None

        apply_signal_zero = st.checkbox("Apply **Signal Zero adjusment** (AliCat0.0 & VFD0.0 baseline)", value=False,
                                        help="If enabled, the means of sensor 1 & 2 will be adjusted.")

        remove_blocks = st.selectbox("**Remove trailing blocks** (1 block = 2048 samples)", ["None", "Auto-adjust to uniform signal length"])
        indicator_choice = st.radio("Select indicator:", [1, 0, "both"], index=0)

        # --- Target aggregation choice ---
        target_method = st.radio("How to compute CRL/AliCat target per window?", ["Median", "Mean"], index=0)

        # --- NEW Cleaning options ---
        remove_nans = st.checkbox("Remove rows with NaNs", value=False)
        remove_dupes = st.checkbox("Remove duplicate rows", value=False)

    # --- Source ---
    #with st.container(border=True):
    st.markdown("#### 📂 Select source ####")
    option = st.radio("Enter the source folder:", ["Local folder", "Upload CSVs"])
    #sensor_cols = ["Sensor5","Sensor3","Sensor1","CRL","Sensor6","Sensor4","Sensor2","AliCat"]

    results = []


    # ======================
    # process_dataframe
    # ======================
    def process_dataframe(df, filename, auto_trim_size=None, zero_means=None):
        original_size = len(df)

        # filter indicator
        if indicator_choice != "both":
            df = df[df["indicator"] == int(indicator_choice)]
            original_size_after_ind = len(df)
        filtered_size = len(df)

        trimmed_size = filtered_size
        blocks_removed = 0

        if auto_trim_size is not None and filtered_size > auto_trim_size:
            df = df.iloc[:auto_trim_size]
            trimmed_size = len(df)
            blocks_removed = (filtered_size - trimmed_size) // 2048
        else:
            trimmed_size = filtered_size

        # --- Choose target function ---
        target_fn = np.median if target_method == "Median" else np.mean

        # --- Whole signal ---
        if use_sliding == "No - Use Entire Signal":
            feats = extract_features(df, selected_time, selected_freq, selected_cross, selected_slug, selected_sensors)

            # Adjust Sensor1 & Sensor2 means if option is enabled
            if zero_means and "mean" in selected_time:
                if "TIME_Sensor1_mean" in feats:
                    feats["TIME_Sensor1_mean"] -= zero_means.get("Sensor1", 0.0)
                if "TIME_Sensor2_mean" in feats:
                    feats["TIME_Sensor2_mean"] -= zero_means.get("Sensor2", 0.0)

            row = {
                "Filename": filename,
                "CRL_target": target_fn(df["CRL"]),
                "AliCat_target": target_fn(df["AliCat"]),
                **feats
            }
            return [row], original_size_after_ind, trimmed_size, blocks_removed

        # --- Sliding windows ---
        else:
            win_list = sliding_window(df, window_size, overlap, fs)
            rows = []
            for i, win in enumerate(win_list):
                feats = extract_features(win, selected_time, selected_freq, selected_cross, selected_slug, selected_sensors)

                # Adjust Sensor1 & Sensor2 means if option is enabled
                if zero_means and "mean" in selected_time:
                    if "TIME_Sensor1_mean" in feats:
                        feats["TIME_Sensor1_mean"] -= zero_means.get("Sensor1", 0.0)
                    if "TIME_Sensor2_mean" in feats:
                        feats["TIME_Sensor2_mean"] -= zero_means.get("Sensor2", 0.0)

                row = {
                    "Filename": filename,
                    "Window": i,
                    "CRL_target": target_fn(win["CRL"]),
                    "AliCat_target": target_fn(win["AliCat"]),
                    **feats
                }
                rows.append(row)
            return rows, original_size_after_ind, trimmed_size, blocks_removed


    # ======================
    # Auto-trim Size Detection
    # ======================
    auto_trim_size = None
    folder_path = None

    if option == "Local folder":
        folder_path = st.text_input("Enter LOCAL folder path containing CSV files:")
        if folder_path and os.path.isdir(folder_path) and remove_blocks == "Automatically Fix to same size signal":
            sample_sizes = []
            check_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")][:4]

            for f in check_files:
                try:
                    df_tmp = pd.read_csv(f)
                    if indicator_choice != "both":
                        df_tmp = df_tmp[df_tmp["indicator"] == int(indicator_choice)]
                    sample_sizes.append(len(df_tmp))
                except Exception:
                    continue

            if sample_sizes:
                block = 2048
                auto_trim_size = (min(sample_sizes) // block) * block
                st.info(f"📏 Auto-trim detected minimum size (after indicator filter): {auto_trim_size} rows")


    # ======================
    # Local folder processing
    # ======================
    if option == "Local folder":
        if st.button("⏵ Generate from Local Folder"):
            if folder_path and os.path.isdir(folder_path):
                # ✅ compute Signal Zero once (not for every file)
                zero_means = None
                if apply_signal_zero and folder_path and os.path.isdir(folder_path):
                    zero_means = get_signal_zero_means(folder_path, indicator_choice, auto_trim_size)
                with st.expander("**💻 Output logs:**", expanded=False):
                    for file in os.listdir(folder_path):
                        if file.endswith(".csv"):
                            try:
                                df = pd.read_csv(os.path.join(folder_path, file))

                                # ✅ pass zero_means into process_dataframe
                                out, orig_size_after_ind, trimmed_size, blocks_removed = process_dataframe(
                                    df, file, auto_trim_size, zero_means
                                )

                                if out:
                                    results.extend(out)
                                    st.success(
                                        f"✅ Processed {file} - original size: **{orig_size_after_ind}**, "
                                        f"trimmed size: **{trimmed_size}**, block removed (2048): **{blocks_removed}**"
                                    )

                            except Exception as e:
                                st.error(f"⚠️ Error in {file}: {e}")
            else:
                st.error(f"⚠️ Please select the source path")
    # ======================
    # Uploaded file processing
    # ======================
    else:
        uploaded_files = st.file_uploader("Upload CSVs", type="csv", accept_multiple_files=True)
        if uploaded_files and st.button("Generate from Uploaded Files"):
            st.info(f"📂 {len(uploaded_files)} file(s) uploaded")
            zero_means = None
            if apply_signal_zero and folder_path and os.path.isdir(folder_path):
                zero_means = get_signal_zero_means(folder_path, indicator_choice, auto_trim_size)
            with st.expander("**💻 Output logs:**", expanded=False):
                for f in uploaded_files:
                    try:
                        df = pd.read_csv(f)
                        out, orig_size_after_ind, trimmed_size, blocks_removed = process_dataframe(
                                        df, f.name, auto_trim_size, zero_means
                                    )
                        if out:
                            results.extend(out)
                            st.success(
                                f"✅ Processed {f.name} - original size: **{orig_size_after_ind}**, "
                                f"trimmed size: **{trimmed_size}**, block removed (2048): **{blocks_removed}**"
                            )           
                    except Exception as e:
                        st.error(f"⚠️ Error in {f.name}: {e}")

    # ======================
    # Save / Show results
    # ======================
    if results:
        output_df = pd.DataFrame(results)
        st.dataframe(output_df)
        csv_bytes = output_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Dataset", csv_bytes, "dataset_with_features.csv", "text/csv")
