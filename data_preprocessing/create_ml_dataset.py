import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch
from scipy.fft import fft
import streamlit as st
import time

def move_std_mean_to_front(df: pd.DataFrame) -> pd.DataFrame:
    # Identify columns containing _std or _mean (case-insensitive)
    priority_cols = [
        col for col in df.columns
        if "_std" in col.lower() or "_mean" in col.lower()
    ]

    # Remaining columns
    other_cols = [col for col in df.columns if col not in priority_cols]

    # Reorder dataframe
    return df[priority_cols + other_cols]

# =========================================================
# 🧮 Feature Extractor
# =========================================================
def extract_features(df_window, selected_time, selected_freq, selected_cross, selected_slug, selected_sensors):
    feats = {}
    cols = selected_sensors

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
        norm_power = power / np.sum(power)
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

# =========================================================
# 🧭 Signal Zero Detection
# =========================================================
def get_signal_zero_means(folder_path, indicator_choice, auto_trim_size=None, sensors=None):
    zero_means = {}
    if sensors is None or len(sensors) < 2:
        return None

    for file in os.listdir(folder_path):
        if file.lower().endswith(".csv") and "alicat0.0" in file.lower() and "vfd0.0" in file.lower():
            try:
                df = pd.read_csv(os.path.join(folder_path, file))
                if indicator_choice != "both" and "indicator" in df.columns:
                    df = df[df["indicator"] == int(indicator_choice)]
                if auto_trim_size and len(df) > auto_trim_size:
                    df = df.iloc[:auto_trim_size]
                for s in sensors[:2]:
                    if s in df.columns:
                        zero_means[s] = df[s].mean()
                st.info(f"📏 Signal Zero detected from `{file}` using {list(zero_means.keys())}")
                return zero_means
            except Exception as e:
                st.error(f"⚠️ Could not read {file}: {e}")
    st.warning("⚠️ No Signal Zero file found (AliCat0.0 & VFD0.0).")
    return None

# =========================================================
# 🧩 Main Streamlit Function
# =========================================================
def show_create_ML_dataset():
    if "results" not in st.session_state:
        st.session_state.results = []
    results = st.session_state.results

    # =====================================================
    # 📂 Source Selection (first)
    # =====================================================
    st.divider()
    st.markdown("### 📂 Select Source Data")
    #option = st.radio("Choose source:", ["Upload CSVs", "Local Folder"], index=0)
    option = st.radio("Choose source:", [ "Local Folder"], index=0)

    folder_path = None
    uploaded_files = []

    if option == "Local Folder":
        folder_path = st.text_input("Enter LOCAL folder path containing CSV files:")
    else:
        uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

    # =====================================================
    # 🔍 Auto-detect Columns from first file
    # =====================================================
    st.divider()
    st.markdown("### 📡 Select Sensors and Targets")
    uploaded_sample, folder_sample = None, None
    sample_df, sample_name = None, None

    if option == "Upload CSVs" and uploaded_files:
        uploaded_sample = pd.read_csv(uploaded_files[0], nrows=5)
        sample_df, sample_name = uploaded_sample, uploaded_files[0].name
    elif option == "Local Folder" and folder_path and os.path.isdir(folder_path):
        csvs = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
        if csvs:
            folder_sample = pd.read_csv(csvs[0], nrows=5)
            sample_df, sample_name = folder_sample, os.path.basename(csvs[0])

    if sample_df is not None:
        all_cols = list(sample_df.columns)
        numeric_cols = [c for c in all_cols if np.issubdtype(sample_df[c].dtype, np.number)]
        st.success(f"✅ Columns auto-detected from `{sample_name}` ({len(all_cols)} total columns)")
    else:
        numeric_cols = []
        st.warning("⚠️ No file detected yet — please upload or enter a valid folder path.")

    possible_sensors = [c for c in numeric_cols if any(x in c.lower() for x in ["sensor", "pt-", "pressure"])]
    possible_targets = [c for c in numeric_cols if any(x in c.lower() for x in ["crl", "alicat", "qg", "ql","qo", "qw", "qgst", "qost", "qwst"])]

    selected_sensors = st.multiselect(
        "Select sensor columns:",
        options= numeric_cols,
        #options=possible_sensors if possible_sensors else numeric_cols,
        default=possible_sensors[:6] if len(possible_sensors) >= 6 else possible_sensors,
    )

    selected_targets = st.multiselect(
        "Select target columns:",
        options=possible_targets if possible_targets else numeric_cols,
        default=possible_targets[:2] if len(possible_targets) >= 2 else [],
    )

    target_method = st.radio("How to compute target per window?", ["Median", "Mean"], index=1)

    # =====================================================
    # ⚙️ Feature Options
    # =====================================================
    st.divider()
    st.markdown("### ⚙️ Feature Options")
    selected_time = st.multiselect("Time-domain features:",
        ["mean","std","min","max","median","range","skew","kurtosis","rms","p25","p75","iqr","cv","entropy"],
        default=["mean","std"])
    selected_freq = st.multiselect("Frequency-domain features:",
        ["dom_freq","spec_centroid","spec_bw","total_power","low_power","mid_power","high_power","spectral_entropy",
         "fft_energy","fft_entropy","fft_peak_mag","fft_peak_freq"],
        default=["dom_freq","spec_centroid"])
    selected_cross = st.multiselect("Cross-sensor features:", ["dp","corr","phase"], default=["corr"])
    selected_slug = st.multiselect("Slugging features:",
        ["peak_count","peak_rate","crest_factor","peak_to_mean","zero_cross"],
        default=["peak_count","peak_rate"])

    # =====================================================
    # 🪟 Sliding Window Options
    # =====================================================
    st.divider()
    st.markdown("### 🪟 Sliding Window Options")
    use_sliding = st.radio("Apply Sliding Window?", ["No - Use Entire Signal", "Yes - Apply Sliding Windows"], index=0)
    if use_sliding == "Yes - Apply Sliding Windows":
        fs = st.number_input("Sampling frequency (Hz)", min_value=1, value=100)
        window_size = st.number_input("Window size (seconds)", min_value=1, value=2)
        overlap_pct = st.slider("Overlap (%)", 0, 90, 50)
        overlap = overlap_pct / 100
    else:
        fs, window_size, overlap = None, None, None

    apply_signal_zero = st.checkbox("Apply Signal Zero adjustment", value=False)
    indicator_choice = st.radio("Select indicator:", [1, 0, "both"], index=0)
    remove_blocks = st.selectbox("Remove trailing blocks (2048 samples)", ["None", "Auto-adjust to uniform signal length"])

    # =====================================================
    # 🧠 Inner helper for dataframe processing
    # =====================================================
    def process_dataframe(df, filename, auto_trim_size=None, zero_means=None):
        usable_targets = selected_targets if selected_targets else [
            c for c in df.columns if any(x in c.lower() for x in ["crl", "alicat", "qg", "ql", "qw","qo", "qgst", "qost", "qwst"])
        ]
        if "indicator" in df.columns and indicator_choice != "both":
            df = df[df["indicator"] == int(indicator_choice)]
        if df.empty:
            return [], 0, 0, 0

        target_fn = np.median if target_method == "Median" else np.mean
        original_size = len(df)
        trimmed_size = len(df)
        blocks_removed = 0
        if auto_trim_size and len(df) > auto_trim_size:
            df = df.iloc[:auto_trim_size]
            trimmed_size = len(df)
            blocks_removed = (original_size - trimmed_size) // 2048

        # --- No Sliding ---
        if use_sliding == "No - Use Entire Signal":
            feats = extract_features(df, selected_time, selected_freq, selected_cross, selected_slug, selected_sensors)
            if zero_means and "mean" in selected_time:
                for s in zero_means:
                    k = f"TIME_{s}_mean"
                    if k in feats:
                        feats[k] -= zero_means[s]
            row = {"Filename": filename}
            for t in usable_targets:
                col = pd.to_numeric(df[t], errors="coerce").dropna()
                row[f"{t}_target"] = target_fn(col) if not col.empty else np.nan
            row.update(feats)
            return [row], original_size, trimmed_size, blocks_removed

        # --- Sliding ---
        else:
            size = int(window_size * fs)
            step = int(size * (1 - overlap))
            starts = range(0, max(len(df)-size+1, 0), step)
            rows = []
            for i, s in enumerate(starts):
                win = df.iloc[s:s+size]
                feats = extract_features(win, selected_time, selected_freq, selected_cross, selected_slug, selected_sensors)
                if zero_means and "mean" in selected_time:
                    for k,v in zero_means.items():
                        key = f"TIME_{k}_mean"
                        if key in feats:
                            feats[key] -= v
                row = {"Filename": filename, "Window": i}
                for t in usable_targets:
                    row[f"{t}_target"] = target_fn(win[t].dropna())
                row.update(feats)
                rows.append(row)
            return rows, original_size, trimmed_size, blocks_removed

    # =====================================================
    # 🚀 Generate Dataset (Local or Upload)
    # =====================================================
    auto_trim_size = None

    if option == "Local Folder":
        if folder_path and os.path.isdir(folder_path):
            if remove_blocks == "Auto-adjust to uniform signal length":
                csvs = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
                sizes = []
                for f in csvs[:4]:
                    try:
                        df_tmp = pd.read_csv(f)
                        if "indicator" in df_tmp.columns and indicator_choice != "both":
                            df_tmp = df_tmp[df_tmp["indicator"] == int(indicator_choice)]
                        sizes.append(len(df_tmp))
                    except:
                        pass
                if sizes:
                    block = 2048
                    auto_trim_size = (min(sizes) // block) * block
                    st.info(f"📏 Auto-trim detected: {auto_trim_size} rows")

        if st.button("⏵ Generate from Local Folder"):
            if not folder_path or not os.path.isdir(folder_path):
                st.error("⚠️ Invalid folder path.")
            else:
                zero_means = get_signal_zero_means(folder_path, indicator_choice, auto_trim_size, selected_sensors) if apply_signal_zero else None
                files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
                total = len(files)
                pb = st.progress(0)

                status_text = st.empty()  # placeholder for dynamic status messages
                for i, file in enumerate(files, 1):
                    #df = pd.read_csv(os.path.join(folder_path, file))
                    file_path = os.path.join(folder_path, file)
                    try:
                        # Read first few lines to detect columns
                        preview = pd.read_csv(file_path, nrows=10)
                        expected_cols = len(preview.columns)

                        # Stream-read in chunks
                        chunks = []
                        for chunk in pd.read_csv(file_path, chunksize=50000, low_memory=False):
                            # Drop malformed rows with unexpected column counts
                            if chunk.shape[1] == expected_cols:
                                chunks.append(chunk)
                        if chunks:
                            df = pd.concat(chunks, ignore_index=True)
                        else:
                            st.warning(f"⚠️ Skipped {file} (no valid chunks)")
                            continue

                        # Optional dtype conversion to reduce memory
                        df = df.convert_dtypes()

                    except pd.errors.ParserError as e:
                        st.warning(f"⚠️ Skipped {file} (parse error): {e}")
                        continue
                    except MemoryError:
                        st.error(f"💥 Out of memory while reading {file}. Try reducing chunksize or file size.")
                        continue
                    except Exception as e:
                        st.error(f"❌ Error reading {file}: {e}")
                        continue


                    # Show dynamic status
                    status_text.info(
                        f"🔄 Processing file **{i}/{total}**: **{file}** — "
                        f"size {df.shape}, window size: {window_size if window_size else 'full'}s, "
                        f"overlap: {int(overlap*100) if overlap else 0}%"
                    )

                    # Process data
                    out, _, _, _ = process_dataframe(df, file, auto_trim_size, zero_means)
                    if out:
                        results.extend(out)

                    # Update progress bar
                    pb.progress(i / total)
                    time.sleep(0.1)  # optional small delay for smoother UI feedback

                status_text.success("✅ All files processed successfully!")


    else:  # Uploaded files
        if uploaded_files and st.button("Generate from Uploaded Files"):
            for f in uploaded_files:
                df = pd.read_csv(f)
                out, _, _, _ = process_dataframe(df, f.name)
                if out:
                    results.extend(out)
            st.success("✅ All uploaded files processed!")

    # =====================================================
    # 💾 Export Results
    # =====================================================
    if results:
        st.success("✅ All uploaded files processed!")
        output_df = pd.DataFrame(results)
        st.markdown("### 📦 Final Dataset")
        st.dataframe(output_df)
        output_df = move_std_mean_to_front(output_df)

        csv_data = output_df.to_csv(index=False).encode("utf-8")
        filename = f"ML_dataset_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        st.download_button("⬇️ Download CSV", data=csv_data, file_name=filename, mime="text/csv")

    if st.session_state.results and st.button("🗑️ Clear Results"):
        st.session_state.results = []
        st.rerun()
