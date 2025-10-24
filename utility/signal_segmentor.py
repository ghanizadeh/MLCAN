import os
import pandas as pd
import streamlit as st
from pathlib import Path
import random

# =========================================================
# 🔧 Utility: Cut Signal
# =========================================================
def cut_signal(df, start_time, duration, indicator_choice, filename, freq=None, auto_trim_size=None):
    df = df.copy()

    # 🔹 Filter by indicator
    if indicator_choice != "both":
        df = df[df["indicator"] == int(indicator_choice)]

    # 🔹 Auto-adjust signal length
    if auto_trim_size is not None and len(df) > auto_trim_size:
        df = df.iloc[:auto_trim_size]

    # 🔹 Frequency-based cutting (row index math)
    if freq is not None:
        start_idx = int(start_time * freq)
        end_idx = int(start_idx + duration * freq)
        cut_df = df.iloc[start_idx:end_idx].copy()
    else:
        # 🔹 Time-based cutting
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            if df["Timestamp"].notna().any():
                start_ts = df["Timestamp"].dropna().iloc[0]
                df["Seconds"] = (df["Timestamp"] - start_ts).dt.total_seconds()
            else:
                raise ValueError("Timestamps not usable.")
        else:
            raise ValueError(f"No 'Timestamp' column in {filename}, and no frequency provided.")

        cut_df = df[(df["Seconds"] >= start_time) &
                    (df["Seconds"] < start_time + duration)].copy()

    cut_df["source_file"] = filename
    return cut_df


# =========================================================
# 🎛️ Main Streamlit App
# =========================================================
def show_signal_segmentor():
    st.title("Signal Segmentor")
    st.markdown("---")
    # Choose folder
    with st.container(border=True):
        folder_path = st.text_input("📁 Enter **Source path** (csv files)")
        dest_folder = st.text_input("📁 Enter **Destination path** (leave empty for parent directory)")

    # --- User chooses segmentation plan ---
    with st.container(border=True):
        num_segments = st.number_input("**✂️ Number of Segments**", min_value=1, value=3)
 
        segment_mode = st.radio("**⏰ Segment Duration Mode**", ["Fixed Duration", "Random Duration [x, y]"])
 
        if segment_mode == "Fixed Duration":
            duration = st.number_input("**⏱ Each Segment Duration (seconds)**", min_value=1, value=10)
            random_bounds = None
        else:
            col1, col2 = st.columns(2)
            min_dur = col1.number_input("Min Duration (x)", min_value=1, value=5)
            max_dur = col2.number_input("Max Duration (y)", min_value=2, value=15)
            duration = None
            random_bounds = (min_dur, max_dur)

        indicator_choice = st.radio("**🔘 Select indicator**", ["0", "1", "both"])
        use_freq = st.checkbox("Use frequency instead of timestamps")
        freq = None
        if use_freq:
            freq = st.number_input("Enter signal frequency (Hz)", min_value=1, value=1000)

        # --- Auto-trim Option ---
        remove_blocks = st.selectbox(
            "🔧 **Signal length** Auto-adjustment  (1 block = 2048 samples)",
            ["None", "Auto-adjust to uniform signal length"]
        )

    auto_trim_size = None
    if folder_path and os.path.isdir(folder_path) and remove_blocks == "Auto-adjust to uniform signal length":
        sample_sizes = []
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        for f in csv_files[:5]:
            try:
                df_tmp = pd.read_csv(os.path.join(folder_path, f))
                if indicator_choice != "both":
                    df_tmp = df_tmp[df_tmp["indicator"] == int(indicator_choice)]
                sample_sizes.append(len(df_tmp))
            except Exception:
                continue

        if sample_sizes:
            block = 2048
            auto_trim_size = (min(sample_sizes) // block) * block
            st.success(f"📏 Auto-trim active — all signals will be trimmed to {auto_trim_size} samples.")

    # ------------------ PROCESS BUTTON ------------------ #
    if st.button("Process CSV files"):
        if not folder_path or not os.path.exists(folder_path):
            st.error("⚠️ Please enter a valid folder path.")
            st.stop()

        if not dest_folder:
            parent_dir = str(Path(folder_path).parent)
            dest_folder = os.path.join(parent_dir, f"{Path(folder_path).stem}_segments")
        os.makedirs(dest_folder, exist_ok=True)

        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        total_files = len(csv_files)
        if total_files == 0:
            st.warning("⚠️ No CSV files found in the selected folder.")
            st.stop()

        progress = st.progress(0)
        status = st.empty()

        processed_count = 0
        failed_count = 0

        for i, file_name in enumerate(csv_files):
            try:
                file_path = os.path.join(folder_path, file_name)
                status.text(f"Processing {i+1}/{total_files}: {file_name}")

                df = pd.read_csv(file_path)
                if "PT-A1" in df.columns:
                    rename_map = {
                            "PT-A1": "Sensor1",
                            "PT-A2": "Sensor2",
                            "PT-A3": "Sensor3",
                            "PT-A4": "Sensor4",
                            "PT-A5": "Sensor5",
                            "PT-A6": "Sensor6",
                            "wc-num": "Qw"
                    }
                    # Apply renaming
                    df.rename(columns=rename_map, inplace=True)
                for seg in range(num_segments):
                    # Random duration per segment if enabled
                    if random_bounds:
                        duration_val = random.uniform(*random_bounds)
                    else:
                        duration_val = duration

                    start_time = seg * duration_val
                    cut_df = cut_signal(
                        df,
                        start_time,
                        duration_val,
                        indicator_choice,
                        file_name,
                        freq=freq,
                        auto_trim_size=auto_trim_size
                    )

                    if not cut_df.empty:
                        out_name = f"{Path(file_name).stem}_seg{seg+1}_{round(duration_val,2)}s.csv"
                        out_path = os.path.join(dest_folder, out_name)
                        cut_df.to_csv(out_path, index=False)
                        processed_count += 1
                    else:
                        failed_count += 1

            except Exception as e:
                failed_count += 1
                st.error(f"❌ {file_name} failed: {e}")

            progress.progress((i + 1) / total_files)

        status.text(f"✅ Completed: {processed_count} segments processed ({failed_count} failed).")
        st.success(f"🎯 Output saved in `{dest_folder}`")
