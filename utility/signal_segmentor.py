import os
import pandas as pd
import streamlit as st
from pathlib import Path
import random

def cut_signal(df, start_time, duration, indicator_choice, filename, freq=None):
    df = df.copy()

    # 🔹 Filter by indicator
    if indicator_choice != "both":
        df = df[df["indicator"] == int(indicator_choice)]

    # 🔹 Frequency-based cutting (row index math)
    if freq is not None:
        start_idx = int(start_time * freq)
        end_idx   = int(start_idx + duration * freq)
        cut_df = df.iloc[start_idx:end_idx].copy()

    else:
        # 🔹 Time-based cutting
        if "Timestamp" in df.columns:
            try:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                if df["Timestamp"].notna().any():
                    start_ts = df["Timestamp"].dropna().iloc[0]
                    df["Seconds"] = (df["Timestamp"] - start_ts).dt.total_seconds()
                else:
                    raise ValueError("Timestamps not usable.")
            except Exception:
                raise ValueError(f"Timestamps not usable in {filename}, and no frequency provided.")
        else:
            raise ValueError(f"No 'Timestamp' column in {filename}, and no frequency provided.")

        cut_df = df[(df["Seconds"] >= start_time) &
                    (df["Seconds"] < start_time + duration)].copy()

    # 🔹 Add source file column
    cut_df["source_file"] = filename
    return cut_df



def show_signal_segmentor():
    # Choose folder (local or Google Drive path string)
    folder_path = st.text_input("Enter local or Google Drive folder path:")

    # Destination folder
    dest_folder = st.text_input("Enter destination folder (leave empty for parent directory):")

    # Cut options
    start_time = st.number_input("Start time (seconds)", min_value=0, value=0)

    # --- Duration Choice ---
    duration_mode = st.radio("Duration Mode", ["Fixed Duration", "Random Duration [x, y]"])
    if duration_mode == "Fixed Duration":
        duration = st.number_input("Duration (seconds)", min_value=1, value=10)
        random_bounds = None
    else:
        col1, col2 = st.columns(2)
        min_dur = col1.number_input("Min Duration (x)", min_value=1, value=5)
        max_dur = col2.number_input("Max Duration (y)", min_value=2, value=15)
        duration = None  # will be set later
        random_bounds = (min_dur, max_dur)

    indicator_choice = st.radio("Select indicator", ["0", "1", "both"])
    use_freq = st.checkbox("Use frequency instead of timestamps")
    freq = None
    if use_freq:
        freq = st.number_input("Enter signal frequency (Hz)", min_value=1, value=1000)    

    # ------------------ PROCESS BUTTON ------------------ #
    if st.button("Process CSV files"):
        if not folder_path or not os.path.exists(folder_path):
            st.error("⚠️ Please enter a valid folder path.")
            return

        # Default destination
        if not dest_folder:
            parent_dir = str(Path(folder_path).parent)
            dest_folder = os.path.join(parent_dir, f"{folder_path}_cut_signals")
        os.makedirs(dest_folder, exist_ok=True)

        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

        processed_count = 0
        for file_name in csv_files:
            try:
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)

                # If random mode, pick duration each time
                if random_bounds:
                    duration_val = random.uniform(*random_bounds)
                else:
                    duration_val = duration

                cut_df = cut_signal(
                    df,
                    start_time,
                    duration_val,
                    indicator_choice,
                    file_name,
                    freq=freq 
                )

                if not cut_df.empty:
                    #out_name = f"{Path(file_name).stem}_{start_time}_{round(start_time+duration_val,2)}_{round(duration_val,2)}_cut.csv"
                    out_name = f"{Path(file_name).stem}_{start_time}-{round(start_time+duration_val,2)}_{round(duration_val,2)}_cut.csv"

                    out_path = os.path.join(dest_folder, out_name)
                    cut_df.to_csv(out_path, index=False)
                    processed_count += 1
                    st.success(f"✅ Processed {file_name} -> {out_name}")
                else:
                    st.warning(f"⚠️ No matching data in {file_name}")

            except Exception as e:
                st.error(f"❌ Failed processing {file_name}: {e}")
        with st.expander("Logs"):
            st.info(f"🎯 Total processed files: {processed_count}/{len(csv_files)}")


    # ------------------ COMBINE BUTTON ------------------ #
    if st.button("Combine Segmented Signals"):
        if not dest_folder or not os.path.exists(dest_folder):
            st.error("⚠️ Destination folder not found. Please process files first.")
            return

        cut_files = [os.path.join(dest_folder, f) for f in os.listdir(dest_folder) if f.endswith("_cut.csv")]
        if not cut_files:
            st.warning("⚠️ No cut files found in destination folder.")
            return

        dfs = []
        # Find common columns across all dataframes
        common_cols = None

        with st.expander("Combine Logs"):
            for fpath in cut_files:
                try:
                    df = pd.read_csv(fpath)
                    if common_cols is None:
                        common_cols = set(df.columns)
                    else:
                        common_cols &= set(df.columns)

                    dfs.append(df)
                    st.write(f"✅ Included: {os.path.basename(fpath)} ({len(df)} rows)")
                except Exception as e:
                    st.error(f"❌ Failed reading {os.path.basename(fpath)}: {e}")

        if not common_cols:
            st.error("❌ No common columns found across cut files.")
            return

        # Convert to list for indexing
        common_cols = list(common_cols)

        combined_df = pd.concat([df[common_cols] for df in dfs], ignore_index=True)

        combined_path = os.path.join(dest_folder, "combined_segments.csv")
        combined_df.to_csv(combined_path, index=False)

        st.success(f"🎉 Combined {len(cut_files)} files -> combined_segments.csv")
        st.dataframe(combined_df)


