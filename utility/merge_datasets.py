import os
import pandas as pd
import numpy as np
import streamlit as st
import re

# ============================================================
# Memory-Safe Ordered Stream Merge (Single or Multi-Batch)
# ============================================================
def merge_csv_files_to_disk(
    file_list,
    selected_cols=None,
    percent=100,
    keep_source=False,
    output_path="merged_output.csv",
    batch_mode=False,
    batch_size=3,
    dest_folder=None,
    fs=100  # ✅ sampling frequency in Hz for time continuity
):
    """Merge large CSVs one-by-one directly into one or multiple CSVs without keeping all in RAM."""
    summaries = []

    def natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(s))]
    file_list = sorted(file_list, key=natural_key)
    total_files = len(file_list)

    if batch_mode and (not dest_folder or not os.path.isdir(dest_folder)):
        st.error("⚠️ Please specify a valid destination folder for batch merging.")
        return None, ["❌ Invalid destination folder."]

    progress = st.progress(0)
    status = st.empty()

    # ============================================================
    # Single Merge Mode (unchanged)
    # ============================================================
    if not batch_mode:
        if os.path.exists(output_path):
            os.remove(output_path)
        for i, file in enumerate(file_list, start=1):
            file_name = os.path.basename(file)
            try:
                preview = pd.read_csv(file, nrows=5000)
                dtype_map = preview.dtypes.to_dict()
                df = pd.read_csv(file, dtype=dtype_map)

                if selected_cols:
                    available_cols = [c for c in selected_cols if c in df.columns]
                    df = df[available_cols]

                rows_total = len(df)
                rows_to_pick = int(rows_total * (percent / 100))
                df = df.head(rows_to_pick)

                if keep_source:
                    df["source_file"] = file_name

                df.to_csv(
                    output_path,
                    mode="a",
                    index=False,
                    header=not os.path.exists(output_path) or os.path.getsize(output_path) == 0,
                )

                summaries.append(f"📄 {file_name} → {rows_to_pick}/{rows_total} rows appended")
            except Exception as e:
                summaries.append(f"❌ {file_name}: {e}")

            progress.progress(i / total_files)
            status.text(f"Appending {i}/{total_files}: {file_name}")

        st.divider()
        progress.empty()
        status.text("✅ Merge complete (written directly to disk)")
        return output_path, summaries

    # ============================================================
    # Batch Merge Mode (✨ fixed Timestamp continuity — no Second col)
    # ============================================================
    else:
        num_batches = (total_files + batch_size - 1) // batch_size
        summaries.append(f"📦 Batch merge mode enabled → {num_batches} output files")

        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, total_files)
            batch_files = file_list[start_idx:end_idx]
            batch_name = f"merged_batch_{b+1}.csv"
            batch_path = os.path.join(dest_folder, batch_name)

            # Remove existing batch file
            if os.path.exists(batch_path):
                os.remove(batch_path)

            #st.info(f"🧩 Merging Batch {b+1}/{num_batches}: {len(batch_files)} files → `{batch_name}`")

            header_written = False
            time_offset = 0.0  # ✅ ensures continuous time across files

            for i, file in enumerate(batch_files, start=start_idx + 1):
                file_name = os.path.basename(file)
                try:
                    preview = pd.read_csv(file, nrows=5000)
                    dtype_map = preview.dtypes.to_dict()
                    df = pd.read_csv(file, dtype=dtype_map)

                    # Select subset
                    if selected_cols:
                        available_cols = [c for c in selected_cols if c in df.columns]
                        df = df[available_cols]

                    rows_total = len(df)
                    rows_to_pick = int(rows_total * (percent / 100))
                    df = df.head(rows_to_pick)

                    # ✅ Ensure Timestamp column exists
                    if "Timestamp" not in df.columns:
                        raise ValueError(f"Missing 'Timestamp' column in {file_name}")

                    # ✅ Detect and convert timestamp format automatically
                    ts_sample = str(df["Timestamp"].dropna().iloc[0]) if df["Timestamp"].dropna().shape[0] > 0 else ""
                    if re.match(r"^\d+(\.\d+)?$", ts_sample):
                        # Numeric (seconds or milliseconds)
                        df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
                    elif re.match(r"^\d{2}:\d{2}:\d{2}", ts_sample):
                        # Time format HH:MM:SS.xxx
                        df["Timestamp"] = pd.to_timedelta(df["Timestamp"]).dt.total_seconds()
                    else:
                        # Datetime format
                        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                        df["Timestamp"] = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()

                    # ✅ Fill NaNs safely
                    df["Timestamp"] = df["Timestamp"].interpolate(limit_direction="both").bfill().ffill()

                    # ✅ Normalize to zero and apply cumulative offset
                    first_valid = df["Timestamp"].iloc[0] if not pd.isna(df["Timestamp"].iloc[0]) else 0.0
                    df["Timestamp"] = df["Timestamp"] - first_valid + time_offset

                    # ✅ Update offset for next file
                    time_offset = df["Timestamp"].iloc[-1] + (1 / fs)

                    if keep_source:
                        df["source_file"] = file_name

                    # ✅ Append to disk incrementally
                    df.to_csv(
                        batch_path,
                        mode="a",
                        index=False,
                        header=not header_written,
                    )
                    header_written = True

                    summaries.append(f"📄 {file_name} → {rows_to_pick}/{rows_total} rows → {batch_name}")

                except Exception as e:
                    summaries.append(f"❌ {file_name}: {e}")

                progress.progress(i / total_files)
                status.text(f"**Merging {i}/{total_files}**: {file_name}")

            summaries.append(f"✅ Batch {b+1} saved → {batch_name}")

        progress.empty()
        status.text("✅ All batches merged successfully")
        summaries.append(f"✅ All {num_batches} batch files saved in: {dest_folder}")
        return dest_folder, summaries


# ============================================================
# Streamlit Page
# ============================================================
def show_merge_page():
    st.title("🟫 Data Integration")
    
    with st.expander("ℹ️ **About This Tool**"):
        st.markdown("""
        ### 🧠 Overview  
        This app performs **memory-safe CSV appeding** by streaming data **directly to disk**, allowing you to combine thousands of files without exhausting RAM.  

        ### ⚙️ Available Modes  
        - 🧩 **Single Merge Mode** – Combines all CSV files into **one continuous output file** (default).  
        - 📦 **Batch Merge Mode** – Merges every **X sequential files** (sorted by filename) into separate outputs:  
        `merged_batch_1.csv`, `merged_batch_2.csv`, `merged_batch_3.csv`, …  

        ### 🪶 Key Features  
        - Processes files in **natural numerical order** (`file1 → file2 → file10`)  
        - Writes results **incrementally** to disk (no giant DataFrame in memory)  
        - Supports **column selection**, **sampling by percentage**, and an optional `source_file` column  
        - Ideal for **large-scale datasets** (sensor logs, time-series measurements, etc.)  
        """)

    st.divider()

    option = st.radio("Choose input method:", ["🖊️ Enter source path","📂 Upload CSV files"])
    percent = st.slider("Select % of each dataset to include", 10, 100, 100, step=10)
    keep_source = st.checkbox("Keep 'source_file' column", value=False)

    # ✅ New option: enable batch mode
    batch_mode = st.checkbox("Enable batch merging (X files → one output)")
    batch_size, dest_folder = None, None
    if batch_mode:
        batch_size = st.number_input("🧩 Number of files per batch (X):", min_value=2, value=3, step=1)
        dest_folder = st.text_input("Destination folder for batch outputs:")

    output_path = st.text_input("Output file path (for single merge):", "merged_output.csv")
    summary = []

    # ---------------- Local Folder Mode ----------------
    if option == "🖊️ Enter source path":
        folder_path = st.text_input("Enter source path:")
        if folder_path and os.path.exists(folder_path):
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
            if files:
                sample_df = pd.read_csv(files[0], nrows=100)
                all_cols = list(sample_df.columns)
                selected_cols = st.multiselect("Select columns:", all_cols, default=all_cols)

                if st.button("🔄 Merge Files"):
                    result, summary = merge_csv_files_to_disk(
                        files,
                        selected_cols,
                        percent,
                        keep_source,
                        output_path,
                        batch_mode,
                        batch_size,
                        dest_folder,
                    )
                    if batch_mode:
                        st.success(f"✅ Batch merged CSVs written in: **{os.path.abspath(result)}**")
                        st.dataframe(result)
                    else:
                        st.success(f"✅ Merged CSV written to: **{os.path.abspath(result)}**")
            else:
                st.warning("⚠️ No CSV files found in folder.")

    # ---------------- Upload Mode ----------------
    elif option == "📂 Upload CSV files":
        uploaded_files = st.file_uploader("Upload multiple CSV files", type=["csv"], accept_multiple_files=True)
        if uploaded_files and st.button("🔄 Merge Files"):
            tmp_dir = "temp_merge"
            os.makedirs(tmp_dir, exist_ok=True)
            file_paths = []
            for f in uploaded_files:
                tmp_path = os.path.join(tmp_dir, f.name)
                with open(tmp_path, "wb") as out:
                    out.write(f.read())
                file_paths.append(tmp_path)
            result, summary = merge_csv_files_to_disk(
                file_paths,
                None,
                percent,
                keep_source,
                output_path,
                batch_mode,
                batch_size,
                dest_folder,
            )
            if batch_mode:
                st.success(f"✅ Batch merged CSVs written in: **{os.path.abspath(result)}**")
                st.dataframe(result)
            else:
                st.success(f"✅ Merged CSV written to: **{os.path.abspath(result)}**")

    # --- Show summary ---
    if summary:
        with st.expander("✅ **Merge Summary**"):
            for s in summary:
                st.write(s)
        st.info("💾 File(s) written directly to disk — ready to open in Excel or Pandas.")
