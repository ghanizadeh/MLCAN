import streamlit as st
import pandas as pd
import os
import re

# ============================================================
# 📌 Helper for parsing filename
# ============================================================
def parse_filename(filename):
    """Extract R, D, V numbers from filename (e.g. R1D2V3.csv)."""
    match = re.search(r'R(\d+)D(\d+)V(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), f"D{match.group(2)}", f"V{match.group(3)}"
    return None, None, None


# ============================================================
# 🚀 Streamlit App
# ============================================================
def show_density_viscosity_appender():

    st.title("🧪 Density / Viscosity Mapper for RxDiVj CSV Files")

    with st.expander("ℹ️ **About This Tool**"):
        st.markdown("""
            ### 🧠 Overview  

            This tool automatically appends **Density (SG)** and **Viscosity (cP)** values  
            to your RxDiVj CSV files by reading the encoded identifiers in each filename  
            (e.g., **R3D2V5.csv** → R=3, D=2, V=5).  

            The extracted codes are matched against your reference dataset to locate  
            the correct **Density (SG)** and **Viscosity (cP)** values, which are then  
            added as new columns to each CSV file.
        """)
 
    st.divider()

    # ============================================================
    # 1️⃣ Input Method
    # ============================================================
    st.header("1️⃣ Input Method")

    input_method = st.radio(
        "Choose how to load RxDiVj CSV files:",
        ["Enter source folder (recursive)", "Upload CSV files"],
        horizontal=False
    )

    ref_method = st.radio(
        "Choose how to load Density/Viscosity reference CSV:",
        ["Upload reference CSV", "Enter path to reference CSV or folder"]
    )

    # ============================================================
    # 2️⃣ Load reference CSV
    # ============================================================
    st.header("2️⃣ Load Reference Density/Viscosity File")
    ref_df = None

    if ref_method == "Upload reference CSV":
        ref_file = st.file_uploader("Upload reference CSV", type=["csv"])

        if ref_file:
            ref_df = pd.read_csv(ref_file)

    else:
        ref_path = st.text_input("Enter reference CSV file OR folder path:")

        if ref_path:
            if os.path.isdir(ref_path):
                inside = [f for f in os.listdir(ref_path) if f.lower().endswith(".csv")]
                if inside:
                    ref_df = pd.read_csv(os.path.join(ref_path, inside[0]))
                    st.success(f"Using: {inside[0]}")
                else:
                    st.error("❌ No CSV inside folder.")
            elif os.path.isfile(ref_path):
                ref_df = pd.read_csv(ref_path)
            else:
                st.error("❌ Invalid reference path.")

    # Build lookup maps
    density_map, viscosity_map = {}, {}

    if ref_df is not None:
        ref_df.columns = [c.strip() for c in ref_df.columns]

        for _, row in ref_df.iterrows():
            if pd.notna(row.get("Density")) and pd.notna(row.get("Density (SG)")):
                density_map[row["Density"]] = row["Density (SG)"]

            if pd.notna(row.get("Viscosity")) and pd.notna(row.get("Viscosity (cP)")):
                viscosity_map[row["Viscosity"]] = row["Viscosity (cP)"]

        st.success("Reference file loaded successfully.")
        st.json({"Density Map": density_map, "Viscosity Map": viscosity_map})

    st.divider()

    # ============================================================
    # 3️⃣ Load RxDiVj Files (Recursive or Upload)
    # ============================================================
    st.header("3️⃣ Load RxDiVj CSV Files")
    csv_files = []

    if input_method == "Enter source folder (recursive)":
        src_folder = st.text_input("Enter source folder path:")

        if src_folder and os.path.exists(src_folder):
            for root, _, files in os.walk(src_folder):
                for f in files:
                    if f.lower().endswith(".csv"):
                        csv_files.append(os.path.join(root, f))
            st.success(f"Found {len(csv_files)} CSV files.")
        elif src_folder:
            st.error("❌ Folder not found.")

    else:
        uploads = st.file_uploader("Upload 1 or more CSV files", type=["csv"], accept_multiple_files=True)
        if uploads:
            csv_files = uploads
            st.success(f"Uploaded {len(uploads)} files.")

    st.divider()

    # ============================================================
    # 4️⃣ Destination folder
    # ============================================================
    st.header("4️⃣ Destination Folder")

    dest_folder = st.text_input("Enter destination folder (will be created if not exist):")

    if dest_folder:
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder, exist_ok=True)
            st.success(f"📁 Created destination folder: {dest_folder}")
        else:
            st.info("📁 Using existing destination folder.")

    st.divider()

    # ============================================================
    # 5️⃣ PROCESS BUTTON
    # ============================================================
    if st.button("🚀 Process Files and Add Density / Viscosity"):
        if not ref_df:
            st.error("❌ Reference CSV not loaded.")
            return

        if not csv_files:
            st.error("❌ No RxDiVj files loaded.")
            return

        if not dest_folder:
            st.error("❌ Destination folder not provided.")
            return

        st.write("### 🔧 Processing...")

        progress = st.progress(0)
        status = st.empty()

        total = len(csv_files)

        for i, file_item in enumerate(csv_files, start=1):

            # Determine file path / uploaded file
            if isinstance(file_item, str):
                file_path = file_item
                df = pd.read_csv(file_path)
                fname = os.path.basename(file_path)
            else:
                df = pd.read_csv(file_item)
                fname = file_item.name

            R, D, V = parse_filename(fname)

            if None in (R, D, V):
                status.warning(f"Skipping (bad name): {fname}")
                continue

            dens = density_map.get(D)
            visc = viscosity_map.get(V)

            if dens is None or visc is None:
                status.warning(f"No match in reference: {fname}")
                continue

            # Insert columns
            df["Density (SG)"] = dens
            df["Viscosity (cP)"] = visc

            save_path = os.path.join(dest_folder, fname)
            df.to_csv(save_path, index=False)

            status.info(f"Processed: {fname} → Density={dens}, Viscosity={visc}")
            progress.progress(i / total)

        st.success("🎉 All done! Files saved in the destination folder.")

 