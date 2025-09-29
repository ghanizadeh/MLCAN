import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from io import StringIO

# -----------------------------
# Clean Cnergreen Dataset
# -----------------------------
def clean_df_cner(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 1. Remove '', 'NA', 'NaN', 'nan', None, "Â" from all cells
    out = out.replace(['', 'NA', 'NaN', 'nan', None, "Â"], np.nan)

    # 2. Remove "*" from all cells
    if out.astype(str).apply(lambda col: col.str.contains(r"\*")).any().any():
        out = out.replace(r"\*", "", regex=True)
        st.info("✨ Removed '*' characters from cells.")

    # 3. Convert specific columns (% , ppm , (cc)) to float
    cols_to_convert = [c for c in out.columns if any(x in c.lower() for x in ["%", "ppm", "(cc)"])]
    if cols_to_convert:
        for col in cols_to_convert:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float).round(2)
        st.info(f"🔄 Converted {len(cols_to_convert)} column(s) to Numeric (float):  \n{', '.join(cols_to_convert)}")

    # 4. Temperature / Tempreture / Temp
    for c in out.columns:
        if any(key in c.lower() for key in ["temperature", "tempreture", "temp"]):
            # Replace RT → 21
            out[c] = out[c].apply(lambda x: 21 if isinstance(x, str) and x.strip().lower() == "rt" else x)
            # Extract numeric part
            out[c] = out[c].astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
            # Convert to numeric
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(21)
            st.info(f"🌡️ Temperature column '{c}' cleaned (RT and NaN → 21)")

    if not cols_to_convert and not out.astype(str).apply(lambda col: col.str.contains(r"\*")).any().any():
        st.success("✅ No cleaning needed. Dataset already clean.")

    # --- 5. Handle % or ppm columns: NaN/"" → 0
    perc_cols = [c for c in out.columns if "%" in c.lower() or "ppm" in c.lower()]
    total_before = 0
    for col in perc_cols:
        before = out[col].isna().sum() + (out[col].astype(str) == "").sum()
        total_before += before
        out[col] = out[col].replace("", np.nan).fillna(0)
    if total_before > 0 and perc_cols:
        st.info(f"🧾 All %/ppm columns: empty/NaN values converted to 0.")

    # --- 6. Clean 'Concentrate Manufacturing Method (Ratio)' ---
    ratio_cols = [c for c in out.columns if "concentrate manufacturing method (ratio)" in c.lower()]
    for col in ratio_cols:
        def clean_ratio(val):
            if pd.isna(val):
                return 1
            s = str(val).lower()
            if "1 stream" in s:
                return 1
            if "2 stream" in s:
                return 2
            if "3:" in s:
                return 2
            digits = ''.join([ch for ch in s if ch.isdigit()])
            return int(digits) if digits else 1
        out[col] = out[col].apply(clean_ratio)
        st.info(f"⚗️ Column '{col}' cleaned (kept only digit).")

    # --- 7. Clean 'Dilution Ratio' ---
    dilution_cols = [c for c in out.columns if "dilution ratio" in c.lower()]
    for col in dilution_cols:
        def clean_dilution(val):
            if pd.isna(val):
                return np.nan
            s = str(val).lower()
            match = re.search(r'(\d+)\s*[xX]', s)
            if match:
                return int(match.group(1))
            match = re.search(r'[xX]\s*(\d+)', s)
            if match:
                return int(match.group(1))
            digits = ''.join([ch for ch in s if ch.isdigit()])
            return int(digits) if digits else np.nan
        out[col] = out[col].apply(clean_dilution)
        st.info(f"🧪 Column '{col}' cleaned (kept only digit).")

    # --- 8. Standardize 'Brine Type' ---
    brine_cols = [c for c in out.columns if "brine type" in c.lower()]
    for col in brine_cols:
        before = out[col].astype(str).copy()
        def clean_brine(val):
            if pd.isna(val):
                return "Field"
            s = str(val).lower()
            if "field" in s or "yates" in s:
                return "Field"
            if "synthetic" in s:
                return "Synthetic"
            return val
        out[col] = out[col].apply(clean_brine)
        changed = (before != out[col].astype(str)).sum()
        if changed > 0:
            st.info(f"🌊 Column '{col}': {changed} values standardized (Field/Synthetic).")

    # --- 9. Standardize Stability Columns ---
    stability_cols = [
        c for c in out.columns
        if any(key in c.lower() for key in [
            "concentrate stability (8c)",
            "concentrate stability (4c)",
            "concentrate stability (ht)",
            "dilution stability"
        ])
    ]
    for col in stability_cols:
        def clean_stability(val):
            if pd.isna(val):
                return "Unknown"
            s = str(val).strip().lower()
            if "false" in s:
                return "False"
            if "true" in s:
                return "True"
            return "Unknown"
        out[col] = out[col].apply(clean_stability)
        st.info(f"🧪 Column '{col}' standardized to (True/False/Unknown).")

    return out


# --------------------------------------
# Strip Filename from Format (drop .csv)
# --------------------------------------
def clean_filename(name: str) -> str:
    """Return a tidy filename stem (no extension), safe for a column value."""
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-]+", "_", stem).strip("_")
    return stem or "unknown"
    

# --------------------------------------
# Page: Data Cleaning
# --------------------------------------
def data_cleaning_page():
    #st.subheader("🧽 Data Cleaning")
    
    # -------------------------------------------
    # Help: Data Cleaning Tasks
    # -------------------------------------------
    with st.expander("ℹ️ Help: What Data Cleaning Does?", expanded=False):
        st.markdown("""
    - **General Cleaning**
        - Removes placeholders: `''`, `NA`, `NaN`, `nan`, `None`, and stray `Â`.
        - Strips `*` characters.
        - Converts `%`, `ppm`, and `(cc)` columns to **float**.
        - Fills empty/NaN values in **%/ppm** columns with `0`.

    - **Temperature Normalization**
        - Converts shorthand `RT` → `21 °C`.
        - Extracts numeric values only.

    - **Column-specific Cleaning**
        - **Concentrate Manufacturing Method (Ratio):** keeps only the digit.
        - **Dilution Ratio:** extracts the digit (e.g., `1x`, `x4` → `1`, `4`).
        - **Brine Type:** standardized to `Field` or `Synthetic`.

    - **Stability Columns Standardization (4C/8C/HT/Dilution stability) **
        - Missing values (`NaN`) → `"Unknown"`.
        - Values containing `"true"` → `"True"`.
        - Values containing `"false"` → `"False"`.
        - Anything else → `"Unknown"`.
        - **Final set:** `True` / `False` / `Unknown`.
        """)

    # -------------------------------------------
    # File Uploader
    # -------------------------------------------
    file = st.file_uploader(
        "📂 Upload your dataset (CSV)",
        type=["csv"],
        accept_multiple_files=False,
        help="Accepted format: CSV"
    )

    if file is not None:
        # Strip Filename from Format: drop .CSV
        stem = clean_filename(file.name)

        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Data Cleaning: Could not read the file: {e}")
            return

        st.success(f"✅ **Upload Successfull:** Filename: **{file.name}** - Size: **({df.shape[0]} rows, {df.shape[1]} cols)**")
        
        # Original Data Preview
        with st.expander("▦ Preview of the original dataset", expanded=False):
            st.dataframe(df, use_container_width=True)

        if st.button("🧼 Clean with clean_df_cner()", type="primary"):
            cleaned = clean_df_cner(df)
            st.success("🎉 Cleaning complete!")
            st.dataframe(cleaned.head(20), use_container_width=True)

            # Download cleaned CSV
            csv_buf = StringIO()
            cleaned.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download cleaned CSV",
                data=csv_buf.getvalue(),
                file_name=f"{stem}_cleaned.csv",
                mime="text/csv"
            )
 
