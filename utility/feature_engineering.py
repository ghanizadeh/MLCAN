import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os

# ==========================================================
# Page Config
# ==========================================================
def add_density_viscosity(
    df: pd.DataFrame,
    density_val: float | None = None,
    viscosity_val: float | None = None,
    overwrite: bool = False
) -> pd.DataFrame:

    out = df.copy()

    # --- Density ---
    if density_val is not None:
        if overwrite or "Density" not in out.columns:
            out["Density"] = density_val

    # --- Viscosity ---
    if viscosity_val is not None:
        if overwrite or "Viscosity" not in out.columns:
            out["Viscosity"] = viscosity_val

    # --- Viscosity–Temperature relationship ---
    if "Sensor1_Temp" in out.columns:
        out["Viscosity-Temp"] = (
            0.695 * out["Sensor1_Temp"] ** 2
            - 70.89 * out["Sensor1_Temp"]
            + 1997.5
        )

    return out

def add_flow_rates(
    df: pd.DataFrame,
    ql_col: str,
    wc_col: str | None = None,
    wc_value: float | None = None,
    rename_gas_from: str | None = None,
    remove_original: bool = False,
) -> pd.DataFrame:
    out = df.copy()

    # --- Safety ---
    if ql_col not in out.columns:
        return out

    out[ql_col] = pd.to_numeric(out[ql_col], errors="coerce")

    # --- Water cut ---
    if wc_col and wc_col in out.columns:
        wc = pd.to_numeric(out[wc_col], errors="coerce") / 100.0
    elif wc_value is not None:
        wc = pd.Series(wc_value / 100.0, index=out.index)
        out["WC"] = wc_value
        wc_col = "WC"
    else:
        return out

    wc = wc.clip(0, 1)

    # --- Compute flows ---
    out["Qw"] = out[ql_col] * wc
    out["Qo"] = out[ql_col] * (1 - wc)

    # --- Optional gas rename ---
    if rename_gas_from:
        for c in out.columns:
            if c.strip().lower() == rename_gas_from.lower():
                out.rename(columns={c: "Qg"}, inplace=True)

    # --- Cleanup ---
    if remove_original:
        drop_cols = [ql_col, wc_col]
        out.drop(columns=[c for c in drop_cols if c in out.columns], inplace=True)

    return out


# ==========================================================
# Helper functions
# ==========================================================
def nearest(value, candidates):
    """Return nearest value from candidates"""
    return min(candidates, key=lambda x: abs(x - value))


def lookup_pseudopressure(P, T, pt_table):
    pressures = pt_table.index.values
    temperatures = pt_table.columns.values

    Pn = nearest(P, pressures)
    Tn = nearest(T, temperatures)

    return pt_table.loc[Pn, Tn]

import numpy as np
import pandas as pd

def add_pressure_derived_features(
    df: pd.DataFrame,
    P_col: str = "Sensor2",
    dp1_col: str = "Sensor4",
    dp2_col: str = "Sensor6",
    T_col: str = "Sensor1_Temp",
    temp_unit: str = "C",
    epsP: float = 0,
    add_logs: bool = True,
    clip_negative_dp: bool = False
) -> pd.DataFrame:

    required = [P_col, dp1_col, dp2_col, T_col]
    if not all(c in df.columns for c in required):
        return df

    out = df.copy()

    # --- Numeric safety ---
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    dp1 = out[dp1_col].to_numpy()
    dp2 = out[dp2_col].to_numpy()
    if clip_negative_dp:
        dp1 = np.maximum(dp1, 0.0)
        dp2 = np.maximum(dp2, 0.0)

    P = out[P_col].to_numpy()
    T = out[T_col].to_numpy()

    P_safe = P + epsP
    T_safe = T + epsP

    # ==========================================================
    # B) Pressure structure
    # ==========================================================
    dptot  = dp1 + dp2
    dpdiff = dp1 - dp2

    out["DP_total"] = dptot
    out["DP_diff"]  = dpdiff
    out["DP_ratio"] = (dp1 + epsP) / (dp2 + epsP)
    out["DP_asym_index"] = dpdiff / (np.abs(dptot) + epsP)

    # ✅ NEW 1–2: dominance fractions
    out["DP1_fraction"] = dp1 / (dptot + epsP)
    out["DP2_fraction"] = dp2 / (dptot + epsP)

    # ✅ NEW 3: bounded asymmetry
    out["DP_asym_tanh"] = np.tanh(out["DP_asym_index"].to_numpy())

    # ==========================================================
    # C) Loss & normalization
    # ==========================================================
    out["Loss_Coefficient"] = dptot / P_safe

    out["DP1_over_P"]    = dp1 / P_safe
    out["DP2_over_P"]    = dp2 / P_safe
    out["DPtot_over_P"]  = dptot / P_safe
    out["DPdiff_over_P"] = dpdiff / P_safe
    out["LossCoeff_over_P"] = out["Loss_Coefficient"] / P_safe

    # ✅ NEW 4–5: stronger pressure-invariant forms
    out["DP_total_over_P2"] = dptot / (P_safe ** 2)
    out["LossCoeff_over_P2"] = out["Loss_Coefficient"] / (P_safe ** 2)

    # ==========================================================
    # D) Compressibility proxy
    # ==========================================================
    out["Temp_C"] = T
    out["P_over_T"] = P_safe / T_safe
    out["rho_proxy"] = out["P_over_T"]
    out["loss_norm"] = dptot / (out["rho_proxy"] + 1e-6)

    # ==========================================================
    # E) Log-space stabilization
    # ==========================================================
    if add_logs:
        out["logP"] = np.log1p(np.maximum(P, 0))
        out["logDP_total"] = np.log1p(np.abs(dptot))
        out["logDPtot_over_P"] = np.log1p(np.abs(out["DPtot_over_P"]))
        out["logDP_ratio"] = np.log((dp1 + epsP) / (dp2 + epsP))

        # ✅ NEW 7–8: interaction terms
        out["logP_x_logDP_total"] = out["logP"] * out["logDP_total"]
        out["logP_x_DP_ratio"] = out["logP"] * out["DP_ratio"]

    # ==========================================================
    # F) Binary regime indicators (tree-friendly)
    # ==========================================================
    # ✅ NEW 9–10
    out["is_DP1_dominant"] = (dp1 > dp2).astype(int)
    out["is_high_asym"] = (np.abs(out["DP_asym_index"]) > 0.3).astype(int)

    return out

# ==========================================================
# 1️⃣ Source Folder
# ==========================================================
def show_pseudopressure_app():
    st.subheader("📂 Source CSV Folder")

    source_dir = st.text_input(
        "Enter full path to folder containing CSV files",
        placeholder="e.g. D:/data/flow_runs/"
    )

    if source_dir and not os.path.isdir(source_dir):
        st.error("❌ Invalid folder path")
        st.stop()

    # ==========================================================
    # 1️⃣b Destination Folder
    # ==========================================================
    st.subheader("📁 Destination Folder (Output)")

    dest_dir = st.text_input(
        "Enter full path to destination folder",
        placeholder="e.g. D:/data/flow_runs_processed/"
    )

    if dest_dir:
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
    # ==========================================================
    # 2️⃣ Upload P–T Table
    # ==========================================================
    st.subheader("📤 Upload P–T Lookup Table")

    pt_file = st.file_uploader(
        "Upload P–T CSV (Pressure rows × Temperature columns)",
        type=["csv"]
    )

    pt_df = None
    if pt_file:
        pt_df = pd.read_csv(pt_file, index_col=0)
        try:
            pt_df.index = pt_df.index.astype(float)
            pt_df.columns = pt_df.columns.astype(float)
            pt_df = pt_df.astype(float)
        except Exception:
            st.error("❌ P–T file must contain numeric pressures and temperatures")
            st.stop()

        st.success("✅ P–T table loaded successfully")
        st.dataframe(pt_df.iloc[:5, :5])

        # ==========================================================
        # 4️⃣ Flow Rate Calculation (Optional)
        # ==========================================================
        st.subheader("💧 Flow Rate Calculation (Optional)")

        numeric_cols = []

        if source_dir:
            sample_files = list(Path(source_dir).glob("*.csv"))
            if sample_files:
                numeric_cols = pd.read_csv(sample_files[0]).select_dtypes(
                    include="number"
                ).columns.tolist()

        enable_flow = st.checkbox("➕ Add Qw / Qo / Qg to dataset", value=False)

        if enable_flow:

            ql_col = st.selectbox(
                "Select Liquid Rate (Ql) column",
                numeric_cols,
                index=0
            )

            wc_mode = st.radio(
                "Water Cut input method:",
                ["Select existing Water Cut column", "Enter constant Water Cut (%)"],
                horizontal=True
            )

            wc_col = None
            wc_value = None

            if wc_mode == "Select existing Water Cut column":
                wc_col = st.selectbox(
                    "Select Water Cut column (%)",
                    numeric_cols
                )
            else:
                wc_value = st.number_input(
                    "Enter Water Cut (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    step=0.5
                )

            rename_gas = st.text_input(
                "Rename gas column to Qg (optional, e.g. 'alicat')",
                value=""
            )

            remove_original = st.checkbox(
                "🧹 Remove original Ql / WC columns after calculation",
                value=False
            )

        # ==========================================================
        # 🧪 Density & Viscosity Input (Optional)
        # ==========================================================
        st.subheader("🧪 Density & Viscosity Input")

        use_const_rheology = st.radio(
            "Density & Viscosity source:",
            [
                "Use existing columns (if available)",
                "Enter constant Density & Viscosity"
            ],
            horizontal=True
        )

        density_val = None
        viscosity_val = None

        if use_const_rheology == "Enter constant Density & Viscosity":
            density_val = st.number_input(
                "Enter Density value",
                min_value=0.0,
                step=0.0001,
                format="%.4f"
            )

            viscosity_val = st.number_input(
                "Enter Viscosity value",
                min_value=0.0,
                step=0.0001,
                format="%.4f"
            )

    # ==========================================================
    # ==========================================================
    # 3️⃣ Process CSV Files
    # ==========================================================
    if pt_df is not None and source_dir and dest_dir:
        
        st.subheader("🚀 Apply P–T Lookup to Source CSVs")

        if st.button("Process All CSV Files"):

            source_path = Path(source_dir)
            output_dir = Path(dest_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            csv_files = list(source_path.glob("*.csv"))

            if not csv_files:
                st.warning("⚠️ No CSV files found")
                st.stop()

            progress = st.progress(0)

            for i, csv_file in enumerate(csv_files):

                df = pd.read_csv(csv_file)
                 
                if {"Sensor2", "Sensor4", "Sensor6", "Sensor1_Temp"}.issubset(df.columns):
                    df = add_pressure_derived_features(
                            df,
                            P_col="Sensor2",
                            dp1_col="Sensor4",
                            dp2_col="Sensor6",
                            T_col="Sensor1_Temp",
                            temp_unit="C",   # change to "K" if your T is already Kelvin
                            epsP=0,
                            add_logs=True,
                            clip_negative_dp=False
                        )
                    df["PseudoPressure_after"] = df.apply(
                        lambda row: lookup_pseudopressure(
                            row["Sensor2"],
                            row["Sensor1_Temp"],
                            pt_df
                        ),
                        axis=1
                    )
                    df["DP_total_over_PseudoPressure"] = df["DP_total"] / (df["PseudoPressure_after"] + 1e-6)


                elif {"PT_aftr", "deltaP", "Temp_aftr"}.issubset(df.columns):
                    st.info(f"Processing {csv_file.name} using PT_after / Temp_after columns")
                    df = add_pressure_derived_features(
                        df,
                        P_col="PT_aftr",
                        dp1_col="deltaP",
                        dp2_col="PT_aftr",   # intentional per your logic
                        T_col="Temp_aftr",
                        temp_unit="C",
                        epsP=0,
                        add_logs=True,
                        clip_negative_dp=False
                    )
                    df["PseudoPressure_after"] = df.apply(
                        lambda row: lookup_pseudopressure(
                            row["PT_aftr"],
                            row["Temp_aftr"],
                            pt_df
                        ),
                        axis=1
                    )
                    df["DP_total_over_PseudoPressure_after"] = df["DP_total"] / (df["PseudoPressure_after"] + 1e-6)
                    df["PseudoPressure_before"] = df.apply(
                        lambda row: lookup_pseudopressure(
                            row["deltaP"],
                            row["Temp_aftr"],
                            pt_df
                        ),
                        axis=1
                    )
                    df["DP_total_over_PseudoPressure_after"] = df["DP_total"] / (df["PseudoPressure_before"] + 1e-6)


                required_cols_1 = {"Sensor2", "Sensor1_Temp"}
                required_cols_2 = {"PT_aftr","deltaP"}
                if not required_cols_1.issubset(df.columns) and not required_cols_2.issubset(df.columns):
                    st.warning(
                        f"Skipping {csv_file.name} (missing Sensor1_Temp or Sensor2)"
                    )
                    continue


                # ======================================================
                # ➕ Add flow rates if enabled
                # ======================================================
                if enable_flow:
                    df = add_flow_rates(
                        df,
                        ql_col=ql_col,
                        wc_col=wc_col if wc_mode == "Select existing Water Cut column" else None,
                        wc_value=wc_value if wc_mode == "Enter constant Water Cut (%)" else None,
                        rename_gas_from=rename_gas if rename_gas else None,
                        remove_original=remove_original
                    )

                # ======================================================
                # 🧪 Add Density / Viscosity if requested
                # ======================================================
                if use_const_rheology == "Enter constant Density & Viscosity":
                    df = add_density_viscosity(
                        df,
                        density_val=density_val,
                        viscosity_val=viscosity_val,
                        overwrite=False   # <-- do NOT overwrite existing columns
                    )

                out_file = output_dir / f"{csv_file.stem}_with_pseudopressure.csv"
                # remove extra cols
                cols_to_remove = [
                    "AliCat_Output",
                    "VFD_Output",
                    "Relay_1",
                    "Relay_2",
                    "indicator",
                    "Density",
                    "Viscosity",
                ]

                df = df.drop(columns=cols_to_remove, errors="ignore")
                df.to_csv(out_file, index=False)

                progress.progress((i + 1) / len(csv_files))

            st.success("✅ All files processed successfully")
            st.info(f"📁 Files written to: {output_dir}")

            st.subheader("📄 Sample Output")
            st.dataframe(df.head())
