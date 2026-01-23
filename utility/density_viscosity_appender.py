import streamlit as st
import pandas as pd
import os
import glob

 
def show_density_viscosity_adder():

    # ---- User Inputs ----
    st.subheader("📁 Folder Paths")
    src_folder = st.text_input("Enter SOURCE folder path containing CSV files:")
    dst_folder = st.text_input("Enter DESTINATION folder path to save processed files:")

    density_val = st.number_input("Enter Density value:", step=0.0001, format="%.4f")
    viscosity_val = st.number_input("Enter Viscosity value:", step=0.0001, format="%.4f")

    process_btn = st.button("Process Files")

    # ---- Processing ----
    if process_btn:

        # Validate source folder
        if not os.path.isdir(src_folder):
            st.error("❌ Invalid SOURCE folder path.")
            return
        
        # Create destination folder if missing
        if not os.path.isdir(dst_folder):
            os.makedirs(dst_folder)
            st.info("📁 Destination folder created.")

        csv_files = glob.glob(os.path.join(src_folder, "*.csv"))

        if not csv_files:
            st.warning("⚠️ No CSV files found in the source folder.")
            return
        
        # Process each file
        for file in csv_files:
            df = pd.read_csv(file)

            # Add Density if missing
            if "Density" not in df.columns:
                df["Density"] = density_val

            # Add Viscosity if missing
            if "Viscosity" not in df.columns:
                df["Viscosity"] = viscosity_val

            # Add Viscosity-Temp if Sensor1_Temp exists
            if "Sensor1_Temp" in df.columns:
                df["Viscosity-Temp"] = (
                    0.695 * df["Sensor1_Temp"]**2
                    - 70.89 * df["Sensor1_Temp"]
                    + 1997.5
                )
            else:
                st.error(f"❌ 'Sensor1_Temp' missing in: {os.path.basename(file)}")
                continue

            # Save to destination folder
            filename = os.path.basename(file)
            save_path = os.path.join(dst_folder, filename)
            df.to_csv(save_path, index=False)

        st.success("✅ Processing complete! Files saved to destination folder.")
        st.info("ℹ️ 'Viscosity-Temp' added using: **0.695*T² − 70.89*T + 1997.5**")
 
