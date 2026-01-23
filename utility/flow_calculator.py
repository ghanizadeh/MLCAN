import streamlit as st
import pandas as pd
import os

def show_flow_calculator():
    st.title("🛢️ Flow Calculator (Qo / Qw)")

    # =========================================================
    # 1️⃣ Input Method Container
    # =========================================================
    with st.container():
        st.subheader("📥 Input Method")
        input_method = st.radio(
            "Choose input method:",
            ["Enter source & destination folders", "Upload files"],
            horizontal=False
        )

    # =========================================================
    # 2️⃣ Handle File Loading
    # =========================================================
    file_list = []
    dest_folder = None

    if input_method == "Enter source & destination folders":
        src_folder = st.text_input("📂 Enter source folder path (containing CSV files):")
        dest_folder = st.text_input("💾 Enter destination folder path (e.g., D:/New Folder):")

        if src_folder and os.path.exists(src_folder):
            file_list = [
                os.path.join(src_folder, f)
                for f in os.listdir(src_folder)
                if f.lower().endswith(".csv")
            ]
            st.success(f"✅ Found {len(file_list)} CSV files.")
        elif src_folder:
            st.error("❌ Source folder not found.")

    else:
        uploaded_files = st.file_uploader(
            "Upload one or more CSV files",
            type=["csv"],
            accept_multiple_files=True
        )
        if uploaded_files:
            file_list = uploaded_files
            st.success(f"✅ {len(uploaded_files)} files uploaded successfully.")
            dest_folder = st.text_input("💾 Enter destination folder path (e.g., D:/New Folder):")

    # =========================================================
    # 3️⃣ Processing Options
    # =========================================================
    if file_list:
        st.divider()
        st.subheader("⚙️ Processing Options")

        # Preview first file
        if isinstance(file_list[0], str):
            df_preview = pd.read_csv(file_list[0])
        else:
            df_preview = pd.read_csv(file_list[0])

        st.write("### 🔍 Preview of First File")
        st.dataframe(df_preview.head())

        numeric_cols = df_preview.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_cols:
            st.error("❌ No numeric columns found in the file.")
            return

        # QL selection
        ql_col = st.selectbox(
            "Select Q_Liquid column:",
            numeric_cols,
            key="ql_col"
        )

        # =========================================================
        # Water Cut Input Mode
        # =========================================================
        wc_mode = st.radio(
            "Water Cut input method:",
            ["Select existing Water Cut column", "Enter constant Water Cut (%)"],
            horizontal=True
        )

        wc_col = None
        wc_value = None

        if wc_mode == "Select existing Water Cut column":
            wc_col = st.selectbox(
                "Select Water Cut column:",
                numeric_cols,
                key="wc_col"
            )
        else:
            wc_value = st.number_input(
                "Enter Water Cut (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=0.5
            )
            wc_col = "WC"

        remove_original = st.checkbox(
            "🧹 Remove original Q_Liquid and Water Cut columns after calculation"
        )

        # =========================================================
        # 4️⃣ Process Files
        # =========================================================
        if st.button("🚀 Process All Files"):
            if not dest_folder:
                st.error("❌ Please specify a valid destination folder.")
                return

            os.makedirs(dest_folder, exist_ok=True)
            st.info(f"📁 Saving outputs to: {dest_folder}")

            last_df = None

            for file in file_list:
                # Load file
                if isinstance(file, str):
                    df = pd.read_csv(file)
                    fname = os.path.basename(file)
                else:
                    df = pd.read_csv(file)
                    fname = file.name

                # Add WC column if constant WC selected
                if wc_mode == "Enter constant Water Cut (%)":
                    df[wc_col] = wc_value

                # Safety check
                if ql_col not in df.columns or wc_col not in df.columns:
                    st.error(f"❌ Missing required columns in file: {fname}")
                    continue

                # Compute flows
                df["Qw"] = df[ql_col] * df[wc_col] / 100
                df["Qo"] = df[ql_col] * (1 - df[wc_col] / 100)

                df.columns = ["Qg" if c.strip().lower() == "alicat" else c for c in df.columns]

                # Remove originals if requested
                if remove_original:
                    cols_to_drop = [ql_col]
                    if wc_mode == "Select existing Water Cut column":
                        cols_to_drop.append(wc_col)
                    df = df.drop(columns=cols_to_drop, errors="ignore")

                # Save output
                out_path = os.path.join(dest_folder, f"processed_{fname}")
                df.to_csv(out_path, index=False)

                last_df = df

            st.success(f"✅ Processed {len(file_list)} files successfully.")

            if last_df is not None:
                st.write("### ✅ Preview of Last Processed File")
                st.dataframe(last_df.head())


 
