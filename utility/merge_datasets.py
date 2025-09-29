import os
import pandas as pd
import streamlit as st

# ================================
# Utility: Merge CSVs
# ================================
def merge_csv_files(file_list, percent=100):
    """Merge multiple CSVs sequentially with a 'source_file' column."""
    merged_df = pd.DataFrame()
    summary = []  # to log how many rows from each file

    for file in file_list:
        try:
            df = pd.read_csv(file)
            rows_total = len(df)
            rows_to_pick = int(rows_total * (percent / 100))
            df = df.head(rows_to_pick)

            # Add file source column
            df["source_file"] = os.path.basename(file.name if hasattr(file, "name") else file)

            # Merge
            merged_df = pd.concat([merged_df, df], ignore_index=True)
            summary.append(f"📄 {os.path.basename(file.name if hasattr(file, 'name') else file)} → {rows_to_pick}/{rows_total} rows")

        except Exception as e:
            st.error(f"❌ Error reading {file}: {e}")
    
    return merged_df, summary


# ================================
# Streamlit Page
# ================================
def show_merge_page():
    #st.header("③ Merge Datasets 📊")

    with st.expander("ℹ️ **About** "):
        st.markdown("""
        Use this page to **merge multiple CSV datasets sequentially**.  
        - Upload CSVs or enter a folder path.  
        - Each row gets a new column **`source_file`** indicating its origin.  
        - Use the slider to include only a **percentage** of rows from each dataset.  
        - After merging, you’ll see a summary + the size of the final data.  
        """)

    # ---------------------------
    # Input method
    # ---------------------------
    option = st.radio("Choose input method:", ["📂 Upload CSV files", "🖊️ Enter folder path"])

    percent = st.slider("Select percentage of each dataset to include", 10, 100, 100, step=10)

    merged_df = pd.DataFrame()
    summary = []

    if option == "📂 Upload CSV files":
        uploaded_files = st.file_uploader("Upload multiple CSV files", type=["csv"], accept_multiple_files=True)
        if uploaded_files and st.button("🔄 Merge Files"):
            merged_df, summary = merge_csv_files(uploaded_files, percent=percent)

    elif option == "🖊️ Enter folder path":
        folder_path = st.text_input("Enter the folder path containing CSV files:")
        if folder_path and st.button("🔄 Merge Files"):
            if os.path.exists(folder_path):
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
                if files:
                    merged_df, summary = merge_csv_files(files, percent=percent)
                else:
                    st.warning("⚠️ No CSV files found in the folder.")
            else:
                st.error("❌ Invalid folder path.")

    # ---------------------------
    # Show results
    # ---------------------------
    if not merged_df.empty:
        st.subheader("✅ Merge Summary")
        for s in summary:
            st.write(s)

        st.success(f"Final merged dataset size: **{len(merged_df)} rows, {len(merged_df.columns)} columns**")
        st.dataframe(merged_df.head(20))

        # Download
        csv = merged_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Merged CSV", csv, "merged.csv", "text/csv")
