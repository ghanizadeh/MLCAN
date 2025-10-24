import streamlit as st
st.set_page_config(page_title="MLCAN", layout="wide")
from streamlit_option_menu import option_menu
import numpy as np
import os
import pandas as pd
from pathlib import Path

# Import pages
from utility.signal_segmentor import show_signal_segmentor
from home.welcome import show_welcome_page
from data_preprocessing.csv_cleaner_streamlit import process_csv_folder
from utility.Olga_convertor import show_olga_convertor_page
from utility.merge_datasets import show_merge_page
from data_preprocessing.create_ml_dataset import show_create_ML_dataset
from model.train import show_ML_model_page
from model.tcn import show_TCN_model_page
from model.model_window import show_RF_window_page
 


# ----------------------------------
# Back to Top button
# ----------------------------------
st.markdown(
    """
    <style>
      html { scroll-behavior: smooth; }
      .back-to-top {
          position: fixed;
          bottom: 24px;
          right: 24px;
          z-index: 9999;
          background: #ccffd2;
          color: #0000;
          padding: 10px 14px;
          border-radius: 10px;
          text-decoration: none;
          font-weight: 600;
          box-shadow: 0 4px 10px rgba(0,0,0,0.15);
      }
      .back-to-top:hover { opacity: .9; }
    </style>
    <a class="back-to-top" href="#top">Back to Top</a>
    """,
    unsafe_allow_html=True
)

# Custom CSS with classes
st.markdown("""
    <style>
    .red-button > button {
        background-color: crimson;
        color: white;
        font-weight: bold;
    }
    .green-button > button {
        background-color: seagreen;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Set page to wide mode but limit main content width
st.markdown("""
<style>
/* Control main container width */
.block-container {
    max-width: 55%;      /* 👈 adjust between 75%–95% as you like */
    padding-left: 3rem;
    padding-right: 3rem;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)
# -----------------------------
# Sidebar with tree-like menu
# -----------------------------
with st.sidebar:
    # 🔹 Logo img
    st.image("img/logo.jpg", use_container_width=True)   
    
    selected = option_menu(
        menu_title="Menu",  
        options=["Home", "① Data Preprocessing", "② Models", "Extra Tools"],  # main menu
        #icons=["house", "table", "gear", "diagram-3"],  # optional icons
        icons=["house", "1", "2", "toolbox"],  # optional icons
        menu_icon="cast",  # main icon
        default_index=0,
    )

    # Submenus depending on main selection
    submenu = None
    if selected == "① Data Preprocessing":
        submenu = option_menu(
            menu_title="Data Preprocessing", 
            options=["① Calibration (mA to psi)", "② Create ML Dataset"],
            icons=["1", "2"],
            menu_icon="table",
            default_index=0,    
        )
    if selected == "Extra Tools":
        submenu = option_menu(
            menu_title="Extra Tools",
            options=["① Signal Segmentor", "② PvT-Sim Convertor", "③ Data Integration"],
            icons=["1", "2", "3"],
            menu_icon="gear",
            default_index=0,
        )
    elif selected == "② Models":
        submenu = option_menu(
            menu_title="Model",
            options=["① Ensemble Learning", "② Adaptive Ensemble Learning", "③ TCN"],
            #icons=["play", "sliders", "check2-circle"],
            icons=["1","2","3"],
            menu_icon="diagram-3",
            default_index=0,
        )
 
# ----------------------------------------
# Main Content Page
# ----------------------------------------
#st.image("img/logo.jpg", use_container_width=True)  
st.markdown('<a name="top"></a>', unsafe_allow_html=True) #Anchor for Top of the page
if selected == "Home":
    show_welcome_page()
# ----------------------------------------
# Data Preprocessing 
# ----------------------------------------
elif selected == "① Data Preprocessing":
    st.title(f"{submenu}")
    st.subheader(f"📊 ① Data Preprocessing → {submenu}")
    # ----------------------------------------
    #  Calibration page 
    # ----------------------------------------
    if submenu == "① Calibration (mA to psi)":
        with st.container(border=True):
            st.markdown("##### 📂 File Input Options")
            option = st.radio("Choose input method:",("Enter source & destination folders", "Upload files"))
            df_list, filenames = [], []
            if option == "Enter source & destination folders":
                source = st.text_input("**📥 Source directory** (local or Google Drive path):")
                dest = st.text_input("**📤 Destination directory** (leave **empty** for **default**):")
                if not dest.strip() and source.strip():
                    source_path = Path(source)
                    dest_path = str(source_path.parent / f"{source_path.name}_processed")
                if source and os.path.isdir(source):
                    files = [f for f in os.listdir(source) if f.endswith(".csv")]
            elif option == "Upload files":
                source = st.file_uploader("**📥 Upload one or more CSV files:**", type=["csv"], accept_multiple_files=True)
                dest = st.text_input("**📤 Destination directory (leave empty for default):**")
                dest_path = dest if dest.strip() else None   # no default for uploads

            # Indicator choice (0, 1, or both)
            indicator_choice = st.radio("**Select indicator:**", options=[0, 1, "both"],index=1)
        if st.button("⏵ Run Calibration"):
            if not source:
                st.error("⚠️ Please enter a source folder.")
            else:
                with st.expander("**💻 Output logs:**"):
                    with st.spinner("Processing files..."):
                        if indicator_choice == "both":
                            process_csv_folder(source, dest if dest else None, indicator="both")
                        elif indicator_choice == 0:
                            process_csv_folder(source, dest if dest else None, indicator=0)
                        elif indicator_choice == 1:
                            process_csv_folder(source, dest if dest else None, indicator=1)
                st.success(f"\n🎯 {len(files)} files were calibrated")
                st.success(f"✅ Calibration completed with **indicator = {indicator_choice}**")
                st.success(f"Destination path: **{dest_path}**")


 
    # ----------------------------------------
    #  Create ML Dataset page 
    # ----------------------------------------  
    elif submenu == "② Create ML Dataset":    
        show_create_ML_dataset()
    
    # Back to Top link
    st.markdown("[Back to Top](#top)")

elif selected == "Extra Tools":
    #st.title(f"⚙️ Extra Tools → {submenu}")
    if submenu == "① Signal Segmentor":
       show_signal_segmentor()
    elif submenu == "② PvT-Sim Convertor":
       show_olga_convertor_page()
    elif submenu == "③ Data Integration":
       show_merge_page()
    

# ========================================================================================
#  ML models
# ========================================================================================
elif selected == "② Models":
    if submenu == "① Ensemble Learning":
        show_ML_model_page()
    elif submenu == "② Adaptive Ensemble Learning":
    #    show_RF_window_page()
        st.title("In progress ...")
    elif submenu == "③ TCN":
        show_TCN_model_page()

        


elif selected == "Results":
    st.header(f"Results → {submenu}")
    if submenu == "Metrics":
        st.write("📊 Show evaluation metrics here...")
    elif submenu == "Plots":
        st.write("📈 Show plots here...")
    elif submenu == "SHAP":
        st.write("🔥 Show SHAP values here...")
