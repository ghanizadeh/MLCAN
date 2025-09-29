import os
import pandas as pd
import re
import streamlit as st
from data_preprocessing.calibration import InputProcessor

def process_csv_folder(source_folder, dest_folder=None, indicator=1):
    # Initialize the input processor
    input_processor = InputProcessor()
    
    if not os.path.exists(source_folder):
        st.info(f"❌ Source folder not found: {source_folder}")
        return

    # Prepare output folder
    if not dest_folder:
        parent_dir = os.path.dirname(os.path.abspath(source_folder))
        root_folder_name = os.path.basename(os.path.normpath(source_folder))
        dest_folder = os.path.join(parent_dir, f"{root_folder_name}_processed")
    
    os.makedirs(dest_folder, exist_ok=True)

    # Get list of all CSV files
    csv_files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]
    st.info(f"📂 Found **{len(csv_files)}** CSV files to process in **{source_folder}**")
    
    for file_name in csv_files:
        file_path = os.path.join(source_folder, file_name)
        st.success(f"Processing **{file_name}**...")

        try:
            # Extract AliCat and VFD values from filename
            alicat_match = re.search(r'AliCat(\d+\.\d+)', file_name)
            vfd_match = re.search(r'VFD(\d+\.\d+)', file_name)

            if alicat_match and vfd_match:
                alicat_val = float(alicat_match.group(1))
                vfd_val    = float(vfd_match.group(1))
            else:
                st.success("  ✗ No AliCat/VFD in filename — checking inside file…")
                df = pd.read_csv(file_path)
                al_cols = [c for c in df.columns if re.search(r'AliCat_', c, re.IGNORECASE)]
                vfd_cols = [c for c in df.columns if re.search(r'VFD', c, re.IGNORECASE)]
                if not al_cols or not vfd_cols:
                    st.success("  ✗ Skipping — no AliCat or VFD in filename or columns")
                    continue
                alicat_val = float(df[al_cols[0]].iloc[0])
                vfd_val    = float(df[vfd_cols[0]].iloc[0])
                st.success(f"  ✓ Found in columns: AliCat={alicat_val}, VFD={vfd_val}")

            # Read CSV
            df = pd.read_csv(file_path)
            original_rows = len(df)

            # Rename to Board channels
            df.rename(columns={
                'Sensor5': 'Board1_I0',
                'Sensor3': 'Board1_I1',
                'Sensor1': 'Board1_I2',
                'CRL':     'Board1_I3',
                'Sensor6': 'Board3_I0',
                'Sensor4': 'Board3_I1',
                'Sensor2': 'Board3_I2',
                'AliCat': 'Board3_I3'
            }, inplace=True)

            # 1. Keep only rows where indicator matches user choice
            #df = df[df['indicator'] == indicator]
            
            if str(indicator).lower() != "both":
                #df = df[df['indicator'].isin([0, 1])]
                df = df[df['indicator'] == int(indicator)]

            

            # 2. Filter by AliCat and VFD values

            tol = 0.01
            df = df[
                (abs(df['AliCat_Output'] - alicat_val) < tol) & 
                (abs(df['VFD_Output'] - vfd_val) < tol)
            ]

            # 3. Apply calibration
            for column in ['Board1_I0', 'Board1_I1', 'Board1_I2', 'Board1_I3', 
                           'Board3_I0', 'Board3_I1', 'Board3_I2', 'Board3_I3']:
                board = int(column.split('_')[0].replace('Board', ''))
                channel = column.split('_')[1]
                
                df[column] = df[column].apply(lambda x: input_processor.scale_input(board, channel, x)[0])
                

            # Rename back
            df.rename(columns={
                'Board1_I0': 'Sensor5',
                'Board1_I1': 'Sensor3',
                'Board1_I2': 'Sensor1',
                'Board1_I3': 'CRL',
                'Board3_I0': 'Sensor6',
                'Board3_I1': 'Sensor4',
                'Board3_I2': 'Sensor2',
                'Board3_I3': 'AliCat'
            }, inplace=True)

            # Save
            base, ext = os.path.splitext(file_name)
            out_name = f"{base}_calibrated{ext}"     # add suffix before extension
            out_path = os.path.join(dest_folder, out_name)

            df.to_csv(out_path, index=False)
            st.success(
                f"  ✓ Complete: saved as {out_name} | {len(df)} rows kept out of {original_rows} with indicator={indicator}"
            )
        except Exception as e:
            st.success(f"  ❌ Error: {str(e)}")


