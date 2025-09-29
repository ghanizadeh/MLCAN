import streamlit as st
import pandas as pd
 
import re
import math

# ==========================
# 1. Read .tab file into DataFrame
# ==========================
def read_tab_file(uploaded_file):
    content = uploaded_file.read().decode("utf-8")

    # --- Find the LAST COLUMNS= (...) block ---
    col_matches = re.findall(r"COLUMNS\s*=\s*\((.*?)\)", content, re.DOTALL)
    if not col_matches:
        raise ValueError("No COLUMNS header found in file")
    cols_text = col_matches[-1]  # ✅ use the last one
    cols = re.split(r"[\s,]+", cols_text.strip())  # split on space/comma

    # --- Extract PVTTABLE POINT rows (after last COLUMNS) ---
    # Only keep the content after the last "COLUMNS"
    content_after_cols = content[content.rfind("COLUMNS"):]
    data = re.findall(r"PVTTABLE POINT\s*=\s*\((.*?)\)", content_after_cols, re.DOTALL)

    rows = []
    for row in data:
        # split on commas, strip spaces/tabs
        values = [x.strip() for x in row.replace("\n", " ").split(",") if x.strip()]
        values = [float(x) for x in values]
        rows.append(values)

    # sanity check
    for i, r in enumerate(rows):
        if len(r) != len(cols):
            raise ValueError(f"Row {i} has {len(r)} values but expected {len(cols)}")

    df = pd.DataFrame(rows, columns=cols)
    return df, cols, content

# ==========================
# 2. Save DataFrame back to .tab format
# ==========================
def save_tab_file(df, cols, original_content):
    new_content = original_content
    pattern = r"(PVTTABLE POINT\s*=\s*\()(.*?)(\))"
    matches = list(re.finditer(pattern, original_content, re.DOTALL))
    if len(matches) != len(df):
        raise ValueError(f"Row mismatch: file has {len(matches)} PVTTABLE POINTs, df has {len(df)} rows")

    for i, match in enumerate(matches):
        row = df.iloc[i].values
        formatted = ",".join(fmt_olga(float(val)) for val in row)
        new_block = f"{match.group(1)}{formatted}{match.group(3)}"
        new_content = new_content.replace(match.group(0), new_block, 1)

    return new_content





def fmt_olga(x: float) -> str:
    """
    Format like OLGA/PVT: .ddddddE±DD (six decimals, no leading 0).
    Examples:
      0        -> .000000E+00
      0.5      -> .500000E+00
      5        -> .500000E+01
     -12.3456  -> -.123456E+02
    """
    if x == 0 or x == 0.0:
        return ".000000E+00"
    sign = "-" if x < 0 else ""
    ax = abs(x)
    # exponent so that mantissa is in [0.1, 1.0)
    e = int(math.floor(math.log10(ax))) + 1
    m = ax / (10 ** e)
    # round to 6 decimals; if it rounds to 1.000000, bump exponent
    m = round(m, 6)
    if m >= 1.0:
        m = 0.1
        e += 1
    mant = f"{m:.6f}"[1:]     # drop the leading '0' -> '.dddddd'
    exp  = f"{e:+03d}"        # sign plus two digits (e.g., +01, -01, +10)
    return f"{sign}{mant}E{exp}"

import re

import re

def _replace_array(content: str, key: str, unit: str, updater):
    pattern = rf"({re.escape(key)}\s*=\s*\()(.*?)(\)\s*{re.escape(unit)}\s*,\\)"
    def repl(m):
        vals = [float(v.strip()) for v in m.group(2).split(",")]
        new_vals = updater(vals)
        return f"{m.group(1)}{','.join(fmt_olga(v) for v in new_vals)}{m.group(3)}"
    return re.sub(pattern, repl, content, flags=re.DOTALL)

def _set_scalar_with_unit(content: str, key: str, unit: str, new_value: float):
    pattern = rf"({re.escape(key)}\s*=\s*)([^\s,]+)(\s*{re.escape(unit)}\s*,\\)"
    return re.sub(
        pattern,
        lambda m: f"{m.group(1)}{fmt_olga(float(new_value))}{m.group(3)}",
        content,
        flags=re.DOTALL
    )

def _get_scalar_with_unit(content: str, key: str, unit: str) -> float | None:
    m = re.search(
        rf"{re.escape(key)}\s*=\s*([^\s,]+)\s*{re.escape(unit)}\s*,\\",
        content, flags=re.DOTALL
    )
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

def update_header(content: str, *, new_val: float, kind: str):
    """
    Update OLGA header depending on kind:
      - ROG  -> Gas density
      - ROWT -> Water density
      - ROHL -> Oil density
    """

    # --- Extract components for labeling ---
    comp_names = []
    m = re.search(r'COMPONENTS\s*=\s*\((.*?)\)', content, flags=re.DOTALL)
    if m:
        comp_names = [p.strip().strip('"').strip("'") for p in m.group(1).split(",")]

    comp_name = None
    comp_index = None
    scalar_key = None
    density_key = None

    if kind.upper() == "ROG":
        scalar_key = "STDGASDENSITY"
        comp_index = 1  # second component
        density_key = "DENSITY"
    elif kind.upper() == "ROWT":
        scalar_key = "STDWATDENSITY"
        comp_index = 0  # first component (water)
        density_key = "DENSITY"
    elif kind.upper() == "ROHL":
        scalar_key = "STDOILDENSITY"
        comp_index = 2  # third component (oil)
        density_key = "DENSITY"
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    if comp_names and 0 <= comp_index < len(comp_names):
        comp_name = comp_names[comp_index]

    # --- OLD value from header ---
    old_val = _get_scalar_with_unit(content, scalar_key, "kg/m3")
    scale = 1.0
    if old_val is not None and new_val != 0.0:
        scale = float(new_val) / float(old_val)

    # --- Update DENSITY[...] (g/cm3) ---
    new_density_g_cm3 = float(new_val) * 1e-3
    def update_density(arr):
        arr[comp_index] = new_density_g_cm3
        return arr
    content = _replace_array(content, density_key, "g/cm3", update_density)

    # --- Update MOLWEIGHT[...] using (old/new) scale ---
    def update_molwt(arr):
        arr[comp_index] = arr[comp_index] * scale
        return arr
    content = _replace_array(content, "MOLWEIGHT", "g/mol", update_molwt)

    # --- Update header scalar ---
    content = _set_scalar_with_unit(content, scalar_key, "kg/m3", float(new_val))

    return content, comp_name, (old_val, new_val, scale)


# ==========================
# 3. Streamlit App
# ==========================
def show_olga_convertor_page():
    #st.subheader("🛠️ OLGA PVT Table Editor")

    with st.expander("ℹ️ **About** "):
        st.markdown("""
        Use this page to **modify the PvT Sim .tab files**.  
        - 📂 Upload `.tab` PVT files.  
        - 🔍 Preview and explore original data in an interactive table.  
        - ✏️ Edit selected columns or update fluid properties (ROG, ROWT, ROHL).  
        - ⚖️ Automatically adjusts related values like **density** and **molecular weight**.  
        - 💾 Export updated files in OLGA-compatible `.tab` format.
        """)


    uploaded_file = st.file_uploader("Upload .tab File", type=["tab"])

    if uploaded_file:
        df, columns, content = read_tab_file(uploaded_file)

        st.write("### Original Data Preview")
        st.dataframe(df)

        # Column selection
        selected_cols = st.multiselect("Select columns to edit:", columns)

        new_values = {}
        for col in selected_cols:
            val = st.text_input(f"Enter new value for {col}:", value="")
            if val.strip():
                try:
                    new_values[col] = float(val)
                except:
                    st.error(f"Invalid number for {col}")

        if st.button("Apply Changes"):
            # Apply DataFrame edits
            for col, val in new_values.items():
                df[col] = val

            # Header updates
            for key in ["ROG", "ROWT", "ROHL"]:
                if key in new_values:
                    content, comp, ratio_info = update_header(content, new_val=new_values[key], kind=key)
                    old, new, scale = ratio_info
                    st.info(
                        f"{key} update for {comp or key}:  \n"
                        f"Old Density: {old:.6g}  \n"
                        f"New Density: {new:.6g}  \n"
                        f"MOLWEIGHT scaled by {scale:.6g}"
                    )

            st.success("Values updated!")
            st.write("### Updated Data")
            st.dataframe(df)

            tab_text = save_tab_file(df, columns, content)
            st.download_button(
                "Download Updated .tab",
                data=tab_text,
                file_name="updated_pvt.tab",
                mime="text/plain"
            )



