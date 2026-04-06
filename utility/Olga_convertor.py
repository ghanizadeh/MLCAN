import streamlit as st
import pandas as pd
 
import re
import math

def render(df):

    st.title("📊 Exploratory Data Analysis")
    st.divider()
    if df is None:
        st.warning("⚠️ No dataset loaded. Go to **📂 Data** first.")
        st.stop()

    # ── Column selector ───────────────────────────────────────────────────
    st.subheader("🎯 Select Columns to Explore")
    with st.expander("Column selector", expanded=True):
        features, target = render_column_selector(
            df,
            default_features=get_value("data.feature_names"),
            default_target=get_value("data.target_name"),
            key_prefix="eda",
        )

    # EDA is read-only — do NOT persist selections to shared state.
    # Column selections here are local to EDA only.

    if not features or target is None:
        st.info("Select at least one feature and a target column above.")
        st.stop()

    selected_cols = features + [target]
    df_sub = df[selected_cols].copy()

    # ── EDA tabs ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔎 EDA Dashboard")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8  = st.tabs(
        ["🔎 Preview", "Data Filtering", "📝 Summary", "📊 Scatter + Histograms", "📊 3D Scatter plots", "📦 Box plots", "🔗 Correlation", "⚠️ Categorical" ]
    )

    # ── Tab 1: Preview ────────────────────────────────────────────────────
    with tab1:
        st.write(f"**Shape:** {df_sub.shape[0]:,} rows × {df_sub.shape[1]} cols")
        st.dataframe(df_sub, use_container_width=True)

    # ── Tab 2: Data Filtering ────────────────────────────────────────────────────
    with tab2:
        st.subheader("🔍 Optional Filtering Before Cleaning")
        # Normalize column names for case-insensitive matching
        df_cols_lower = {col.lower(): col for col in df_sub.columns}
        # 🔥 Smart column finder
        def find_column(df_cols_lower, keywords):
            for col_lower, original_col in df_cols_lower.items():
                if all(k in col_lower for k in keywords):
                    return original_col
            return None
        # 🔥 Define filters with fallback logic
        filter_cols = {
            "Concentrate Ratio": find_column(df_cols_lower, ["concentrate", "ratio"]),
            "Dilution Ratio": find_column(df_cols_lower, ["dilution", "ratio"]),
            "Brine Type": find_column(df_cols_lower, ["brine", "type"]),
            "Brine Name": find_column(df_cols_lower, ["brine", "name"]),
        }

        for label, actual_col in filter_cols.items():

            if actual_col is None:
                #st.warning(f"⚠️ No column found for **{label}**")
                continue

            # 🔥 Inform user if fallback happened
            if label.lower() not in actual_col.lower():
                st.info(f"ℹ️ Using detected column for {label}: **{actual_col}**")

            unique_vals = df_sub[actual_col].dropna().unique()

            if len(unique_vals) > 1:
                st.markdown(f"**Filter by {actual_col}:**")

                selected_vals = st.multiselect(
                    f"Choose {actual_col} values to keep",
                    options=unique_vals,
                    default=list(unique_vals),
                    key=f"eda_filter_{actual_col}"   # eda-namespaced key — never collides with preprocessing
                )

                df_sub = df_sub[df_sub[actual_col].isin(selected_vals)]

                st.info(
                    f"✅ Filtered {actual_col}: {len(selected_vals)} values selected → {len(df_sub)} rows remain."
                )

            else:
                st.warning(f"⚠️ {actual_col} has only one unique value — no filtering applied.")

        # Update after filtering
        st.success(f"✅ Data filtered successfully. Original Shape: {df.shape} - Current shape: {df_sub.shape}")


    # ── Tab 3: Statistical summary ────────────────────────────────────────
    with tab3:
        st.markdown("#### 🔢 Numeric Summary")
        st.dataframe(extended_describe(df_sub), use_container_width=True)

        cat_df = categorical_summary(df_sub)
        if not cat_df.empty:
            st.markdown("#### 🏷️ Categorical Summary")
            st.dataframe(cat_df, use_container_width=True)

    # ── Tab 4: Scatter + Histograms ─────────────────────────────────────────────
    with tab4:
        num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df_sub[c])]
        if not num_cols:
            st.warning("No numeric feature columns available for distribution plots.")
        else:

            st.markdown("#### 🔄 Scatter + Marginal Histograms")
            num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df_sub[c])]
            if not num_cols:
                st.warning("No numeric feature columns available.")
            else:
                y = df_sub[target]  # use filtered df_sub to keep index aligned
                unique_classes = y.dropna().unique()
                n_classes = len(unique_classes)
                # =========================
                # CASE 1: Regression
                # =========================
                if pd.api.types.is_numeric_dtype(y) and n_classes > 5:
                    st.success("Detected: Regression")
                    show_trendline = st.checkbox("Show R² trendline", value=True, key="scatter_trendline")
                    figs = draw_pairwise_scatter_with_hist(
                        df_sub[num_cols + [target]],
                        target,
                        show_trendline
                    )
                # =========================
                # CASE 2: Classification (2–3 classes)
                # =========================
                elif n_classes <= 3:
                    
                    # 🔥 Convert target for plotting
                    y_encoded, mapping = pd.factorize(y)
                    df_plot = df_sub.copy()
                    df_plot[target] = y_encoded
                    st.info(f"Detected: Classification ({n_classes} classes) --- Class mapping: {dict(enumerate(mapping))}")
                    show_trendline = False
                    figs = draw_pairwise_scatter_with_hist(
                        df_plot[num_cols + [target]],
                        target,
                        show_trendline
                    )
                # =========================
                # CASE 3: Too many classes
                # =========================
                else:
                    st.warning(
                        f"⚠️ Target has {n_classes} classes. "
                        "Scatter plots are disabled for >3 classes."
                    )
                    figs = []
                if not figs:
                    st.info("Not enough data to render plots.")
                else:
                    for row_start in range(0, len(figs), 2):
                        cols = st.columns(2)
                        for col_idx, fig in enumerate(figs[row_start : row_start + 2]):
                            with cols[col_idx]:
                                fig_to_st(fig)

            st.divider()
            st.markdown("#### 📊 Histograms")
            bins = st.slider("Histogram bins", 5, 100, 30, key="eda_bins")
            fig = draw_histograms(df_sub, num_cols, bins=bins)
            fig_to_st(fig)
            st.divider()

            st.markdown("#### 🟢 Scatter Plot - All Features vs Target")
            # Checkbox option
            plot_all_one = st.checkbox("Show all features in one figure",value=True,key="eda_scatter_all_one")
            # --------------------------------------------------
            # OPTION 1 — Separate figures (your current style)
            # --------------------------------------------------
            if not plot_all_one:
                for feature in num_cols:
                    fig = draw_scatter(
                        df_sub,
                        x_col=feature,
                        y_col=target,
                        hue_col=None
                    )
                    fig_to_st(fig)
            # -------------------------------------------------
            # OPTION 2 — One image with subplots
            # --------------------------------------------------
            else:
                n = len(num_cols)
                cols = 3
                rows = math.ceil(n / cols)
                fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
                axes = axes.flatten()
                for i, feature in enumerate(num_cols):
                    ax = axes[i]
                    # Drop NaNs and coerce to numeric to avoid matplotlib
                    # category-axis errors when target contains mixed types
                    plot_df = df_sub[[feature, target]].dropna()
                    x_vals = pd.to_numeric(plot_df[feature], errors="coerce")
                    y_vals = pd.to_numeric(plot_df[target],  errors="coerce")
                    valid  = x_vals.notna() & y_vals.notna()
                    ax.scatter(x_vals[valid], y_vals[valid], alpha=0.7)
                    ax.set_xlabel(feature)
                    ax.set_ylabel(target)
                    ax.set_title(f"{feature} vs {target}")
                # Remove empty axes
                for j in range(i+1, len(axes)):
                    fig.delaxes(axes[j])
                plt.tight_layout()
                fig_to_st(fig)


    # ── Tab 5: 3D Scatter ───────────────────────────────────────────────
    with tab5:
        st.markdown("#### 📊 3D Scatter Plotsddddd")
        if len(num_cols) >= 2:
            c1, c2, c3 = st.columns(3)
            # X axis
            x_col = c1.selectbox("X axis",num_cols,key="eda_3d_x")
            # Y axis
            y_col = c2.selectbox("Y axis",num_cols,index=1,key="eda_3d_y")
            # Color column (default = target)
            color_options = ["None"] + df_sub.columns.tolist()
            default_index = color_options.index(target) if target in color_options else 0
            hue_col = c3.selectbox("Color by",color_options,index=default_index,key="eda_3d_color")
            if hue_col == "None":
                hue_col = None
            # 🔥 FIX: ensure column names are strings
            fig = draw_scatter(df_sub, x_col, y_col, hue_col=hue_col)
            fig_to_st(fig)
 
    # ── Tab 6: Boxplots ───────────────────────────────────────────────
    with tab6:
        st.markdown("#### 📦 Boxplots")
        use_standardize = st.checkbox("Standardize features for visualization (recommended when scales differ)",value=False)
        df_plot = df_sub.copy()
        if use_standardize:
            scaler = StandardScaler()
            df_plot[num_cols] = scaler.fit_transform(df_sub[num_cols])
        fig = draw_boxplots(df_plot, num_cols)
        fig_to_st(fig)
        st.divider()

    # ── Tab 7: Correlation ────────────────────────────────────────────────
    with tab7:
        st.markdown("### 🟣 Correlation Matrix")
        num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df_sub[c])]
        if not num_cols:
            st.warning("No numeric features available for correlation analysis.")
        elif len(num_cols) > MAX_HEATMAP_FEATURES:
            st.warning(
                f"⚠️ Too many numeric features ({len(num_cols)}) — "
                f"heatmap limited to {MAX_HEATMAP_FEATURES}. "
                "Showing raw correlation table only."
            )
            st.dataframe(df_sub[num_cols].corr(), use_container_width=True)
        else:
            st.dataframe(df_sub[num_cols + [target]].corr(), use_container_width=True)
            fig, corr, rec, high_corr = draw_correlation_heatmap(
                df_sub,
                columns=num_cols + [target],
                target=target
            )
            fig_to_st(fig)
            st.markdown("### ⭐ Recommended Features")
            if rec is not None:
                st.dataframe(rec, use_container_width=True)
            st.markdown("### ⚠️ Highly Correlated Features")
            if not high_corr.empty:
                st.dataframe(high_corr, use_container_width=True)
            else:
                st.success("No severe multicollinearity detected.")

    # ── Tab 8: Categorical diagnostics ───────────────────────────────────
    with tab8:
        warn_df = categorical_warnings(df_sub)
        if warn_df.empty:
            st.success("✅ No major categorical issues detected.")
        else:
            st.dataframe(warn_df, use_container_width=True)

        with st.expander("📊 Class Imbalance Details"):
            imb_df = categorical_imbalance(df_sub)
            if imb_df.empty:
                st.info("No categorical columns found.")
            else:
                st.dataframe(imb_df, use_container_width=True)


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

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_corr_heatmap(
    df,
    method="pearson",
    figsize=(14, 12),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    triangle=False,
    font_scale=0.7   # 👈 control everything from here
):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Convert to numeric (safe)
    df_num = df.apply(pd.to_numeric, errors="coerce").select_dtypes(include="number")

    if df_num.shape[1] < 2:
        st.warning("⚠️ Not enough numeric columns.")
        return

    corr = df_num.corr(method=method)

    mask = np.triu(np.ones_like(corr, dtype=bool)) if triangle else None

    # 👇 control global scaling
    sns.set_context("notebook", font_scale=font_scale)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        center=center,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7},
        annot_kws={"size": 8},   # 👈 numbers inside cells
        ax=ax
    )

    ax.set_title(f"{method.capitalize()} Correlation", fontsize=12)
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()

    st.pyplot(fig)

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
        plot_corr_heatmap(df)
        render(df)


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



