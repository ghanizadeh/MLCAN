def show_ML_model_page():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shap
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from catboost import CatBoostRegressor
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split, KFold, GroupKFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.inspection import PartialDependenceDisplay
    import os
    import joblib
    import os
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV


    # ================================
    # Utility Functions
    # ================================
    def plot_predicted_vs_measured_separately(y_true, y_pred, dataset_type, model_name, target):
        color = 'teal' if "Train" in dataset_type else 'orange'

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        plt.figure(figsize=(5, 5))
        max_val = np.max(np.concatenate([y_true, y_pred]))
        min_val = 0

        plt.scatter(y_true, y_pred, c=color, edgecolors='black',
                    marker='o' if "Train" in dataset_type else 'v', alpha=0.7)
 
        #

        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        plt.plot([min_val, max_val], [min_val, 1.2 * max_val], 'r--', linewidth=1, label='20% Error')
        plt.plot([min_val, max_val], [min_val, 0.8 * max_val], 'r--', linewidth=1)

        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.xlabel("Measured " + target)
        plt.ylabel("Predicted " + target)
        plt.title(f"{model_name} - {dataset_type} Set")

        #plt.legend(title=f"{dataset_type}:\nR²={r2:.2f} & MAE={mae:.2f}")
        plt.legend(title=f"New Dataset:\nR²={r2:.2f} & MAE={mae:.2f}")

        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    def a20_index(y_true, y_pred):
        ratio = y_pred / y_true
        ratio = np.where(ratio < 1, 1 / ratio, ratio)
        return np.mean(ratio <= 1.2)

    def get_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        a20 = a20_index(y_true, y_pred)
        return [mae, mse, rmse, r2, a20]

    def plot_pairwise_corr_with_hist(df, target_col):
        plots = []
        numeric_cols = df.select_dtypes(include='number').columns.drop(target_col)
        colors = plt.cm.get_cmap('Set1', len(numeric_cols)).colors

        for i, col in enumerate(numeric_cols):
            data = df[[col, target_col]].dropna()
            x = data[col]
            y = data[target_col]
            if len(data) < 2:
                continue
            try:
                corr = np.corrcoef(x, y)[0, 1]
                r2 = corr ** 2
                slope, intercept = np.polyfit(x, y, 1)
                line_x = np.linspace(x.min(), x.max(), 100)
                line_y = slope * line_x + intercept
            except np.linalg.LinAlgError:
                continue

            fig = plt.figure(figsize=(7, 7))
            grid = plt.GridSpec(4, 4, hspace=0.05, wspace=0.05)
            ax_main = fig.add_subplot(grid[1:4, 0:3])
            ax_xhist = fig.add_subplot(grid[0, 0:3], sharex=ax_main)
            ax_yhist = fig.add_subplot(grid[1:4, 3], sharey=ax_main)

            ax_main.scatter(x, y, color=colors[i], alpha=0.7,
                            edgecolor='black', linewidth=0.5)
            #ax_main.plot(line_x, line_y, color='red', linestyle='--',
            #            linewidth=2, label=f'R² = {r2:.2f}')
            #ax_main.legend(loc='upper center')
            ax_main.set_xlabel(col, fontsize=8)
            ax_main.set_ylabel(target_col, fontsize=8)

            ax_xhist.hist(x, bins=15, color='green', edgecolor='black')

            ax_yhist.hist(y, bins=15, orientation='horizontal',
                        color='green', edgecolor='black')
            # X histogram
            counts_x, bins_x, patches_x = ax_xhist.hist(
                x, bins=15, color='green', edgecolor='black'
            )

            for count, patch in zip(counts_x, patches_x):
                if count > 0:
                    ax_xhist.text(
                        patch.get_x() + patch.get_width() / 2,
                        count,
                        int(count),
                        ha='center',
                        va='bottom',
                        fontsize=7
                    )

            # Y histogram
            counts_y, bins_y, patches_y = ax_yhist.hist(
                y, bins=15, orientation='horizontal',
                color='green', edgecolor='black'
            )

            for count, patch in zip(counts_y, patches_y):
                if count > 0:
                    ax_yhist.text(
                        count,
                        patch.get_y() + patch.get_height() / 2,
                        int(count),
                        va='center',
                        ha='left',
                        fontsize=7
                    )

            # Remove unwanted spines (borders)
            ax_xhist.spines["top"].set_visible(False)
            ax_xhist.spines["right"].set_visible(False)
            ax_xhist.spines["left"].set_visible(False)

            ax_yhist.spines["right"].set_visible(False)
            ax_yhist.spines["top"].set_visible(False)
            ax_yhist.spines["bottom"].set_visible(False)    

            plots.append(fig)

        return plots

    # ================================
    # Streamlit GUI App
    # ================================
    st.title("🟪 Ensemble Learning Models")
    st.divider()
    st.header("📁 Import Dataset")

    # -------------------------------
    # Option selector
    # -------------------------------
    input_mode = st.radio(
        "Choose data source:",
        ["Upload file", "Read from path"],
        horizontal=True
    )

    df = None

    # ===============================
    # 📤 OPTION 1: Upload file
    # ===============================
    if input_mode == "Upload file":
        uploaded_file = st.file_uploader(
            "► Upload your file (.csv or .csv.gz)",
            type=["csv", "gz"]
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".gz"):
                    df = pd.read_csv(uploaded_file, compression="gzip")
                else:
                    df = pd.read_csv(uploaded_file)

                st.success(f"Loaded file: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Failed to read file: {e}")

    # ===============================
    # 📂 OPTION 2: Read from path
    # ===============================
    else:
        file_path = st.text_input(
            "► Enter full file path (.csv or .csv.gz)",
            placeholder="e.g. /data/my_file.csv.gz"
        )

        if file_path:
            if not os.path.exists(file_path):
                st.error("Path does not exist.")
            else:
                try:
                    if file_path.endswith(".gz"):
                        df = pd.read_csv(file_path, compression="gzip")
                    else:
                        df = pd.read_csv(file_path)

                    st.success(f"Loaded file from path: {file_path}")
                except Exception as e:
                    st.error(f"Failed to read file: {e}")

    # ===============================
    # ✅ Final check
    # ===============================
    if df is not None:
        st.info(f"Dataset shape: {df.shape}")
        filename_col = next(c for c in df.columns if c.lower() == "filename")
        df_base = df.dropna(subset=[filename_col]).drop_duplicates() # minimal cleaning (NO feature logic)
 
        # ============================================================
        # 0️⃣ Advanced Physics-Aware Splitting
        # ============================================================
        st.divider()
        st.header("⓪ Advanced Physics-Aware Data Splitting")

        enable_adv_split = st.checkbox("Enable advanced splitting", value=False)

        train_idx = test_idx = None
        split_report = {}

        numeric_cols = df_base.select_dtypes(include="number").columns.tolist()
        groups = df_base[filename_col]

        if enable_adv_split:

            numeric_cols = df_base.select_dtypes(include="number").columns.tolist()
            split_type = st.radio(
                "Select advanced split type",
                [
                    "Group-aware sorted split (Pressure / GOR extrapolation)",
                    "Multi-bin regime split (Low / Mid / High) [Group-aware]",
                    "LORO + Pressure Tail (Deployment Stress Test)"
                ]
            )


            # ========================================================
            # 1️⃣ GROUP-AWARE SORTED SPLIT
            # ========================================================
            if "Group-aware" in split_type:

                # ========================================================
                # GROUP-AWARE: Random TEST + Sorted VALIDATION
                # ========================================================

                sort_col = st.selectbox(
                    "Select physics sorting column (for VALIDATION)",
                    numeric_cols
                )

                # --- TEST: random, group-aware ---
                test_percent = st.slider(
                    "Test set size (%) — RANDOM",
                    10, 40, 20, step=5
                ) / 100

                # --- VALIDATION: sorted, group-aware ---
                val_percent = st.slider(
                    "Validation set size (%) — SORTED",
                    10, 40, 20, step=5
                ) / 100

                direction = st.radio(
                    "Which side goes to VALIDATION?",
                    ["Highest values", "Lowest values"],
                    horizontal=True
                )

                if test_percent + val_percent >= 0.6:
                    st.error("❌ Test + Validation must be < 60%")
                    st.stop()

                # --------------------------------------------------
                # STEP 1️⃣ Random TEST (group-aware)
                # --------------------------------------------------
                unique_runs = df_base[filename_col].unique()

                rng = np.random.default_rng(42)
                n_test_runs = max(1, int(len(unique_runs) * test_percent))
                test_runs = rng.choice(unique_runs, size=n_test_runs, replace=False)

                df_remaining = df_base[~df_base[filename_col].isin(test_runs)]

                # --------------------------------------------------
                # STEP 2️⃣ Sorted VALIDATION (group-aware)
                # --------------------------------------------------
                run_stat = (
                    df_remaining
                    .groupby(filename_col)[sort_col]
                    .mean()
                    .sort_values(ascending=(direction == "Lowest values"))
                )

                n_val_runs = max(1, int(len(run_stat) * val_percent))
                val_runs = run_stat.iloc[:n_val_runs].index

                train_runs = run_stat.iloc[n_val_runs:].index

                # --------------------------------------------------
                # STEP 3️⃣ Final indices
                # --------------------------------------------------
                train_idx = df_base[df_base[filename_col].isin(train_runs)].index
                val_idx   = df_base[df_base[filename_col].isin(val_runs)].index
                test_idx  = df_base[df_base[filename_col].isin(test_runs)].index

                # --------------------------------------------------
                # Safety checks
                # --------------------------------------------------
                assert not set(train_idx) & set(val_idx)
                assert not set(train_idx) & set(test_idx)
                assert not set(val_idx) & set(test_idx)

                split_report = {
                    "mode": "Group-aware random TEST + sorted VALIDATION",
                    "column": sort_col,
                    "test_percent": test_percent,
                    "val_percent": val_percent,
                    "validation_direction": direction,
                    "train_runs": len(train_runs),
                    "val_runs": len(val_runs),
                    "test_runs": len(test_runs)
                }

                st.success("✅ Random TEST + Physics-aware VALIDATION split applied")


            # ========================================================
            # 2️⃣ MULTI-BIN REGIME SPLIT
            # ========================================================
            elif "Multi-bin" in split_type:

                regime_col = st.selectbox(
                    "Select regime column (e.g. GLR or GOR)",
                    numeric_cols
                )

                test_percent = st.slider(
                    "Test size (%) — EXTREME regime",
                    10, 40, 20, step=5
                ) / 100

                val_percent = st.slider(
                    "Validation size (%) — NEAR-extreme regime",
                    10, 40, 20, step=5
                ) / 100

                direction = st.radio(
                    "Which side is extreme?",
                    ["High regime", "Low regime"],
                    horizontal=True
                )

                if test_percent + val_percent >= 0.6:
                    st.error("❌ Test + Validation must be < 60%")
                    st.stop()

                run_stat = (
                    df_base
                    .groupby(filename_col)[regime_col]
                    .mean()
                    .sort_values(ascending=(direction == "Low regime"))
                )

                n_test = max(1, int(len(run_stat) * test_percent))
                n_val  = max(1, int(len(run_stat) * val_percent))

                test_runs = run_stat.iloc[:n_test].index
                val_runs  = run_stat.iloc[n_test:n_test + n_val].index
                train_runs = run_stat.iloc[n_test + n_val:].index

                train_idx = df_base[df_base[filename_col].isin(train_runs)].index
                val_idx   = df_base[df_base[filename_col].isin(val_runs)].index
                test_idx  = df_base[df_base[filename_col].isin(test_runs)].index

                split_report = {
                    "mode": "Multi-bin regime split",
                    "column": regime_col,
                    "direction": direction,
                    "train_runs": len(train_runs),
                    "val_runs": len(val_runs),
                    "test_runs": len(test_runs),
                }

                st.success("✅ Multi-bin Train / Validation / Test split applied")



            # ========================================================
            # 3️⃣ LORO + PRESSURE TAIL
            # ========================================================
            elif "LORO" in split_type:

                pressure_col = st.selectbox("Select pressure column", numeric_cols)

                test_run = st.selectbox(
                    "Select run to leave out",
                    df_base[filename_col].unique()
                )

                test_tail = st.slider("Test tail (%)", 10, 40, 20, step=5) / 100
                val_tail  = st.slider("Validation tail (%)", 10, 40, 20, step=5) / 100

                if test_tail + val_tail >= 0.6:
                    st.error("❌ Test + Validation tail too large")
                    st.stop()

                df_run = df_base[df_base[filename_col] == test_run]
                df_other = df_base[df_base[filename_col] != test_run]

                df_sorted = df_run.sort_values(pressure_col)

                n_test = max(1, int(len(df_sorted) * test_tail))
                n_val  = max(1, int(len(df_sorted) * val_tail))

                test_idx = df_sorted.iloc[-n_test:].index
                val_idx  = df_sorted.iloc[-(n_test + n_val):-n_test].index
                train_idx = df_other.index.union(df_sorted.iloc[:-(n_test + n_val)].index)

                split_report = {
                    "mode": "LORO + pressure tail",
                    "test_run": test_run,
                    "pressure_col": pressure_col,
                    "test_tail": test_tail,
                    "val_tail": val_tail
                }

                st.success("✅ LORO Train / Validation / Test split applied")

                
            # ========================================================
            # 3️⃣ VISUAL PREVIEW
            # ========================================================
            with st.expander("📊 Preview split distribution", expanded=True):

                preview_col = split_report["column"]  

                fig, ax = plt.subplots(figsize=(8, 4))
                fig, ax = plt.subplots(figsize=(8, 4))

                ax.hist(
                    df_base.loc[train_idx, preview_col],
                    bins=30, alpha=0.6, label="Train", color="teal"
                )

                if val_idx is not None and len(val_idx) > 0:
                    ax.hist(
                        df_base.loc[val_idx, preview_col],
                        bins=30, alpha=0.6, label="Validation", color="green"
                    )

                ax.hist(
                    df_base.loc[test_idx, preview_col],
                    bins=30, alpha=0.6, label="Test", color="orange"
                )

                ax.set_xlabel(preview_col)
                ax.set_ylabel("Count")
                ax.set_title("Train / Validation / Test Distribution")
                ax.legend()


                st.pyplot(fig)
                plt.close(fig)
                st.write(split_report)
                st.write("Train rows:", len(train_idx))
                st.write("Validation rows:", len(val_idx) if val_idx is not None else 0)
                st.write("Test rows:", len(test_idx))
                st.write("Unique runs (train):", groups.iloc[train_idx].nunique())
                st.write("Unique runs (val):", groups.iloc[val_idx].nunique() if val_idx is not None else 0)
                st.write("Unique runs (test):", groups.iloc[test_idx].nunique())
                st.write("range of Train's selected feature:",
                        df_base.loc[train_idx, preview_col].min(),
                        "→",
                        df_base.loc[train_idx, preview_col].max())
                st.write("range of Validation's selected feature:",
                        df_base.loc[val_idx, preview_col].min() if val_idx is not None else "N/A",
                        "→",
                        df_base.loc[val_idx, preview_col].max() if val_idx is not None else "N/A")
                st.write("range of Test's selected feature:",
                        df_base.loc[test_idx, preview_col].min(),
                        "→",
                        df_base.loc[test_idx, preview_col].max())

                # ========================================================
                # ⬇️ DOWNLOAD TRAIN / TEST SPLITS
                # ========================================================
                st.divider()
                st.subheader("⬇️ Download Split Datasets")

                if train_idx is not None and test_idx is not None:

                    train_df_export = df_base.loc[train_idx].copy()
                    test_df_export  = df_base.loc[test_idx].copy()

                    col1, col2 = st.columns(2)

                    with col1:
                        st.download_button(
                            "⬇️ Download TRAIN set",
                            data=train_df_export.to_csv(index=False).encode("utf-8"),
                            file_name="train_split_physics_aware.csv",
                            mime="text/csv"
                        )

                    with col2:
                        st.download_button(
                            "⬇️ Download TEST set",
                            data=test_df_export.to_csv(index=False).encode("utf-8"),
                            file_name="test_split_physics_aware.csv",
                            mime="text/csv"
                        )

                else:
                    st.info("ℹ️ Enable advanced splitting to download Train/Test datasets.")

                
        st.session_state["split_indices"] = {
            "train_idx": train_idx,
            "val_idx": val_idx if "val_idx" in locals() else None,
            "test_idx": test_idx,
            "report": split_report
        }

        st.divider()
        st.header("① Select Features & Target")
        cols = df_base.columns.tolist()
        selected_features = st.multiselect("► Select **Features**", options=cols)
        selected_target = st.selectbox("► Select **Target**",
                                    options=[col for col in cols if col not in selected_features])

        if selected_features and selected_target:

            # Subset datafram

            X_all = df_base[selected_features]
            y_all = df_base[selected_target]

            split_info = st.session_state.get("split_indices", None)

            if split_info and split_info["train_idx"] is not None:
                train_idx = split_info["train_idx"]
                test_idx = split_info["test_idx"]
            else:
                # fallback (simple random split)
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                train_idx, test_idx = next(gss.split(X_all, y_all, groups))

            X_train = X_all.iloc[train_idx]
            y_train = y_all.iloc[train_idx]

            X_test = X_all.iloc[test_idx]
            y_test = y_all.iloc[test_idx]





            # -------------------------------------------------
            # User control: Run EDA or not
            # ------------------------------------------------- 
            st.divider()
            # -------------------------------------------------
            # Decide which data EDA should see
            # -------------------------------------------------
            split_info = st.session_state.get("split_indices", None)

            eda_df = df_base.copy()  # default = full dataset

            if split_info and split_info["train_idx"] is not None:
                eda_df = df_base.loc[split_info["train_idx"]]
                st.info("📊 EDA is running on TRAIN set only (split enabled)")
            else:
                st.info("📊 EDA is running on FULL dataset (no split)")

            run_eda = st.checkbox(
                "📊 Run Exploratory Data Analysis (EDA)",
                value=False,
                help="Enable this only if you want to compute statistics and plots (can be slow for large datasets)."
            )

            if run_eda:
                with st.expander("📊 EDA (Show/Hide)", expanded=False):
                    st.subheader("🟢 Exploratory Data Analysis (EDA)")
                    st.markdown("#### 🟣 Summary of Statistics")
                    summary = eda_df[selected_features + [selected_target]].describe().T
                    summary['skew'] = eda_df[selected_features + [selected_target]].skew()
                    summary['kurtosis'] = eda_df[selected_features + [selected_target]].kurtosis()


                    st.dataframe(summary.round(3))

                    st.markdown("#### 🟣 Scatter Plots + Histograms")
                    plots = plot_pairwise_corr_with_hist(
                        eda_df[selected_features + [selected_target]],
                        selected_target
                    )

                    for fig in plots:
                        st.pyplot(fig)

                    st.markdown("#### 🟣 Correlation Heatmap")
                    corr = eda_df[selected_features + [selected_target]].corr()

                    mask = np.triu(np.ones_like(corr, dtype=bool))
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr,  annot=True, fmt=".2f",
                                cmap="coolwarm", cbar=True, ax=ax,
                                square=True, linewidths=0.5,
                                annot_kws={"size": 9})
                    #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="left")
                    #ax.xaxis.set_ticks_position('bottom')
                    #ax.xaxis.set_label_position('bottom')
                    st.pyplot(fig)
                    st.caption(
                        f"EDA rows used: {len(eda_df)} "
                        f"({eda_df[filename_col].nunique()} unique runs)"
                    )

            # ================= Model =================
            #with st.container(border=True):
            st.divider()
            st.header("② Model Training and Evaluation")
            model_choice = st.radio(
                "Select Model",
                ["Random Forest", "CatBoost", "LightGBM", "GBRT"]
            )

            eval_method = st.radio(
                "Evaluation Method",
                [
                    "Train-Test Split",
                    "Train-Validation-Test (Industry Standard)",
                    "Group K-Fold Cross-Validation (Run-based)"
                ]
            )
 

            use_gridsearch = st.checkbox(
                "🔍 Enable Grid Search Hyperparameter Optimization",
                value=False,
                help="Uses GroupKFold and may take longer"
            )

            #if model_choice == "Random Forest":
            if model_choice == "Random Forest":
                base_model = RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1
                )

            elif model_choice == "CatBoost":
                base_model = CatBoostRegressor(
                    iterations=500,
                    depth=6,
                    learning_rate=0.05,
                    loss_function="RMSE",
                    verbose=False,
                    random_state=42
                )

            elif model_choice == "LightGBM":
                base_model = lgb.LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=-1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )

            elif model_choice == "GBRT":
                base_model = GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
            def get_param_grid(model_name):
                if model_name == "Random Forest":
                    return {
                        "n_estimators": [200, 400],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5],
                        "min_samples_leaf": [1, 3],
                        "max_features": ["sqrt"]
                    }

                elif model_name == "CatBoost":
                    return {
                        "depth": [4, 6, 8],
                        "learning_rate": [0.03, 0.05],
                        "iterations": [300, 500]
                    }

                elif model_name == "LightGBM":
                    return {
                        "n_estimators": [300, 600],
                        "learning_rate": [0.03, 0.05],
                        "num_leaves": [31, 63],
                        "subsample": [0.8],
                        "colsample_bytree": [0.8]
                    }

                elif model_name == "GBRT":
                    return {
                        "n_estimators": [200, 400],
                        "learning_rate": [0.03, 0.05],
                        "max_depth": [3, 4]
                    }

            final_model = None

            split_info = st.session_state.get("split_indices", None)

            has_predefined_split = (
                split_info is not None
                and split_info["train_idx"] is not None
                and split_info["test_idx"] is not None
                and "val_idx" in split_info
            )

            if eval_method == "Train-Test Split":
                if has_predefined_split:
                    # ✅ Use Section 0 split
                    st.info("📌 Using predefined split from Section 0")

                    train_idx = split_info["train_idx"]
                    test_idx  = split_info["test_idx"] 

                else:

                    # 🆕 User did NOT define a split → create one here
                    test_size = st.slider("Test Size (%)", 10, 50, 25) / 100

                    gss = GroupShuffleSplit(
                        n_splits=1,
                        test_size=test_size,
                        random_state=42
                    )

                    train_idx, test_idx = next(
                        gss.split(X_all, y_all, groups=groups)
                    )

                # ---------- Apply indices ----------
                X_train = X_all.loc[train_idx]
                y_train = y_all.loc[train_idx]

                X_test  = X_all.loc[test_idx]
                y_test  = y_all.loc[test_idx]

                st.info(f"Size of **Train set**: {X_train.shape}")
                st.info(f"Size of **Test set**: {X_test.shape}")

                if use_gridsearch:
                    param_grid = get_param_grid(model_choice)
                    cv = GroupKFold(n_splits=5)

                    pipeline = GridSearchCV(
                        estimator=base_model,
                        param_grid=param_grid,
                        scoring="neg_root_mean_squared_error",
                        cv=cv,
                        n_jobs=-1,
                        verbose=1
                    )

                    pipeline.fit(X_train, y_train, groups=groups.iloc[train_idx])

                    st.success("✅ Grid Search Completed")
                    st.write("🔧 Best Parameters:", pipeline.best_params_)

                    final_model = pipeline.best_estimator_

                else:
                    final_model = clone(base_model)
                    final_model.fit(X_train, y_train)

                # ---------- Predictions ----------
                y_train_pred = final_model.predict(X_train)
                y_test_pred = final_model.predict(X_test)

                train_metrics = get_metrics(y_train, y_train_pred)
                test_metrics = get_metrics(y_test, y_test_pred)

                result_df = pd.DataFrame(
                    [train_metrics, test_metrics],
                    columns=["MAE", "MSE", "RMSE", "R²", "A20 Index"],
                    index=["Train", "Test"]
                )

                st.subheader("Model Performance")
                st.dataframe(result_df.round(4))

                st.subheader("Predicted vs. Measured")
                st.markdown("##### 🔵 Training Set")
                plot_predicted_vs_measured_separately(
                    y_train, y_train_pred, "Train", model_choice, selected_target
                )
                st.markdown("##### 🟠 Testing Set")
                plot_predicted_vs_measured_separately(
                    y_test, y_test_pred, "Test", model_choice, selected_target
                )

                st.write("✅ Unique runs — Train:", groups.iloc[train_idx].nunique())
                st.write("✅ Unique runs — Test:", groups.iloc[test_idx].nunique())

 

                # --- Optional diagnostic plots (test set only) ---
                st.markdown("### Optional Performance Plots")
                with st.expander("Show / Hide Performance Plots", expanded=False):
                    show_tracking = st.checkbox("1️⃣ Prediction Tracking (with Mean Lines)")
                    show_error_mean_std = st.checkbox("2️⃣ Prediction Error with Mean ± STD")  
                    show_residuals_pred = st.checkbox("3️⃣ Residuals vs Predicted")
                    show_residual_dist = st.checkbox("4️⃣ Residual Distribution (Histogram + KDE)")
                    show_feat_import = st.checkbox("5️⃣ Feature Importance")
                    show_abs_err = st.checkbox("6️⃣ Error Magnitude (Absolute Error)")
                    
                    # ===== 1️⃣ Prediction Tracking (with Mean Lines) =====
                    if show_tracking:
                        fig, ax = plt.subplots(figsize=(10,5))
                        ax.plot(y_test.values, label="True Signal", color='teal', linewidth=2)
                        ax.plot(y_test_pred, label="Predicted Signal", color='orange', linestyle='--', linewidth=2)

                        # ✅ Mean lines
                        mean_true = np.mean(y_test.values)
                        mean_pred = np.mean(y_test_pred)
                        ax.axhline(mean_true, color='teal', linestyle=':', linewidth=1.5, label=f"Mean True = {mean_true:.2f}")
                        ax.axhline(mean_pred, color='orange', linestyle=':', linewidth=1.5, label=f"Mean Pred = {mean_pred:.2f}")

                        ax.set_xlabel("Sample Index")
                        ax.set_ylabel(selected_target)
                        ax.set_title("True vs. Predicted Signal (with Mean Lines)")
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)


                    # ===== 2️⃣ Prediction Error with Mean ± STD =====
                    if show_error_mean_std:
                        error_signal = y_test_pred - y_test.values
                        mean_error = np.mean(error_signal)
                        std_error = np.std(error_signal)
                        fig, ax = plt.subplots(figsize=(10,4))
                        ax.plot(error_signal, label="Error (Pred - True)", color='red', alpha=0.6)
                        ax.axhline(mean_error, color='blue', linestyle='--', linewidth=2, label=f"Mean Error = {mean_error:.4f}")
                        ax.axhline(0, color='black', linestyle=':', linewidth=1)
                        ax.fill_between(range(len(error_signal)),
                                        mean_error - std_error,
                                        mean_error + std_error,
                                        color='blue', alpha=0.1,
                                        label=f"±1 STD ({std_error:.4f})")
                        ax.set_xlabel("Sample Index")
                        ax.set_ylabel("Error (Pred - True)")
                        ax.set_title("Prediction Error with Mean ± STD")
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)

                    # ===== 3️⃣ Residuals vs Predicted =====
                    if show_residuals_pred:
                        residuals = y_test_pred - y_test.values
                        fig, ax = plt.subplots(figsize=(10,4))
                        ax.scatter(y_test_pred, residuals, color="purple", alpha=0.6)
                        ax.axhline(0, color="black", linestyle="--")
                        ax.set_xlabel("Predicted"); ax.set_ylabel("Residuals")
                        ax.set_title("Residuals vs Predicted")
                        st.pyplot(fig); plt.close(fig)

                    # ===== 4️⃣ Residual Distribution =====
                    if show_residual_dist:
                        fig, ax = plt.subplots(figsize=(10,4))
                        sns.histplot(residuals, kde=True, bins=25, color="orange", ax=ax)
                        ax.set_title("Residual Distribution (Histogram + KDE)")
                        st.pyplot(fig); plt.close(fig)

                    # ===== 5️⃣ Feature Importance =====
                    if show_feat_import and hasattr(final_model, "feature_importances_"):
                        imp = final_model.feature_importances_

                        fig, ax = plt.subplots(figsize=(10,4))
                        pd.Series(imp, index=selected_features).sort_values().plot.barh(ax=ax, color="teal")
                        ax.set_title("Feature Importance")
                        st.pyplot(fig); plt.close(fig)

                    # ===== 6️⃣ Absolute Error =====
                    if show_abs_err:
                        abs_err = np.abs(y_test_pred - y_test.values)
                        fig, ax = plt.subplots(figsize=(10,4))
                        ax.plot(abs_err, color="red", alpha=0.7)
                        ax.set_title("Absolute Error per Sample")
                        ax.set_xlabel("Sample Index"); ax.set_ylabel("|Error|")
                        st.pyplot(fig); plt.close(fig)

            #if eval_method == "Train-Test Split":
            elif eval_method == "Train-Validation-Test (Industry Standard)":

                if has_predefined_split:
                    st.info("📌 Using physics-aware Train / Validation / Test split from Section 0")

                    train_idx = split_info["train_idx"]
                    val_idx   = split_info["val_idx"]
                    test_idx  = split_info["test_idx"]

                    X_train = X_all.loc[train_idx]
                    y_train = y_all.loc[train_idx]

                    X_val = X_all.loc[val_idx]
                    y_val = y_all.loc[val_idx]

                    X_test = X_all.loc[test_idx]
                    y_test = y_all.loc[test_idx]

                else:
                    st.warning("⚠️ No predefined split — creating Train / Validation / Test")

                    col1, col2 = st.columns(2)

                    with col1:
                        val_size = st.slider(
                            "Validation Size (%)",
                            min_value=10,
                            max_value=30,
                            value=20,
                            step=5
                        ) / 100

                    with col2:
                        test_size = st.slider(
                            "Test Size (%)",
                            min_value=10,
                            max_value=30,
                            value=20,
                            step=5
                        ) / 100

                    if val_size + test_size >= 0.5:
                        st.error("❌ Validation + Test size must be < 50%")
                        st.stop()
                    # -------------------------
                    # Stage 1: TEST split
                    # -------------------------
                    gss_test = GroupShuffleSplit(
                        n_splits=1,
                        test_size=test_size,
                        random_state=42
                    )

                    trainval_idx, test_idx = next(
                        gss_test.split(X_all, y_all, groups=groups)
                    )

                    X_test = X_all.loc[test_idx]
                    y_test = y_all.loc[test_idx]

                    X_trainval = X_all.loc[trainval_idx]
                    y_trainval = y_all.loc[trainval_idx]
                    groups_trainval = groups.loc[trainval_idx]

                    # -------------------------
                    # Stage 2: VALIDATION split
                    # -------------------------
                    val_fraction_of_trainval = val_size / (1.0 - test_size)

                    gss_val = GroupShuffleSplit(
                        n_splits=1,
                        test_size=val_fraction_of_trainval,
                        random_state=42
                    )

                    tr_idx, val_idx = next(
                        gss_val.split(X_trainval, y_trainval, groups=groups_trainval)
                    )

                    X_train = X_trainval.iloc[tr_idx]
                    y_train = y_trainval.iloc[tr_idx]

                    X_val = X_trainval.iloc[val_idx]
                    y_val = y_trainval.iloc[val_idx]

                # -------------------------
                # Final sanity info
                # -------------------------
                st.info(f"Train size: {X_train.shape}")
                st.info(f"Validation size: {X_val.shape}")
                st.info(f"Test size (unseen): {X_test.shape}")

                # ==================================================
                # MODEL TRAINING
                # ==================================================
                if use_gridsearch:
                    param_grid = get_param_grid(model_choice)
                    cv = GroupKFold(n_splits=5)

                    grid = GridSearchCV(
                        estimator=base_model,
                        param_grid=param_grid,
                        scoring="neg_root_mean_squared_error",
                        cv=cv,
                        n_jobs=-1,
                        verbose=1
                    )

                    # ⚠️ Grid search sees ONLY train + validation
                    grid.fit(X_trainval, y_trainval, groups=groups_trainval)

                    final_model = grid.best_estimator_
                    st.success("✅ Grid Search Completed")

                else:
                    final_model = clone(base_model)
                    final_model.fit(X_train, y_train)

                # ==================================================
                # PREDICTIONS (single source of truth)
                # ==================================================
                y_train_pred = final_model.predict(X_train)
                y_val_pred   = final_model.predict(X_val)
                y_test_pred  = final_model.predict(X_test)

                # ==================================================
                # METRICS (single table, no duplication)
                # ==================================================
                results_df = pd.DataFrame(
                    [
                        get_metrics(y_train, y_train_pred),
                        get_metrics(y_val, y_val_pred),
                        get_metrics(y_test, y_test_pred)
                    ],
                    columns=["MAE", "MSE", "RMSE", "R²", "A20 Index"],
                    index=["Train", "Validation", "Test (Unseen)"]
                )

                st.subheader("Model Performance")
                st.dataframe(results_df.round(4))

                # ==================================================
                # PLOTS (UNCHANGED STYLE)
                # ==================================================
                st.subheader("Predicted vs. Measured")

                st.markdown("##### 🔵 Training Set")
                plot_predicted_vs_measured_separately(
                    y_train, y_train_pred, "Train", model_choice, selected_target
                )

                st.markdown("##### 🟢 Validation Set")
                plot_predicted_vs_measured_separately(
                    y_val, y_val_pred, "Validation", model_choice, selected_target
                )

                st.markdown("##### 🟠 Testing Set")
                plot_predicted_vs_measured_separately(
                    y_test, y_test_pred, "Test", model_choice, selected_target
                )

                st.write("✅ Number of unique runs (entire dataset):", groups.nunique())

                # ==================================================
                # OPTIONAL PERFORMANCE PLOTS (UNCHANGED)
                # ==================================================
                st.markdown("### Optional Performance Plots")
                with st.expander("Show / Hide Performance Plots", expanded=False):
                    show_tracking = st.checkbox("1️⃣ Prediction Tracking (with Mean Lines)")
                    show_error_mean_std = st.checkbox("2️⃣ Prediction Error with Mean ± STD")
                    show_residuals_pred = st.checkbox("3️⃣ Residuals vs Predicted")
                    show_residual_dist = st.checkbox("4️⃣ Residual Distribution (Histogram + KDE)")
                    show_feat_import = st.checkbox("5️⃣ Feature Importance")
                    show_abs_err = st.checkbox("6️⃣ Error Magnitude (Absolute Error)")

                    # ===== 1️⃣ Prediction Tracking (with Mean Lines) =====
                    if show_tracking:
                        fig, ax = plt.subplots(figsize=(10,5))
                        ax.plot(y_test.values, label="True Signal", color='teal', linewidth=2)
                        ax.plot(y_test_pred, label="Predicted Signal", color='orange', linestyle='--', linewidth=2)

                        mean_true = np.mean(y_test.values)
                        mean_pred = np.mean(y_test_pred)

                        ax.axhline(mean_true, color='teal', linestyle=':', linewidth=1.5)
                        ax.axhline(mean_pred, color='orange', linestyle=':', linewidth=1.5)

                        ax.set_xlabel("Sample Index")
                        ax.set_ylabel(selected_target)
                        ax.set_title("True vs. Predicted Signal (with Mean Lines)")
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)

                    # ===== 2️⃣ Prediction Error with Mean ± STD =====
                    if show_error_mean_std:
                        error_signal = y_test_pred - y_test.values
                        mean_error = np.mean(error_signal)
                        std_error = np.std(error_signal)

                        fig, ax = plt.subplots(figsize=(10,4))
                        ax.plot(error_signal, color='red', alpha=0.6)
                        ax.axhline(mean_error, color='blue', linestyle='--', linewidth=2)
                        ax.axhline(0, color='black', linestyle=':')
                        ax.fill_between(
                            range(len(error_signal)),
                            mean_error - std_error,
                            mean_error + std_error,
                            color='blue',
                            alpha=0.1
                        )
                        ax.set_xlabel("Sample Index")
                        ax.set_ylabel("Error (Pred - True)")
                        st.pyplot(fig)
                        plt.close(fig)

                    # ===== 3️⃣ Residuals vs Predicted =====
                    if show_residuals_pred:
                        residuals = y_test_pred - y_test.values
                        fig, ax = plt.subplots(figsize=(10,4))
                        ax.scatter(y_test_pred, residuals, color="purple", alpha=0.6)
                        ax.axhline(0, color="black", linestyle="--")
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Residuals")
                        st.pyplot(fig)
                        plt.close(fig)

                    # ===== 4️⃣ Residual Distribution =====
                    if show_residual_dist:
                        fig, ax = plt.subplots(figsize=(10,4))
                        sns.histplot(residuals, kde=True, bins=25, color="orange", ax=ax)
                        ax.set_title("Residual Distribution")
                        st.pyplot(fig)
                        plt.close(fig)

                    # ===== 5️⃣ Feature Importance =====
                    if show_feat_import and hasattr(final_model, "feature_importances_"):
                        imp = final_model.feature_importances_
                        fig, ax = plt.subplots(figsize=(10,4))
                        pd.Series(imp, index=selected_features).sort_values().plot.barh(ax=ax, color="teal")
                        ax.set_title("Feature Importance")
                        st.pyplot(fig)
                        plt.close(fig)

                    # ===== 6️⃣ Absolute Error =====
                    if show_abs_err:
                        abs_err = np.abs(y_test_pred - y_test.values)
                        fig, ax = plt.subplots(figsize=(10,4))
                        ax.plot(abs_err, color="red", alpha=0.7)
                        ax.set_title("Absolute Error per Sample")
                        ax.set_xlabel("Sample Index")
                        ax.set_ylabel("|Error|")
                        st.pyplot(fig)
                        plt.close(fig)


            else:  # ================= Group K-Fold CV + Hold-out Test =================
                if has_predefined_split:
                    st.info("📌 Using predefined TEST set from Section 0")

                    train_idx = split_info["train_idx"]
                    test_idx  = split_info["test_idx"]

                else:
                    test_size = st.slider("Test Size (%)", 10, 50, 20) / 100

                    gss_test = GroupShuffleSplit(
                        n_splits=1,
                        test_size=test_size,
                        random_state=42
                    )

                    train_idx, test_idx = next(
                        gss_test.split(X_all, y_all, groups=groups)
                    )

                # --- Apply split ---
                X_train = X_all.loc[train_idx]
                y_train = y_all.loc[train_idx]
                groups_train = groups.loc[train_idx]

                X_test = X_all.loc[test_idx]
                y_test = y_all.loc[test_idx]


                st.info(f"Train size (CV pool): {X_train.shape}")
                st.info(f"Test size (unseen): {X_test.shape}")

                # ==========================================================
                # 2️⃣ GRID SEARCH (CV only on TRAIN)
                # ==========================================================
                if use_gridsearch:
                    param_grid = get_param_grid(model_choice)

                    grid = GridSearchCV(
                        estimator=base_model,
                        param_grid=param_grid,
                        scoring="neg_root_mean_squared_error",
                        cv=gkf,
                        n_jobs=-1,
                        verbose=1
                    )

                    # ⚠️ VERY IMPORTANT: fit ONLY on training data
                    grid.fit(X_train, y_train, groups=groups_train)

                    st.success("✅ Grid Search Completed")
                    st.write("🔧 Best Parameters:", grid.best_params_)

                    final_model = grid.best_estimator_

                    # ---------- Final evaluation on UNSEEN TEST ----------
                    y_test_pred = final_model.predict(X_test)
                    test_metrics = get_metrics(y_test, y_test_pred)

                    st.subheader("Final Unseen Test Performance")
                    st.dataframe(
                        pd.DataFrame(
                            [test_metrics],
                            columns=["MAE", "MSE", "RMSE", "R²", "A20 Index"],
                            index=["Test"]
                        ).round(4)
                    )

                # ==========================================================
                # 3️⃣ PURE CV (NO GRID SEARCH) + FINAL TEST
                # ==========================================================
                else:
                    train_scores, val_scores = [], []
                    all_train_true, all_train_pred = [], []
                    all_val_true, all_val_pred = [], []

                    # -------------------------------
                    # CV loop (TRAIN ONLY)
                    # -------------------------------
                    for fold, (tr_idx, val_idx) in enumerate(
                        gkf.split(X_train, y_train, groups=groups_train), 1
                    ):
                        X_tr = X_train.iloc[tr_idx]
                        y_tr = y_train.iloc[tr_idx]

                        X_val = X_train.iloc[val_idx]
                        y_val = y_train.iloc[val_idx]

                        model = clone(base_model)
                        model.fit(X_tr, y_tr)

                        y_tr_pred = model.predict(X_tr)
                        y_val_pred = model.predict(X_val)

                        train_scores.append(get_metrics(y_tr, y_tr_pred))
                        val_scores.append(get_metrics(y_val, y_val_pred))

                        all_train_true.extend(y_tr)
                        all_train_pred.extend(y_tr_pred)

                        all_val_true.extend(y_val)
                        all_val_pred.extend(y_val_pred)

                    # -------------------------------
                    # CV summary
                    # -------------------------------
                    avg_train = np.mean(train_scores, axis=0)
                    avg_val = np.mean(val_scores, axis=0)

                    results_df = pd.DataFrame(
                        {
                            "Train (CV Avg)": avg_train,
                            "Validation (CV Avg)": avg_val
                        },
                        index=["MAE", "MSE", "RMSE", "R²", "A20 Index"]
                    ).T

                    st.subheader("Cross-Validation Performance")
                    st.dataframe(results_df.round(4))

                    st.subheader("Predicted vs. Measured (CV)")
                    plot_predicted_vs_measured_separately(
                        np.array(all_train_true),
                        np.array(all_train_pred),
                        "Train", model_choice, selected_target
                    )
                    plot_predicted_vs_measured_separately(
                        np.array(all_val_true),
                        np.array(all_val_pred),
                        "Validation", model_choice, selected_target
                    )

                    # -------------------------------
                    # 4️⃣ FINAL MODEL → UNSEEN TEST
                    # -------------------------------
                    final_model = clone(base_model)
                    final_model.fit(X_train, y_train)

                    y_test_pred = final_model.predict(X_test)
                    test_metrics = get_metrics(y_test, y_test_pred)

                    st.subheader("Final Unseen Test Performance")
                    st.dataframe(
                        pd.DataFrame(
                            [test_metrics],
                            columns=["MAE", "MSE", "RMSE", "R²", "A20 Index"],
                            index=["Test"]
                        ).round(4)
                    )


            # ==================================================
            #                   Interpretation  
            # ==================================================
            st.divider()
            st.header("③ Model Interpretation")
            show_shap = st.checkbox("4.1 Show SHAP Plots")

            if show_shap and final_model:

                try:
                    shap_dataset_choice = st.radio(
                        "Compute SHAP on:",
                        ["Train Set", "Test Set"],
                        horizontal=True
                    )

                    shap_X = X_train if shap_dataset_choice == "Train Set" else X_test

                    explainer = shap.Explainer(final_model, shap_X)
                    shap_values = explainer(shap_X)

                    shap_plot_type = st.radio(
                        "Select visualization type:",
                        [
                            "1️⃣ Waterfall Plot",
                            "2️⃣ Force Plot",
                            "3️⃣ Summary / Beeswarm Plot",
                            "4️⃣ Dependence Plot",
                            "5️⃣ Bar Plot"
                        ]
                    )

                    if shap_plot_type == "1️⃣ Waterfall Plot":
                        idx = st.number_input(
                            "Select sample index",
                            0,
                            len(shap_X) - 1,
                            0
                        )
                        fig = plt.figure()
                        shap.plots.waterfall(shap_values[idx], show=False)
                        st.pyplot(fig)
                        plt.close(fig)

                    elif shap_plot_type == "2️⃣ Force Plot":
                        idx = st.number_input(
                            "Select sample index",
                            0,
                            len(shap_X) - 1,
                            0
                        )
                        fig = plt.figure()
                        shap.plots.force(shap_values[idx], matplotlib=True, show=False)
                        st.pyplot(fig)
                        plt.close(fig)

                    elif shap_plot_type == "3️⃣ Summary / Beeswarm Plot":
                        fig = plt.figure()
                        shap.plots.beeswarm(shap_values, show=False)
                        st.pyplot(fig)
                        plt.close(fig)

                    elif shap_plot_type == "4️⃣ Dependence Plot":
                        selected_feat = st.selectbox("Select feature", selected_features)
                        fig = plt.figure()
                        shap.plots.scatter(shap_values[:, selected_feat], show=False)
                        st.pyplot(fig)
                        plt.close(fig)

                    elif shap_plot_type == "5️⃣ Bar Plot":
                        fig = plt.figure()
                        shap.plots.bar(shap_values, show=False)
                        st.pyplot(fig)
                        plt.close(fig)

                except Exception as e:
                    st.warning(f"⚠️ SHAP visualization failed: {e}")

            # ===============================================
            #                   New Dataset  
            # ===============================================
            st.divider()
            st.header("④ Test Final Model")

            if final_model:
                test_mode = st.radio(
                    #"🞉 ## Choose how to test the model: ##",
                    "**🞉 Choose how to test the model:**",
                    [
                        "1️⃣ Upload CSV file",
                        "2️⃣ Enter feature values manually",
                        "3️⃣ Enter source path (multiple datasets)"
                    ]
                )

                # ---------------------------
                # Option 1: Upload a file
                # ---------------------------
                if test_mode == "1️⃣ Upload CSV file":
                    new_data_file = st.file_uploader("📤 Upload a dataset (CSV)", type=["csv"], key="new_dataset")
                    if new_data_file:
                        new_df = pd.read_csv(new_data_file)

                        missing_cols = [col for col in selected_features + [selected_target] if col not in new_df.columns]
                        if missing_cols:
                            st.warning(f"Missing required columns: {missing_cols}")
                        else:
                            new_X = new_df[selected_features]
                            new_y_true = new_df[selected_target]

                            # --- Generate predictions
                            new_y_pred = final_model.predict(new_X)

                            with st.container(border=True):
                                # --- Option: insert predictions into dataset
                                st.markdown("##### ➕ Insert Predicted Target into Uploaded Dataset")
                                insert_pred_option = st.checkbox("Insert predicted target column into uploaded data")

                                if insert_pred_option:
                                    pred_col_name = f"Predicted_{selected_target}"
                                    new_df[pred_col_name] = new_y_pred
                                    st.success(f"✅ Column '{pred_col_name}' added to the dataset!")

                                    st.markdown("##### 📄 Preview of Updated Dataset")
                                    st.dataframe(new_df.head())

                                    # Allow user to export the updated dataset
                                    csv_pred = new_df.to_csv(index=False).encode("utf-8")
                                    file_label = os.path.splitext(new_data_file.name)[0]
                                    out_name = f"{file_label}_with_{pred_col_name}.csv"
                                    st.download_button(
                                        "⬇️ Download Dataset with Predictions",
                                        data=csv_pred,
                                        file_name=out_name,
                                        mime="text/csv"
                                    )

                            with st.container(border=True):
                                # Show metrics
                                new_metrics = get_metrics(new_y_true, new_y_pred)
                                metrics_df = pd.DataFrame(
                                    [new_metrics],
                                    columns=["MAE", "MSE", "RMSE", "R²", "A20 Index"],
                                    index=[new_data_file.name]
                                )
                                st.markdown("##### 🎯 Performance on New Dataset")

                                # --- Compute mean comparison metrics ---
                                mean_true = np.mean(new_y_true)
                                mean_pred = np.mean(new_y_pred)
                                mean_diff = mean_pred - mean_true
                                error_percent = (mean_diff / mean_true) * 100

                                metrics_df["Error_Mean"] = error_percent
                                metrics_df["Mean_True"] = mean_true
                                metrics_df["Mean_Predicted"] = mean_pred
                                st.dataframe(metrics_df.round(8))

                                file_label = os.path.splitext(new_data_file.name)[0]
                                st.markdown("##### 📊 Predicted vs. Measured (New Dataset)")
                                plot_predicted_vs_measured_separately(
                                    new_y_true,
                                    new_y_pred,
                                    file_label,
                                    model_choice,
                                    selected_target
                                )

                                st.markdown("##### 📈 Performance Plots (New Dataset)")
                                with st.expander("Show / Hide Diagnostic Plots", expanded=False):
                                    show_tracking = st.checkbox("1️⃣ Prediction Tracking (with Mean Lines)", key="new_tracking")
                                    show_error_mean_std = st.checkbox("2️⃣ Prediction Error with Mean ± STD", key="new_err_meanstd")
                                    show_residuals_pred = st.checkbox("3️⃣ Residuals vs Predicted", key="new_resid_pred")
                                    show_residual_dist = st.checkbox("4️⃣ Residual Distribution (Histogram + KDE)", key="new_resid_dist")
                                    show_abs_err = st.checkbox("5️⃣ Error Magnitude (Absolute Error)", key="new_abs_err")

                                    residuals = new_y_pred - new_y_true
                                    abs_err = np.abs(residuals)
                                    mean_error = np.mean(residuals)
                                    std_error = np.std(residuals)

                                    if show_tracking:
                                        fig, ax = plt.subplots(figsize=(10,5))
                                        ax.plot(new_y_true, label="True Signal", color='teal', linewidth=2)
                                        ax.plot(new_y_pred, label="Predicted Signal", color='orange', linestyle='--', linewidth=2)
                                        ax.axhline(np.mean(new_y_true), color='teal', linestyle=':', linewidth=1.5)
                                        ax.axhline(np.mean(new_y_pred), color='orange', linestyle=':', linewidth=1.5)
                                        ax.set_xlabel("Sample Index")
                                        ax.set_ylabel(selected_target)
                                        ax.set_title("True vs. Predicted Signal (with Mean Lines)")
                                        ax.legend()
                                        st.pyplot(fig)
                                        plt.close(fig)

                                    if show_error_mean_std:
                                        fig, ax = plt.subplots(figsize=(10,4))
                                        ax.plot(residuals, color='red', alpha=0.6)
                                        ax.axhline(mean_error, color='blue', linestyle='--', linewidth=2)
                                        ax.axhline(0, color='black', linestyle=':')
                                        ax.fill_between(range(len(residuals)),
                                                        mean_error - std_error,
                                                        mean_error + std_error,
                                                        color='blue', alpha=0.1)
                                        ax.set_xlabel("Sample Index")
                                        ax.set_ylabel("Error (Pred - True)")
                                        st.pyplot(fig)
                                        plt.close(fig)

                                    if show_residuals_pred:
                                        fig, ax = plt.subplots(figsize=(10,4))
                                        ax.scatter(new_y_pred, residuals, color="purple", alpha=0.6)
                                        ax.axhline(0, color="black", linestyle="--")
                                        ax.set_xlabel("Predicted")
                                        ax.set_ylabel("Residuals")
                                        st.pyplot(fig)
                                        plt.close(fig)

                                    if show_residual_dist:
                                        fig, ax = plt.subplots(figsize=(10,4))
                                        sns.histplot(residuals, kde=True, bins=25, color="orange", ax=ax)
                                        ax.set_title("Residual Distribution")
                                        st.pyplot(fig)
                                        plt.close(fig)

                                    if show_abs_err:
                                        fig, ax = plt.subplots(figsize=(10,4))
                                        ax.plot(abs_err, color="red", alpha=0.7)
                                        ax.set_title("Absolute Error per Sample")
                                        ax.set_xlabel("Sample Index")
                                        ax.set_ylabel("|Error|")
                                        st.pyplot(fig)
                                        plt.close(fig)

                # ---------------------------
                # Option 2: Manual input
                # ---------------------------
                elif test_mode == "2️⃣ Enter feature values manually":
                    st.subheader("2️⃣ Enter feature values manually")
                    sample_input = {}
                    for feat in selected_features:
                        val = st.number_input(f"Enter value for {feat}", value=float(df[feat].mean()))
                        sample_input[feat] = val

                    if st.button("Predict Target"):
                        sample_df = pd.DataFrame([sample_input])
                        pred = final_model.predict(sample_df)[0]
                        st.success(f"🎯 Predicted {selected_target}: **{pred:.4f}**")
                # ---------------------------
                # Option 3: Folder with multiple datasets
                # ---------------------------
                elif test_mode == "3️⃣ Enter source path (multiple datasets)":
                    import glob, zipfile
                    st.subheader("📂 Apply Model to Multiple Datasets")
                    src_folder = st.text_input("Enter the source folder path containing CSV files (subfolders will also be checked):")

                    if src_folder and os.path.isdir(src_folder):
                        # Recursively search all subfolders for .csv files
                        csv_files = glob.glob(os.path.join(src_folder, "**", "*.csv"), recursive=True)

                        if not csv_files:
                            st.warning("⚠️ No CSV files found in the given folder or its subfolders.")
                        else:
                            st.info(f"📂 Found {len(csv_files)} CSV files (including subfolders).")

                            pred_col_name = f"Predicted_{selected_target}"
                            progress = st.progress(0)
                            processed_files = []

                            for i, fpath in enumerate(csv_files, 1):
                                fname = os.path.basename(fpath)
                                try:
                                    df_new = pd.read_csv(fpath)
                                    if all(col in df_new.columns for col in selected_features):
                                        X_new = df_new[selected_features]
                                        preds = final_model.predict(X_new)
                                        df_new[pred_col_name] = preds

                                        out_path = os.path.join(
                                            os.path.dirname(fpath),
                                            f"{os.path.splitext(fname)[0]}_with_{pred_col_name}.csv"
                                        )
                                        df_new.to_csv(out_path, index=False)
                                        processed_files.append(out_path)
                                    else:
                                        st.warning(f"⚠️ Skipping {fname} (missing required columns).")
                                except Exception as e:
                                    st.error(f"❌ Error processing {fname}: {e}")
                                progress.progress(i / len(csv_files))

                            '''
                            # Create ZIP (preserve folder structure)
                            if processed_files:
                                zip_path = os.path.join(src_folder, "datasets_with_predictions.zip")
                                with zipfile.ZipFile(zip_path, "w") as zipf:
                                    for f in processed_files:
                                        arcname = os.path.relpath(f, src_folder)
                                        zipf.write(f, arcname)

                                with open(zip_path, "rb") as f:
                                    st.download_button(
                                        "⬇️ Download All Updated Datasets (ZIP)",
                                        data=f,
                                        file_name="datasets_with_predictions.zip",
                                        mime="application/zip"
                                    )
                                st.success(f"✅ Processed {len(processed_files)} files successfully!")
                            '''
                    else:
                        st.warning("Please enter a valid folder path.")
