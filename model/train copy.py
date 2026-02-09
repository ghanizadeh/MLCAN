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
    uploaded_file = st.file_uploader("► Upload your file (csv)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.divider()
        st.header("① Select Features & Target")
        cols = df.columns.tolist()
        selected_features = st.multiselect("► Select **Features**", options=cols)
        selected_target = st.selectbox("► Select **Target**",
                                    options=[col for col in cols if col not in selected_features])

        if selected_features and selected_target:
            # Subset dataframe
            # Find Filename column case-insensitively
            filename_col = next(
                c for c in df.columns if c.lower() == "filename"
            )
            df_sub = df[selected_features + [selected_target, filename_col]]

            # Record original size
            original_size = df_sub.shape[0]

            # Clean: drop rows where any selected feature OR target is NaN
            df_clean = df_sub.dropna().drop_duplicates()

            # Record cleaned size
            cleaned_size = df_clean.shape[0]

            # Report to Streamlit
            st.info(f"🧹 Data cleaned (NAN & Dulpicattion removal): Original size = **{original_size}**, Cleaned size = **{cleaned_size}**")

            # Update X and y AFTER cleaning
            X = df_clean[selected_features]
            y = df_clean[selected_target]
            groups = df_clean[filename_col]

            if y.isna().any():
                st.error("❌ Target column still has NaN values after cleaning!")
                st.stop()

            st.divider()

            # -------------------------------------------------
            # User control: Run EDA or not
            # ------------------------------------------------- 
            run_eda = st.checkbox(
                "📊 Run Exploratory Data Analysis (EDA)",
                value=False,
                help="Enable this only if you want to compute statistics and plots (can be slow for large datasets)."
            )

            if run_eda:
                with st.expander("📊 EDA (Show/Hide)", expanded=False):
                    st.subheader("🟢 Exploratory Data Analysis (EDA)")
                    st.markdown("#### 🟣 Summary of Statistics")
                    summary = df[selected_features + [selected_target]].describe().T
                    summary['skew'] = df[selected_features + [selected_target]].skew()
                    summary['kurtosis'] = df[selected_features + [selected_target]].kurtosis()

                    st.dataframe(summary.round(3))

                    st.markdown("#### 🟣 Scatter Plots + Histograms")
                    plots = plot_pairwise_corr_with_hist(df[selected_features + [selected_target]], selected_target)
                    for fig in plots:
                        st.pyplot(fig)

                    st.markdown("#### 🟣 Correlation Heatmap")
                    corr = df[selected_features + [selected_target]].corr()
                    mask = np.triu(np.ones_like(corr, dtype=bool))
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                                cmap="coolwarm", cbar=True, ax=ax,
                                square=True, linewidths=0.5,
                                annot_kws={"size": 9})
                    #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="left")
                    #ax.xaxis.set_ticks_position('bottom')
                    #ax.xaxis.set_label_position('bottom')
                    st.pyplot(fig)

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

            if eval_method == "Train-Validation-Test (Industry Standard)":
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

            if eval_method == "Train-Test Split":
                test_size = st.slider("Test Size (%)", 10, 50, 25) / 100
                gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
                train_idx, test_idx = next(gss.split(X, y, groups=groups))

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

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

                # ==================================================
                # Stage 0: sanity check (already defined above)
                # val_size, test_size come from sliders
                # ==================================================

                # ---------- Stage 1: Hold-out TEST ----------
                gss_test = GroupShuffleSplit(
                    n_splits=1,
                    test_size=test_size,
                    random_state=42
                )

                trainval_idx, test_idx = next(
                    gss_test.split(X, y, groups=groups)
                )

                X_trainval = X.iloc[trainval_idx]
                y_trainval = y.iloc[trainval_idx]
                groups_trainval = groups.iloc[trainval_idx]

                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]

                # ---------- Stage 2: Split TRAIN / VALIDATION ----------
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

                # -------------------------------
                # User controls
                # -------------------------------
                k = st.slider("Number of Groups (Folds)", 2, 10, 5)
                test_size = st.slider("Test Size (%)", 10, 50, 20) / 100

                gkf = GroupKFold(n_splits=k)

                # -------------------------------
                # 1️⃣ HOLD-OUT TEST (UNSEEN)
                # -------------------------------
                gss_test = GroupShuffleSplit(
                    n_splits=1,
                    test_size=test_size,
                    random_state=42
                )

                train_idx, test_idx = next(
                    gss_test.split(X, y, groups=groups)
                )

                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                groups_train = groups.iloc[train_idx]

                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]

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
