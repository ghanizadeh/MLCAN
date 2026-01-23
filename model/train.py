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
            ax_main.plot(line_x, line_y, color='red', linestyle='--',
                        linewidth=2, label=f'R² = {r2:.2f}')
            ax_main.legend(loc='upper center')
            ax_main.set_xlabel(col, fontsize=8)
            ax_main.set_ylabel(target_col, fontsize=8)

            ax_xhist.hist(x, bins=15, color='green', edgecolor='black')

            ax_yhist.hist(y, bins=15, orientation='horizontal',
                        color='green', edgecolor='black')


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
                with st.expander("📊 EDddddA (Show/Hide)", expanded=False):
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
            eval_method = st.radio("Evaluation Method", ["Train-Test Split", "Group K-Fold Cross-Validation (Run-based)"])
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
                    #st.write("🔹 Best Parameters:", grid.best_params_)

                    #model = pipeline.best_estimator_
                else:
                    pipeline = clone(base_model)
                    pipeline.fit(X_train, y_train)
               
                #pipeline = base_model
                #pipeline.fit(X_train, y_train)
                final_model = pipeline
                y_train_pred = pipeline.predict(X_train)
                y_test_pred = pipeline.predict(X_test)

                train_metrics = get_metrics(y_train, y_train_pred)
                test_metrics = get_metrics(y_test, y_test_pred)

                result_df = pd.DataFrame([train_metrics, test_metrics],
                                        columns=["MAE", "MSE", "RMSE", "R²", "A20 Index"],
                                        index=["Train", "Test"])
                st.subheader("Model Performance")
                st.dataframe(result_df.round(4))

                st.subheader("Predicted vs. Measured")
                st.markdown("##### 🔵 Training Set")
                plot_predicted_vs_measured_separately(y_train, y_train_pred, "Train", model_choice, selected_target)
                st.markdown("##### 🟠 Testing Set")
                plot_predicted_vs_measured_separately(y_test, y_test_pred, "Test", model_choice, selected_target)
 
                st.write("✅ Number of unique runs:", groups.nunique())
 

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

            #else:  # K-Fold CV
            else:  # Group K-Fold CV
                k = st.slider("Number of Groups (Folds)", 2, 10, 5)
                gkf = GroupKFold(n_splits=k)

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

                    grid.fit(X, y, groups=groups)

                    st.success("✅ Grid Search Completed")
                    st.write("🔧 Best Parameters:", grid.best_params_)

                    final_model = grid.best_estimator_

                    # --- Evaluate CV performance of best model ---
                    y_pred = final_model.predict(X)
                    metrics = get_metrics(y, y_pred)

                    results_df = pd.DataFrame(
                        [metrics],
                        columns=["MAE", "MSE", "RMSE", "R²", "A20 Index"],
                        index=["CV (Optimized)"]
                    )

                    st.subheader("Model Performance")
                    st.dataframe(results_df.round(4))

                else:
                    # ---- Your ORIGINAL loop (unchanged) ----
                    train_scores, val_scores = [], []
                    all_train_true, all_train_pred = [], []
                    all_val_true, all_val_pred = [], []

                    for fold, (train_idx, val_idx) in enumerate(
                        gkf.split(X, y, groups=groups), 1
                    ):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                        model = clone(base_model)
                        model.fit(X_train, y_train)

                        y_train_pred = model.predict(X_train)
                        y_val_pred = model.predict(X_val)

                        train_scores.append(get_metrics(y_train, y_train_pred))
                        val_scores.append(get_metrics(y_val, y_val_pred))

                        all_train_true.extend(y_train)
                        all_train_pred.extend(y_train_pred)
                        all_val_true.extend(y_val)
                        all_val_pred.extend(y_val_pred)

                    avg_train = np.mean(train_scores, axis=0)
                    avg_val = np.mean(val_scores, axis=0)

                    results_df = pd.DataFrame(
                        {
                            "Train (CV Avg)": avg_train,
                            "Validation (CV Avg)": avg_val
                        },
                        index=["MAE", "MSE", "RMSE", "R²", "A20 Index"]
                    ).T

                    st.subheader("Model Performance")
                    st.dataframe(results_df.round(4))

                    st.subheader("Predicted vs. Measured")
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

                    final_model = clone(base_model)
                    final_model.fit(X, y)

            # ==================================================
            #                   Interpretation  
            # ==================================================
            st.divider()
            st.header("③ Model Interpretation")
            show_shap = st.checkbox("4.1 Show SHAP Plots")

            if show_shap and final_model:
                try:
                    feature_names = selected_features

                    # Scale X and create explainer
                    explainer = shap.Explainer(final_model, X)
                    shap_values = explainer(X)
 


                    # --- User selects which SHAP visualization to show ---
                    st.markdown("### Choose SHAP Plot Type")
                    shap_plot_type = st.radio(
                        "Select visualization type:",
                        [
                            "1️⃣ Waterfall Plot",
                            "2️⃣ Force Plot",
                            "3️⃣ Summary / Beeswarm Plot",
                            "4️⃣ Dependence Plot",
                            "5️⃣ Bar Plot (Feature Importance)"
                        ],
                        horizontal=False
                    )

                    # ---------------------------
                    # 1️⃣ Waterfall Plot
                    # ---------------------------
                    if "Waterfall" in shap_plot_type:
                        st.info(
                            "💧 **Waterfall Plot:** Shows how each feature pushes an individual prediction "
                            "from the baseline value to the final prediction."
                        )
                        idx = st.number_input("Select sample index", 0, len(X) - 1, 0)
                        fig = plt.figure()
                        shap.plots.waterfall(shap_values[idx], show=False)
                        st.pyplot(fig)
                        plt.close()

                    # ---------------------------
                    # 2️⃣ Force Plot
                    # ---------------------------
                    elif "Force" in shap_plot_type:
                        st.info(
                            "🧭 **Force Plot:** Compact version of the waterfall plot, showing features "
                            "as arrows pushing toward higher or lower predictions."
                        )
                        idx = st.number_input("Select sample index", 0, len(X) - 1, 0)

                        fig = plt.figure()
                        shap.plots.waterfall(shap_values[idx], show=False)
                        st.pyplot(fig)
                        plt.close(fig)

                    # ---------------------------
                    # 3️⃣ Summary / Beeswarm Plot
                    # ---------------------------
                    elif "Summary" in shap_plot_type:
                        st.info(
                            "🐝 **Beeswarm Plot:** Displays feature importance and the effect of feature "
                            "values across all samples."
                        )
                        fig = plt.figure()
                        shap.plots.beeswarm(shap_values, show=False)
                        st.pyplot(fig)
                        plt.close(fig)


                    # ---------------------------
                    # 4️⃣ Dependence Plot
                    # ---------------------------
                    elif "Dependence" in shap_plot_type:
                        st.info(
                            "📈 **Dependence Plot:** Shows how a single feature’s value impacts its SHAP "
                            "contribution. Useful for detecting non-linear or interaction effects."
                        )

                        selected_feat = st.selectbox("Select feature", feature_names)

                        fig = plt.figure()
                        shap.plots.scatter(
                            shap_values[:, selected_feat],
                            color=shap_values
                        )
                        st.pyplot(fig)
                        plt.close(fig)


                    # ---------------------------
                    # 5️⃣ Bar Plot
                    # ---------------------------
                    elif "Bar Plot" in shap_plot_type:
                        st.info(
                            "📊 **Bar Plot:** Displays the average absolute SHAP value for each feature, "
                            "indicating overall feature importance."
                        )
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

