def show_ML_model_page():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shap
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.inspection import PartialDependenceDisplay
    from sklearn.pipeline import Pipeline
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    import os

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

        plt.legend(title=f"{dataset_type}:\nR²={r2:.2f} & MAE={mae:.2f}")
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
            df_sub = df[selected_features + [selected_target]]

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

            if y.isna().any():
                st.error("❌ Target column still has NaN values after cleaning!")
                st.stop()
            # ================= EDA =================
            st.divider()
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
            model_choice = st.radio("Select Model", ["Random Forest", "XGBoost", "GPR"])
            eval_method = st.radio("Evaluation Method", ["Train-Test Split", "K-Fold Cross-Validation"])

            if model_choice == "Random Forest":
                base_model = RandomForestRegressor(random_state=42)
            elif model_choice == "XGBoost":
                base_model = xgb.XGBRegressor(random_state=42)
            else:  # GPR
                # Kernel: constant * RBF (tunable length_scale)
                kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
                base_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
            

            final_model = None


            if eval_method == "Train-Test Split":
                test_size = st.slider("Test Size (%)", 10, 50, 25) / 100
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                st.info(f"Size of **Train set**: {X_train.shape}")
                st.info(f"Size of **Test set**: {X_test.shape}")

                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", base_model)
                ])
                pipeline.fit(X_train, y_train)

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
                final_model = pipeline


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
                    if show_feat_import and hasattr(final_model["model"], "feature_importances_"):
                        imp = final_model["model"].feature_importances_
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

            else:  # K-Fold CV
                k = st.slider("Number of Folds (K)", 2, 10, 5)
                kf = KFold(n_splits=k, shuffle=True, random_state=42)
                train_scores, val_scores = [], []
                all_train_true, all_train_pred = [], []
                all_val_true, all_val_pred = [], []

                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    pipeline = Pipeline([
                        ("scaler", StandardScaler()),
                        ("model", base_model)
                    ])
                    pipeline.fit(X_train, y_train)

                    y_train_pred = pipeline.predict(X_train)
                    y_val_pred = pipeline.predict(X_val)

                    train_scores.append(get_metrics(y_train, y_train_pred))
                    val_scores.append(get_metrics(y_val, y_val_pred))
                    all_train_true.extend(y_train)
                    all_train_pred.extend(y_train_pred)
                    all_val_true.extend(y_val)
                    all_val_pred.extend(y_val_pred)

                avg_train = np.mean(train_scores, axis=0)
                avg_val = np.mean(val_scores, axis=0)

                results = {
                    "Train (CV Avg)": avg_train,
                    "Validation (CV Avg)": avg_val
                }

                results_df = pd.DataFrame(results,
                                        index=["MAE", "MSE", "RMSE", "R²", "A20 Index"]).T
                st.subheader("Model Performance")
                st.dataframe(results_df.round(4))

                st.subheader("Predicted vs. Measured")
                st.markdown("##### 🔵 Training Set (CV Aggregated)")
                plot_predicted_vs_measured_separately(np.array(all_train_true), np.array(all_train_pred),
                                                    "Train", model_choice, selected_target)
                st.markdown("##### 🟠 Validation Set (CV Aggregated)")
                plot_predicted_vs_measured_separately(np.array(all_val_true), np.array(all_val_pred),
                                                    "Validation", model_choice, selected_target)

                # retrain full model for new dataset later
                final_model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", base_model)
                ])
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
                    X_scaled = final_model["scaler"].transform(X)
                    explainer = shap.Explainer(
                        final_model["model"],
                        X_scaled,
                        feature_names=feature_names
                    )
                    shap_values = explainer(X_scaled, check_additivity=False)

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
                        idx = st.number_input("Select sample index", 0, len(X) - 1, 0, key="force_idx")
                        st_shap = shap.plots.force(shap_values[idx], matplotlib=True, show=False)
                        st.pyplot(st_shap)

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
                        plt.close()

                    # ---------------------------
                    # 4️⃣ Dependence Plot
                    # ---------------------------
                    elif "Dependence" in shap_plot_type:
                        st.info(
                            "📈 **Dependence Plot:** Shows how a single feature’s value impacts its SHAP "
                            "contribution. Useful for detecting non-linear or interaction effects."
                        )

                        selected_feat = st.selectbox("Select feature", feature_names)
                        fig, ax = plt.subplots(figsize=(7, 5))

                        # shap.dependence_plot writes directly to the current axes
                        shap.dependence_plot(
                            selected_feat,
                            shap_values.values,
                            X,
                            feature_names=feature_names,
                            interaction_index=None,
                            ax=ax,
                            show=False
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
                        plt.close()

                except Exception as e:
                    st.warning(f"⚠️ SHAP visualization failed: {e}")


            # ================= New Dataset =================
            st.divider()
            st.header("④ Test Final Model")

            if final_model:
                test_mode = st.radio(
                    "Choose how to test the model:",
                    ["Upload CSV file", "Enter feature values manually"]
                )

                # ---------------------------
                # Option 1: Upload a file
                # ---------------------------
                if test_mode == "Upload CSV file":
                    new_data_file = st.file_uploader("Upload a dataset (CSV)", type=["csv"], key="new_dataset")
                    if new_data_file:
                        new_df = pd.read_csv(new_data_file)

                        missing_cols = [col for col in selected_features + [selected_target] if col not in new_df.columns]
                        if missing_cols:
                            st.warning(f"Missing required columns: {missing_cols}")
                        else:
                            new_X = new_df[selected_features]
                            new_y_true = new_df[selected_target]
                            new_y_pred = final_model.predict(new_X)

                            # Show metrics
                            new_metrics = get_metrics(new_y_true, new_y_pred)
                            metrics_df = pd.DataFrame(
                                [new_metrics],
                                columns=["MAE", "MSE", "RMSE", "R²", "A20 Index"],
                                index=[new_data_file.name]
                            )
                            st.subheader("Performance on New Dataset")
                            # --- Compute mean comparison metrics ---
                            mean_true = np.mean(new_y_true)
                            mean_pred = np.mean(new_y_pred)
                            mean_diff = mean_pred - mean_true
                            error_percent = (mean_diff / mean_true) * 100

                            # --- Add new columns to metrics_df ---
                            metrics_df["Error_Mean"] = error_percent
                            metrics_df["Mean_True"] = mean_true
                            metrics_df["Mean_Predicted"] = mean_pred
                            

                            # --- Display all metrics in one row ---
                            st.dataframe(metrics_df.round(4))
 
                            
                            show_signal_plot = st.checkbox("Visualize predicted vs. true signal and mean error line (for new dataset)")
                            if show_signal_plot:
                                fig, ax = plt.subplots(figsize=(8,3))
                                ax.plot(new_y_true, label="True", color="teal")
                                ax.plot(new_y_pred, label="Pred", color="orange", linestyle="--")
                                mean_err = np.mean(new_y_pred - new_y_true)
                                ax.axhline(mean_err, color="blue", linestyle="--", label=f"Mean Error = {mean_err:.3f}")
                                ax.legend(); ax.set_title("True vs Predicted Signal with Mean Error Line")
                                st.pyplot(fig); plt.close(fig)
                                
                            # Plot
                            file_label = os.path.splitext(new_data_file.name)[0]
                            st.subheader("Predicted vs. Measured (New Dataset)")
                            plot_predicted_vs_measured_separately(
                                new_y_true,
                                new_y_pred,
                                file_label,
                                model_choice,
                                selected_target
                            )

                            # --- Optional diagnostic plots for new dataset ---
                            st.markdown("### Optional Diagnostic Plots (New Dataset)")
                            with st.expander("Show / Hide Diagnostic Plots", expanded=False):
                                show_tracking = st.checkbox("1️⃣ Prediction Tracking (with Mean Lines)", key="new_tracking")
                                show_error_mean_std = st.checkbox("2️⃣ Prediction Error with Mean ± STD", key="new_err_meanstd")
                                show_residuals_pred = st.checkbox("3️⃣ Residuals vs Predicted", key="new_resid_pred")
                                show_residual_dist = st.checkbox("4️⃣ Residual Distribution (Histogram + KDE)", key="new_resid_dist")
                                show_abs_err = st.checkbox("5️⃣ Error Magnitude (Absolute Error)", key="new_abs_err")

                                # --- common residuals and errors ---
                                residuals = new_y_pred - new_y_true
                                abs_err = np.abs(residuals)
                                mean_true = np.mean(new_y_true)
                                mean_pred = np.mean(new_y_pred)
                                mean_error = np.mean(residuals)
                                std_error = np.std(residuals)

                                # ===== 1️⃣ Prediction Tracking (with Mean Lines) =====
                                if show_tracking:
                                    fig, ax = plt.subplots(figsize=(10,5))
                                    ax.plot(new_y_true, label="True Signal", color='teal', linewidth=2)
                                    ax.plot(new_y_pred, label="Predicted Signal", color='orange', linestyle='--', linewidth=2)
                                    ax.axhline(mean_true, color='teal', linestyle=':', linewidth=1.5, label=f"Mean True = {mean_true:.2f}")
                                    ax.axhline(mean_pred, color='orange', linestyle=':', linewidth=1.5, label=f"Mean Pred = {mean_pred:.2f}")
                                    ax.set_xlabel("Sample Index")
                                    ax.set_ylabel(selected_target)
                                    ax.set_title("True vs. Predicted Signal (with Mean Lines)")
                                    ax.legend()
                                    st.pyplot(fig)
                                    plt.close(fig)

                                # ===== 6️⃣ Prediction Error with Mean ± STD =====
                                if show_error_mean_std:
                                    fig, ax = plt.subplots(figsize=(10,4))
                                    ax.plot(residuals, label="Error (Pred - True)", color='red', alpha=0.6)
                                    ax.axhline(mean_error, color='blue', linestyle='--', linewidth=2, label=f"Mean Error = {mean_error:.4f}")
                                    ax.axhline(0, color='black', linestyle=':', linewidth=1)
                                    ax.fill_between(range(len(residuals)),
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

                                # ===== 2️⃣ Residuals vs Predicted =====
                                if show_residuals_pred:
                                    fig, ax = plt.subplots(figsize=(10,4))
                                    ax.scatter(new_y_pred, residuals, color="purple", alpha=0.6)
                                    ax.axhline(0, color="black", linestyle="--")
                                    ax.set_xlabel("Predicted")
                                    ax.set_ylabel("Residuals")
                                    ax.set_title("Residuals vs Predicted")
                                    st.pyplot(fig)
                                    plt.close(fig)

                                # ===== 3️⃣ Residual Distribution =====
                                if show_residual_dist:
                                    fig, ax = plt.subplots(figsize=(10,4))
                                    sns.histplot(residuals, kde=True, bins=25, color="orange", ax=ax)
                                    ax.set_title("Residual Distribution (Histogram + KDE)")
                                    st.pyplot(fig)
                                    plt.close(fig)

                                # ===== 4️⃣ Absolute Error =====
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
                else:
                    st.subheader("Enter feature values manually")
                    sample_input = {}
                    for feat in selected_features:
                        # You can adjust default values or ranges here
                        val = st.number_input(f"Enter value for {feat}", value=float(df[feat].mean()))
                        sample_input[feat] = val

                    if st.button("Predict Target"):
                        sample_df = pd.DataFrame([sample_input])
                        pred = final_model.predict(sample_df)[0]
                        st.success(f"🎯 Predicted {selected_target}: **{pred:.4f}**")
