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

    # ================================
    # Utility Functions
    # ================================
    def plot_predicted_vs_measured_separately(y_true, y_pred, dataset_type, model_name, target):
        color = 'teal' if "Train" in dataset_type else 'orange'

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        plt.figure(figsize=(7, 7))
        max_val = np.max(np.concatenate([y_true, y_pred]))
        min_val = 0

        plt.scatter(y_true, y_pred, c=color, edgecolors='black',
                    marker='o' if "Train" in dataset_type else 'v', alpha=0.7)

        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='100% agreement')
        plt.plot([min_val, max_val], [min_val, 1.2 * max_val], 'r--', linewidth=1, label='+20%')
        plt.plot([min_val, max_val], [min_val, 0.8 * max_val], 'r--', linewidth=1, label='-20%')

        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.xlabel("Measured " + target)
        plt.ylabel("Predicted " + target)
        plt.title(f"{model_name} - {dataset_type} Set")

        plt.legend(title=f"{dataset_type}:\nR²={r2:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
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
            ax_main.set_xlabel(col, fontsize=11)
            ax_main.set_ylabel(target_col, fontsize=11)

            ax_xhist.hist(x, bins=15, color='green', edgecolor='black')
    
            ax_yhist.hist(y, bins=15, orientation='horizontal',
                        color='green', edgecolor='black')
    

            plots.append(fig)

        return plots

    # ================================
    # Streamlit App
    # ================================
    #st.title("MLCan Simple AI")

    # -------------------------------
    # Page UI
    # -------------------------------
    uploaded_file = st.file_uploader("Upload your file (csv)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.header("1. Select Features and Target")
        with st.expander("Column Selection"):
            cols = df.columns.tolist()
            selected_features = st.multiselect("Select Feature Columns", options=cols)
            selected_target = st.selectbox("Select Target Column",
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
            st.info(f"🧹 Data cleaned: Original size = {original_size}, Cleaned size = {cleaned_size}")

            # Update X and y AFTER cleaning
            X = df_clean[selected_features]
            y = df_clean[selected_target]

            if y.isna().any():
                st.error("❌ Target column still has NaN values after cleaning!")
                st.stop()
            # ================= EDA =================
            st.header("2. Exploratory Data Analysis")
            with st.expander("Show/hide EDA", expanded=True):
                st.subheader("2.1 Summary Statistics")
                summary = df[selected_features + [selected_target]].describe().T
                summary['skew'] = df[selected_features + [selected_target]].skew()
                summary['kurtosis'] = df[selected_features + [selected_target]].kurtosis()
                st.dataframe(summary.round(3))

                st.subheader("2.2 Scatter Plots + Histograms")
                plots = plot_pairwise_corr_with_hist(df[selected_features + [selected_target]], selected_target)
                for fig in plots:
                    st.pyplot(fig)

                st.subheader("2.3 Correlation Heatmap")
                corr = df[selected_features + [selected_target]].corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                            cmap="coolwarm", cbar=True, ax=ax,
                            square=True, linewidths=0.5,
                            annot_kws={"size": 9})
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="left")
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                st.pyplot(fig)

            # ================= Model =================
            st.header("3. Model Training and Evaluation")
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

            # ================= Interpretation =================
            st.header("4. Model Interpretation")
            show_shap = st.checkbox("4.1 Show SHAP Plots")
            if show_shap and final_model:
                with st.expander("SHAP Waterfall + Beeswarm Plots"):
                    try:
                        # Keep track of original feature names
                        feature_names = selected_features

                        # Transform X using scaler
                        X_scaled = final_model["scaler"].transform(X)

                        # Create explainer with feature names
                        explainer = shap.Explainer(
                            final_model["model"],
                            X_scaled,
                            feature_names=feature_names
                        )
                        shap_values = explainer(X_scaled, check_additivity=False)

                        # Beeswarm plot with correct feature names
                        fig = plt.figure()
                        shap.plots.beeswarm(shap_values, show=False)
                        st.pyplot(fig)

                        # Optional: Waterfall plot for the first sample
                        fig = plt.figure()
                        shap.plots.waterfall(shap_values[0], show=False)
                        st.pyplot(fig)

                    except Exception as e:
                        st.warning(f"SHAP failed: {e}")


            # ================= New Dataset =================
            st.header("5. Test Final Model")

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
                            st.dataframe(metrics_df.round(4))
                            st.text("Predicted Target:")
                            st.dataframe(new_y_pred)

                            # Plot
                            import os
                            file_label = os.path.splitext(new_data_file.name)[0]
                            st.subheader("Predicted vs. Measured (New Dataset)")
                            plot_predicted_vs_measured_separately(
                                new_y_true,
                                new_y_pred,
                                file_label,
                                model_choice,
                                selected_target
                            )
                    
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







