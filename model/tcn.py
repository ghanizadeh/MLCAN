# model/tcn.py

import os
import io
import gc
import time
import random
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

# Silence TF logs before importing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, LayerNormalization, ReLU, Add, Dropout, Dense
    from tensorflow.keras.optimizers import Adam
except Exception as e:
    tf = None

# ===============================
# Globals / Config
# ===============================
REQUIRED_COLS = ["Timestamp","Sensor1","Sensor2","Sensor3","Sensor4","Sensor5","Sensor6","CRL","AliCat"]
SENSOR_COLS = ["Sensor1","Sensor2","Sensor3","Sensor4","Sensor5","Sensor6"]

# ===============================
# Utilities
# ===============================
def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)

def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="float")
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df

def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    # avoid division by zero; ddof=0 matches paper-style normalization
    return (df - df.mean()) / df.std(ddof=0).replace(0, np.nan)

def _first_difference(df: pd.DataFrame) -> pd.DataFrame:
    return df.diff().dropna()

def _iter_source_files(folder_path: str) -> List[str]:
    return sorted([str(p) for p in Path(folder_path).glob("*.csv")])

def _gpu_status_msg():
    if tf is None:
        return "❌ TensorFlow not installed"
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        names = [getattr(g, 'name', 'GPU') for g in gpus]
        return f"✅ GPU detected: {', '.join(names)}"
    return "ℹ️ Using CPU (no GPU detected)"

# ===============================
# Robust CSV loader (chunk-safe)
# ===============================
def _safe_read_csv(
    src: Union[str, os.PathLike, io.BytesIO],
    usecols: List[str],
    chunksize: int = 200_000
) -> pd.DataFrame:
    """
    Safe CSV reader with:
      - usecols (reduces memory)
      - on_bad_lines='skip' (skips malformed rows)
      - chunked fallback (reduces peak memory, avoids tokenizing OOM)
    """
    # Fast path
    try:
        df = pd.read_csv(
            src,
            usecols=[c for c in usecols if isinstance(c, str)],
            low_memory=True,
            on_bad_lines="skip",
            engine="c"
        )
        return df
    except Exception:
        # Chunked fallback
        pieces = []
        try:
            for chunk in pd.read_csv(
                src,
                usecols=[c for c in usecols if isinstance(c, str)],
                low_memory=True,
                on_bad_lines="skip",
                engine="c",
                chunksize=chunksize
            ):
                pieces.append(chunk)
            if not pieces:
                return pd.DataFrame(columns=usecols)
            return pd.concat(pieces, ignore_index=True)
        except Exception:
            # Last resort: python engine
            pieces = []
            for chunk in pd.read_csv(
                src,
                usecols=[c for c in usecols if isinstance(c, str)],
                low_memory=True,
                on_bad_lines="skip",
                engine="python",
                chunksize=chunksize
            ):
                pieces.append(chunk)
            if not pieces:
                return pd.DataFrame(columns=usecols)
            return pd.concat(pieces, ignore_index=True)

def _load_one_csv(src) -> pd.DataFrame:
    df = _safe_read_csv(src, usecols=REQUIRED_COLS)
    # retain available required cols
    available = [c for c in REQUIRED_COLS if c in df.columns]
    if len([c for c in SENSOR_COLS if c in available]) < 3:
        # too few sensors — skip
        raise ValueError(f"Insufficient sensor columns in file. Found: {available}")

    df = df[available].copy()

    # Timestamp ordering if present
    if "Timestamp" in df.columns:
        with pd.option_context("mode.chained_assignment", None):
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        if df["Timestamp"].notna().any():
            df = df.sort_values("Timestamp").reset_index(drop=True)

    # Ensure all expected sensor cols exist; pad missing with zeros (rare)
    for s in SENSOR_COLS:
        if s not in df.columns:
            df[s] = 0.0

    return _downcast_numeric(df)

# ===============================
# Sequence maker
# ===============================
def _make_sequences(
    sensors_df: pd.DataFrame,
    target_series: pd.Series,
    seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(sensors_df)
    if n <= seq_len:
        return np.empty((0, seq_len, sensors_df.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # Align target to sensors length
    target_series = target_series.iloc[-n:].reset_index(drop=True)

    X_list, y_list = [], []
    for i in range(n - seq_len):
        X_list.append(sensors_df.iloc[i:i+seq_len].values)
        y_list.append(target_series.iloc[i+seq_len])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y

def _build_dataset_from_sources(
    sources: List[Union[str, os.PathLike, io.BytesIO]],
    seq_len: int,
    preproc_mode: str,
    target_col: str,
    progress: st.delta_generator.DeltaGenerator = None
) -> Tuple[np.ndarray, np.ndarray]:
    X_all, y_all = [], []
    total = len(sources)
    for idx, src in enumerate(sources, 1):
        try:
            df = _load_one_csv(src)
            sensors = df[SENSOR_COLS].astype("float32")
            # target column
            if target_col not in df.columns:
                raise ValueError(f"Target '{target_col}' not found in file.")
            tgt = df[target_col].astype("float32")

            # Preprocess
            if preproc_mode == "Z-Score":
                sensors_pp = _zscore(sensors)
                sensors_pp = sensors_pp.replace([np.inf, -np.inf], np.nan).dropna()
                # re-align target length
                tgt_pp = tgt.iloc[len(tgt) - len(sensors_pp):].reset_index(drop=True)
            else:
                sensors_pp = _first_difference(sensors)
                tgt_pp = tgt.iloc[len(tgt) - len(sensors_pp):].reset_index(drop=True)

            if len(sensors_pp) < (seq_len + 1):
                continue

            X, y = _make_sequences(sensors_pp, tgt_pp, seq_len)
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)

        except Exception as e:
            if progress:
                progress.write(f"⚠️ Skipped a file due to error: {e}")
        finally:
            if progress:
                progress.progress(idx / max(total, 1))

    if not X_all:
        return np.empty((0, seq_len, 6), dtype=np.float32), np.empty((0,), dtype=np.float32)
    X_cat = np.concatenate(X_all, axis=0)
    y_cat = np.concatenate(y_all, axis=0)
    return X_cat, y_cat

# ===============================
# TCN model
# ===============================
def _temporal_block(x, n_filters, kernel_size=2, dilation_rate=1, dropout=0.2):
    res = x
    x = Conv1D(n_filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout)(x)

    x = Conv1D(n_filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout)(x)

    if res.shape[-1] != n_filters:
        res = Conv1D(n_filters, 1, padding='same')(res)
    x = Add()([x, res])
    return x

def _build_tcn(input_shape, lr=1e-3, n_filters=256, dropout=0.2):
    inp = Input(shape=input_shape)
    x = inp
    for d in [1,2,4,8,16,32,64,128]:
        x = _temporal_block(x, n_filters=n_filters, kernel_size=2, dilation_rate=d, dropout=dropout)
    x = Dense(128, activation='relu')(x)
    out = Dense(1)(x)  # sequence output; we'll use last step
    model = Model(inp, out)
    model.compile(optimizer=Adam(lr), loss="mse")
    return model

def _expand_y_to_seq(y_scalar: np.ndarray, timesteps: int) -> np.ndarray:
    return np.repeat(y_scalar[:, None], timesteps, axis=1)[..., None].astype(np.float32)

def _split_train_val_test(X, y, val_ratio=0.1, test_ratio=0.1, seed=42):
    _set_seed(seed)
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    X = X[idx]; y = y[idx]
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    X_test, y_test = X[:n_test], y[:n_test]
    X_val,  y_val  = X[n_test:n_test+n_val], y[n_test:n_test+n_val]
    X_train, y_train = X[n_test+n_val:], y[n_test+n_val:]
    return X_train, y_train, X_val, y_val, X_test, y_test

def _metrics(y_true, y_pred):
    mse = float(np.mean((y_true - y_pred)**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rho = float(np.corrcoef(y_true, y_pred)[0,1]) if len(y_true) > 1 else float("nan")
    return dict(MSE=mse, RMSE=rmse, MAE=mae, R=rho)

# ===============================
# Plots
# ===============================
def _plot_learning_curve(history, title="Loss vs Epochs"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(history.history.get("loss", []), label="Train")
    ax.plot(history.history.get("val_loss", []), label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.legend()
    return fig

def _scatter_plot(y_true, y_pred, title="Predicted vs True"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, s=10, alpha=0.6)
    # 45-degree line
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    ax.plot([mn,mx],[mn,mx], linestyle='--')
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    return fig

def _residual_plot(y_true, y_pred, title="Residuals"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, s=10, alpha=0.6)
    ax.axhline(0, linestyle='--')
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted - True")
    ax.set_title(title)
    return fig

# ===============================
# Train wrapper (one target+prep)
# ===============================
def _train_one(
    X, y, epochs, batch_size, lr, n_filters, dropout, seed, progress_cb=None
):
    if len(X) == 0:
        return None

    X_train, y_train, X_val, y_val, X_test, y_test = _split_train_val_test(X, y, 0.1, 0.1, seed)
    model = _build_tcn((X.shape[1], X.shape[2]), lr=lr, n_filters=n_filters, dropout=dropout)

    y_train_seq = _expand_y_to_seq(y_train, X_train.shape[1])
    y_val_seq   = _expand_y_to_seq(y_val,   X_val.shape[1])

    callbacks = []
    if progress_cb and tf is not None:
        class _Prog(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_cb.progress((epoch+1)/epochs)
        callbacks.append(_Prog())

    history = model.fit(
        X_train, y_train_seq,
        validation_data=(X_val, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks
    )

    # Predict (last step)
    y_pred_train = model.predict(X_train, verbose=0)[:, -1, 0]
    y_pred_val   = model.predict(X_val,   verbose=0)[:, -1, 0]
    y_pred_test  = model.predict(X_test,  verbose=0)[:, -1, 0]

    return {
        "model": model,
        "history": history,
        "splits": {
            "train": (y_train, y_pred_train),
            "val":   (y_val,   y_pred_val),
            "test":  (y_test,  y_pred_test),
        }
    }

# ===============================
# MAIN PAGE (compact layout)
# ===============================
def show_TCN_model_page():
    st.subheader("③ TCN — Multiphase Flowrate Prediction (CRL & AliCat) · Compact")

    # GPU info (as requested: "show")
    st.info(_gpu_status_msg())

    if tf is None:
        st.error("TensorFlow is not installed. Please install `tensorflow` to use this page.")
        return

    _set_seed(42)

    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        - Inputs: **Sensor1–Sensor6** (time-series)
        - Targets: **CRL (liquid)** and **AliCat (gas)**
        - Preprocessing:
            - **Single mode** → choose **First-Difference** *or* **Z-Score**
            - **Benchmark mode** → trains **both FD & Z-Score** for the selected target(s)
        - Window: default **128** timesteps (editable)
        - Split: 80% train / 10% val / 10% test
        - Metrics: MSE, RMSE, MAE, Pearson **R**
        - Safe loader: **chunked reading**, **usecols**, **on_bad_lines='skip'**, **float32 downcast**
        """)

    # --------------------------
    # Data input options
    # --------------------------
    st.markdown("### 1) Data Source")
    input_mode = st.radio("Choose input method:", ["Folder path", "Upload files"], horizontal=True)
    sources = []
    if input_mode == "Folder path":
        folder = st.text_input("📂 Folder containing CSV files:", value="")
        if folder and os.path.isdir(folder):
            files = _iter_source_files(folder)
            st.info(f"Found **{len(files)}** CSV files.")
            sources = files
        elif folder:
            st.warning("Folder not found.")
    else:
        uploads = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)
        if uploads:
            st.info(f"Uploaded **{len(uploads)}** files.")
            sources = uploads

    # --------------------------
    # Config (Targets + Preproc)
    # --------------------------
    st.markdown("### 2) Configuration")
    colA, colB, colC = st.columns(3)
    with colA:
        target_choice = st.selectbox("Train on target(s)", ["CRL", "AliCat", "Both"], index=2)
    with colB:
        mode = st.selectbox("Preprocessing Mode", ["Single (choose one)", "Benchmark (FD vs Z-Score)"], index=1)
    with colC:
        chosen_preproc = st.selectbox("If Single mode → Preprocessing", ["First-Difference", "Z-Score"], index=0, disabled=(mode.startswith("Benchmark")))

    colD, colE, colF = st.columns(3)
    with colD:
        seq_len = st.number_input("Sequence length (timesteps)", min_value=16, max_value=2048, value=128, step=16)
    with colE:
        batch_size = st.number_input("Batch size", min_value=8, max_value=512, value=32, step=8)
    with colF:
        epochs = st.number_input("Epochs", min_value=5, max_value=200, value=30, step=5)

    colG, colH, colI = st.columns(3)
    with colG:
        lr = st.number_input("Learning rate", min_value=1e-5, max_value=1e-2, value=1e-3, step=1e-5, format="%.5f")
    with colH:
        n_filters = st.selectbox("TCN filters", [64, 128, 256, 384, 512], index=2)
    with colI:
        dropout = st.slider("Dropout", min_value=0.0, max_value=0.8, value=0.2, step=0.05)

    if st.button("⏵ Build & Train"):
        if not sources:
            st.error("Please provide CSV files via a folder or upload.")
            return

        start = time.time()
        seq_len = int(seq_len); batch_size = int(batch_size); epochs = int(epochs); n_filters = int(n_filters)

        st.markdown("### 3) Building Datasets")
        pb = st.progress(0.0)
        log = st.empty()

        selected_targets = []
        if target_choice in ("CRL", "Both"): selected_targets.append("CRL")
        if target_choice in ("AliCat", "Both"): selected_targets.append("AliCat")

        # Determine preprocessing variants
        if mode.startswith("Benchmark"):
            prep_variants = ["First-Difference", "Z-Score"]
        else:
            prep_variants = [chosen_preproc]

        datasets = {}  # dict[(target, preproc)] = (X, y)
        for t in selected_targets:
            for prep in prep_variants:
                log.write(f"🔧 Preparing dataset for **{t} — {prep}** …")
                pb.progress(0.0)
                X, y = _build_dataset_from_sources(
                    sources=sources,
                    seq_len=seq_len,
                    preproc_mode=prep,
                    target_col=t,
                    progress=pb
                )
                datasets[(t, prep)] = (X, y)
                st.info(f"{t} — {prep}: X={X.shape}, y={y.shape}")

        st.markdown("### 4) Training")
        results = {}  # dict[(target, preproc)] = result_dict
        # Compact layout: each (target, prep) in its own expander
        for t in selected_targets:
            for prep in prep_variants:
                X, y = datasets[(t, prep)]
                with st.expander(f"📘 {t} — {prep}", expanded=True):
                    if len(X) == 0:
                        st.warning("No sequences created. Skipping.")
                        continue
                    pb_train = st.progress(0.0)
                    res = _train_one(
                        X, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=float(lr),
                        n_filters=n_filters,
                        dropout=dropout,
                        seed=42,
                        progress_cb=pb_train
                    )
                    results[(t, prep)] = res
                    if res is None:
                        st.error("Training failed or insufficient data.")
                        continue

                    # Learning curves
                    st.pyplot(_plot_learning_curve(res["history"], title=f"{t} — {prep}: Loss vs Epochs"))

                    # Metrics + Plots
                    y_tr, yptr = res["splits"]["train"]
                    y_va, ypva = res["splits"]["val"]
                    y_te, ypte = res["splits"]["test"]

                    m_tr = _metrics(y_tr, yptr)
                    m_va = _metrics(y_va, ypva)
                    m_te = _metrics(y_te, ypte)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Train RMSE", f"{m_tr['RMSE']:.4f}")
                    c2.metric("Val RMSE", f"{m_va['RMSE']:.4f}")
                    c3.metric("Test RMSE", f"{m_te['RMSE']:.4f}")
                    c1.metric("Train MAE", f"{m_tr['MAE']:.4f}")
                    c2.metric("Val MAE", f"{m_va['MAE']:.4f}")
                    c3.metric("Test MAE", f"{m_te['MAE']:.4f}")
                    c1.metric("Train R", f"{m_tr['R']:.4f}")
                    c2.metric("Val R", f"{m_va['R']:.4f}")
                    c3.metric("Test R", f"{m_te['R']:.4f}")

                    st.pyplot(_scatter_plot(y_te, ypte, title=f"{t} — {prep}: Predicted vs True (Test)"))
                    st.pyplot(_residual_plot(y_te, ypte, title=f"{t} — {prep}: Residuals (Test)"))

                    # Downloads
                    pred_df = pd.DataFrame({"y_true": y_te.astype(float), "y_pred": ypte.astype(float)})
                    st.download_button(
                        f"⬇️ Download Test Predictions — {t} ({prep})",
                        data=pred_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"tcn_{t.lower()}_{prep.replace(' ','').lower()}_test_predictions.csv",
                        mime="text/csv"
                    )

                    # Optional: Save model
                    with st.expander("💾 Save Model (optional)"):
                        save_dir = st.text_input("Directory to save this model (leave blank to skip):", value="")
                        if st.button(f"Save {t} — {prep}"):
                            if save_dir.strip():
                                Path(save_dir).mkdir(parents=True, exist_ok=True)
                                res["model"].save(os.path.join(save_dir, f"tcn_{t.lower()}_{prep.replace(' ','').lower()}.keras"))
                                st.success(f"Saved to: {save_dir}")
                            else:
                                st.info("No directory specified. Skipped saving.")

        st.success(f"✅ Finished in {time.time()-start:.1f}s")
        gc.collect()
