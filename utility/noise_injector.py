import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.signal import welch
import matplotlib.pyplot as plt

# =====================================================
# 🔧 Noise Utilities
# =====================================================
def add_noise(signals, noise_type='gaussian', noise_level=0.01, seed=None):
    np.random.seed(seed)
    x = signals.copy()

    if noise_type == 'gaussian':
        std = np.std(x, axis=0)
        noise = np.random.normal(0, noise_level * std, x.shape)
        x_noisy = x + noise

    elif noise_type == 'colored':  # Pink / 1/f noise
        rows, cols = x.shape
        freqs = np.fft.rfftfreq(rows)
        freqs[0] = 1e-6
        spectrum = 1 / np.sqrt(freqs)
        white = np.random.randn(rows, cols)
        colored = np.fft.irfft(np.fft.rfft(white, axis=0) * spectrum[:, None], axis=0)
        colored /= np.std(colored, axis=0)
        x_noisy = x + noise_level * np.std(x, axis=0) * colored

    elif noise_type == 'quantization':
        bits = 12
        q_step = (np.max(x) - np.min(x)) / (2 ** bits)
        x_noisy = np.round(x / q_step) * q_step

    elif noise_type == 'spike':
        x_noisy = x.copy()
        num_spikes = int(0.001 * x.size)
        idx = np.unravel_index(np.random.choice(x.size, num_spikes, replace=False), x.shape)
        x_noisy[idx] += noise_level * np.max(x)

    elif noise_type == 'drift':
        t = np.arange(x.shape[0])
        drift = noise_level * np.sin(2 * np.pi * 0.05 * t / x.shape[0])
        x_noisy = x + drift[:, None]

    elif noise_type == 'correlated':
        base_noise = np.random.normal(0, noise_level * np.std(x.values), x.shape[0])
        x_noisy = x + base_noise[:, None]

    elif noise_type == 'bias':
        bias = noise_level * np.mean(x, axis=0)
        x_noisy = x + bias

    else:
        x_noisy = x

    return x_noisy


# =====================================================
# 📊 Diagnostic Plots
# =====================================================
def compute_snr(signal, noisy_signal):
    noise = noisy_signal - signal
    snr = 10 * np.log10(np.var(signal) / np.var(noise))
    return snr

def plot_signal_comparison(original, noisy, fs, title):
    t = np.arange(len(original)) / fs

    # --- Calculate statistics ---
    mean_orig, std_orig = np.mean(original), np.std(original)
    mean_noisy, std_noisy = np.mean(noisy), np.std(noisy)

    # ============================
    # 🟢 Full-signal view
    # ============================
    plt.figure(figsize=(9, 4))
    plt.plot(t, original, label=f"Original (μ={mean_orig:.2f}, σ={std_orig:.2f})", alpha=0.8)
    plt.plot(t, noisy, linestyle='--', 
             label=f"Noisy (μ={mean_noisy:.2f}, σ={std_noisy:.2f})", alpha=0.6)
    plt.title(f"{title} (Full Signal)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    # Optional horizontal lines for means
    plt.axhline(mean_orig, color='blue', linestyle=':', alpha=0.5)
    plt.axhline(mean_noisy, color='orange', linestyle=':', alpha=0.5)

    st.pyplot(plt.gcf())
    plt.close()

    # ============================
    # 🔍 Zoomed 10-second view
    # ============================
    duration = len(original) / fs
    zoom_end = min(1, duration)  # show only first 10s (or shorter if signal <10s)
    zoom_mask = t <= zoom_end

    plt.figure(figsize=(9, 4))
    plt.plot(t[zoom_mask], original[zoom_mask],
             label=f"Original (μ={mean_orig:.2f}, σ={std_orig:.2f})", alpha=0.8)
    plt.plot(t[zoom_mask], noisy[zoom_mask], linestyle='--',
             label=f"Noisy (μ={mean_noisy:.2f}, σ={std_noisy:.2f})", alpha=0.6)
    plt.title(f"{title} (Zoomed: First {zoom_end:.1f}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.axhline(mean_orig, color='blue', linestyle=':', alpha=0.5)
    plt.axhline(mean_noisy, color='orange', linestyle=':', alpha=0.5)

    st.pyplot(plt.gcf())
    plt.close()



def plot_psd(signal, fs, title):
    f, Pxx = welch(signal, fs, nperseg=2048)
    plt.figure(figsize=(8, 4))
    plt.semilogy(f, Pxx)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title(title)
    st.pyplot(plt.gcf())
    plt.close()

def show_noise_injecton_app():
    # ===========================================================================================================
    # 🧠 Streamlit Layout (No Sidebar)
    # ===========================================================================================================
    # ====================== File Input Section ======================
    st.markdown("### 📁 File Input Options")
 
    input_method = st.radio("Choose input method:", ["Enter source & destination folders", "Upload files"])

    files = []
    if input_method == "Upload files":
        uploaded = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)
        dst = st.text_input("💾 Destination directory to save noisy files:", "noisy_signals")  # ✅ Added this line

        if uploaded:
            for f in uploaded:
                df = pd.read_csv(f)
                files.append((f.name, df))
        os.makedirs(dst, exist_ok=True)  # ✅ Create folder also for upload case

    else:
        src = st.text_input("📂 Source directory (local or Google Drive path):")
        dst = st.text_input("💾 Destination directory (leave empty for default):", "noisy_signals")

        if src and os.path.isdir(src):
            for fname in os.listdir(src):
                if fname.endswith(".csv"):
                    files.append((fname, pd.read_csv(os.path.join(src, fname))))
        os.makedirs(dst, exist_ok=True)


    if not files:
        st.warning("Please upload or enter a valid source directory.")
        st.stop()

    # ====================== Sensor Selection ======================
    sample_df = files[0][1]
    cols = sample_df.columns.tolist()

    st.markdown("### ⚙️ Signal Configuration")
    sensors = st.multiselect("Select sensor columns:", cols)
    targets = st.multiselect("Select target columns (optional):", cols)
    fs = st.number_input("Sampling frequency (Hz):", value=1111.0, step=1.0)

    # ====================== Noise Selection ======================
    st.markdown("### 🧩 Noise Injection Setup")

    noise_options = {
        "gaussian": ("Random zero-mean fluctuations", "Electronic / ADC noise"),
        "Frequency-dependent": ("Frequency-dependent 1/f noise (colored)", "Pressure drift, tubing resonance"),
        "ADC discretization": ("ADC discretization error", "ADC resolution limits"),
        "spike": ("Random spikes or dropouts", "EMI, vibration, loose connectors"),
        #"drift": ("Slowly varying baseline offset", "Temperature effects, long-term drift"),
        #"correlated": ("Shared noise between sensors", "Shared wiring or mechanical coupling"),
        "bias": ("Offset or scaling error", "Calibration shift")
    }

    noise_type = st.selectbox(
        "Select noise type:",
        list(noise_options.keys()),
        format_func=lambda x: x.capitalize()
    )

    desc, src_desc = noise_options[noise_type]
    st.info(f"**Description:** {desc}\n\n**Typical Source:** {src_desc}")

    level = st.slider("Noise level (fraction of signal std):", 0.001, 0.2, 0.02)

    # ====================== Run Section ======================
    if st.button("🚀 Inject Noise"):
        st.write(f"Injecting **{noise_type.upper()}** noise into {len(files)} files...")

        for fname, df in files:
            x = df[sensors].copy()
            noisy = add_noise(x, noise_type, level)
            noisy_df = df.copy()
            noisy_df[sensors] = noisy
            save_path = os.path.join(dst, f"noisy_{fname}")
            noisy_df.to_csv(save_path, index=False)

        st.success(f"✅ Noise added successfully! Files saved in `{dst}`.")

        # ================= Summary & Diagnostics =================
        st.write("---")
        st.subheader("📊 Signal Diagnostic Summary (First File Example)")
        sig = files[0][1][sensors[0]].values
        noisy_sig = add_noise(sig.reshape(-1, 1), noise_type, level).flatten()

        snr_value = compute_snr(sig, noisy_sig)
        st.metric("Signal-to-Noise Ratio (SNR)", f"{snr_value:.2f} dB")

        col1, col2 = st.columns(2)
        with col1:
            plot_psd(sig, fs, "Original Signal PSD")
        with col2:
            plot_psd(noisy_sig, fs, "Noisy Signal PSD")

        plot_signal_comparison(sig, noisy_sig, fs, f"{sensors[0]}: Original vs Noisy")

        st.markdown("""
        ### 🞉 Additional Insights
        - **Power Spectral Density (Welch)**: shows how noise redistributes signal energy.
        - **SNR** quantifies degradation (lower = noisier).
        - **Low-frequency rise** in PSD → drift or 1/f noise.
        - **High-frequency boost** → Gaussian or quantization noise.
        """)

        st.markdown("""
        ### 🞉 Recommendations for Flow-Meter Signals
        | Goal | Recommended Noise | Why |
        |------|--------------------|-----|
        | Test sensor precision | Gaussian ±0.5–5 % σ | Simulates electronic noise |
        | Test temperature drift | Low-freq drift (0.05 Hz) | Mimics thermal baseline |
        | Test vibration/resonance | Add sinusoidal 5–10 Hz | Represents mechanical oscillation |
        | Test data loss | Impulse spikes or NaNs | Tests preprocessing robustness |
        | Test correlation | Same Gaussian across sensors | Emulates shared manifold noise |
        """)
