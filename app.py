import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
import tifffile

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReflectCalib",
    page_icon="🛰️",
    layout="wide",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Top bar title */
h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    color: #58a6ff !important;
}

h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    color: #c9d1d9 !important;
}

/* Step label */
.step-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #58a6ff;
    letter-spacing: 2px;
    margin-bottom: 4px;
    text-transform: uppercase;
}

/* Card */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}

/* Metric highlight */
.metric-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #3fb950;
}
.metric-label {
    font-size: 0.72rem;
    color: #8b949e;
    margin-top: 2px;
}

/* Status badge */
.badge-ok {
    display: inline-block;
    background: #1a3a27;
    color: #3fb950;
    border: 1px solid #3fb950;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
}
.badge-warn {
    display: inline-block;
    background: #3a2e1a;
    color: #d29922;
    border: 1px solid #d29922;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
}

/* Divider */
hr { border-color: #21262d !important; }

/* Upload area */
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 1.5px dashed #30363d;
    border-radius: 10px;
    padding: 0.5rem;
}

/* Buttons */
.stButton > button {
    background: #1f6feb;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    padding: 0.45rem 1.2rem;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #388bfd;
}

/* Slider */
[data-testid="stSlider"] { accent-color: #58a6ff; }

/* Info box */
.info-block {
    background: #132030;
    border-left: 3px solid #58a6ff;
    border-radius: 0 6px 6px 0;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #8b949e;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Helper functions ──────────────────────────────────────────────────────────

def load_image(uploaded_file):
    """Load JPEG or TIFF image, return numpy array (float32, 0-1 range)."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith((".tif", ".tiff")):
        arr = tifffile.imread(io.BytesIO(raw)).astype(np.float32)
        # Normalize to 0-1 if not already
        if arr.max() > 1.0:
            arr = arr / arr.max()
        return arr, "tiff"
    else:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.array(pil).astype(np.float32) / 255.0
        return arr, "jpeg"


def detect_gray_panel(img_float):
    """
    Auto-detect the gray panel region using color clustering.
    Returns (x, y, w, h) bounding rect and a mask.
    Strategy: find region with low saturation + medium brightness.
    """
    img_uint8 = (img_float * 255).astype(np.uint8)
    if img_uint8.ndim == 2:
        # Grayscale TIFF
        gray = img_uint8
        # Low variance region
        blurred = cv2.GaussianBlur(gray, (31, 31), 0)
        diff = cv2.absdiff(gray, blurred)
        _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY_INV)
    else:
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # Low saturation (gray-ish) + mid brightness
        mask = ((s < 40) & (v > 80) & (v < 220)).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, w, h), mask


def extract_mean_dn(img_float, x, y, w, h):
    """Extract mean DN value of ROI. Returns per-channel means."""
    roi = img_float[y:y+h, x:x+w]
    if roi.ndim == 2:
        return np.array([roi.mean()])
    else:
        return roi.reshape(-1, roi.shape[-1]).mean(axis=0)


def compute_correction_factors(mean_dn, reflectance_value):
    """Factor = known_reflectance / measured_DN"""
    mean_dn = np.clip(mean_dn, 1e-6, None)
    return reflectance_value / mean_dn


def apply_correction(img_float, factors):
    """Multiply each channel by its factor, clip to 0-1."""
    corrected = img_float.copy()
    if corrected.ndim == 2:
        corrected = corrected * factors[0]
    else:
        for c in range(min(len(factors), corrected.shape[-1])):
            corrected[:, :, c] = corrected[:, :, c] * factors[c]
    return np.clip(corrected, 0.0, 1.0)


def to_display_rgb(arr):
    """Convert float array to uint8 RGB for display."""
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] >= 3:
        rgb = arr[:, :, :3]
    else:
        rgb = arr
    return (rgb * 255).astype(np.uint8)


def draw_roi_on_image(img_uint8, x, y, w, h):
    """Draw green bounding box on image."""
    out = img_uint8.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 100), 3)
    cv2.putText(out, "Panel ROI", (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
    return out


def array_to_download_bytes(arr, fmt="jpeg"):
    """Convert corrected array to downloadable bytes."""
    rgb = to_display_rgb(arr)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    if fmt == "tiff":
        pil.save(buf, format="TIFF")
    else:
        pil.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ─── UI ────────────────────────────────────────────────────────────────────────

st.markdown("# 🛰️ ReflectCalib")
st.markdown('<p style="color:#8b949e; margin-top:-0.8rem; font-size:0.9rem;">Drone image reflectance calibration — gray panel method</p>', unsafe_allow_html=True)
st.markdown("---")

# ── STEP 1 ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="step-label">Step 1 — Reference Panel</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

col_info, col_upload = st.columns([1, 1])

with col_info:
    st.markdown('<div class="info-block">Upload foto close-up kertas abu-abu (gray card / calibrated reflectance panel). Pastikan panel terlihat jelas dan tidak terkena bayangan.</div>', unsafe_allow_html=True)
    reflectance_val = st.slider(
        "Nilai reflektansi panel (dari lab / datasheet)",
        min_value=0.05, max_value=1.0, value=0.18, step=0.01,
        format="%.2f",
        help="Default: 0.18 (18% gray card standar)"
    )
    st.markdown(f'<div class="metric-box"><div class="metric-value">{reflectance_val:.0%}</div><div class="metric-label">Panel Reflectance</div></div>', unsafe_allow_html=True)

with col_upload:
    panel_file = st.file_uploader("Upload foto panel abu-abu", type=["jpg", "jpeg", "tif", "tiff"], key="panel")

st.markdown('</div>', unsafe_allow_html=True)

# ── STEP 2 ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="step-label">Step 2 — Drone Survey Image</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="info-block">Upload citra lahan hasil pemotretan drone. Bisa berupa orthophoto JPEG atau GeoTIFF.</div>', unsafe_allow_html=True)
scene_file = st.file_uploader("Upload citra lahan drone", type=["jpg", "jpeg", "tif", "tiff"], key="scene")

st.markdown('</div>', unsafe_allow_html=True)

# ── PROCESS ────────────────────────────────────────────────────────────────────
if panel_file and scene_file:
    st.markdown("---")
    st.markdown('<div class="step-label">Step 3 — Processing</div>', unsafe_allow_html=True)

    with st.spinner("Memuat gambar..."):
        panel_arr, panel_type = load_image(panel_file)
        scene_arr, scene_type = load_image(scene_file)

    # ── Auto-detect panel ROI ──
    st.markdown("#### Deteksi Area Panel")

    bbox, mask = detect_gray_panel(panel_arr)
    panel_display = to_display_rgb(panel_arr)

    auto_detected = bbox is not None
    if auto_detected:
        x0, y0, w0, h0 = bbox
        area_pct = (w0 * h0) / (panel_arr.shape[0] * panel_arr.shape[1]) * 100
        status_badge = f'<span class="badge-ok">✓ Panel terdeteksi ({area_pct:.1f}% frame)</span>'
    else:
        x0, y0 = 0, 0
        w0, h0 = panel_arr.shape[1] // 2, panel_arr.shape[0] // 2
        status_badge = '<span class="badge-warn">⚠ Tidak terdeteksi otomatis — atur manual</span>'

    st.markdown(status_badge, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Manual adjust ──
    with st.expander("🔧 Adjust ROI manual (opsional)", expanded=not auto_detected):
        h_img, w_img = panel_arr.shape[:2]
        col1, col2 = st.columns(2)
        with col1:
            x0 = st.slider("X start", 0, w_img - 10, x0)
            w0 = st.slider("Lebar ROI", 10, w_img - x0, min(w0, w_img - x0))
        with col2:
            y0 = st.slider("Y start", 0, h_img - 10, y0)
            h0 = st.slider("Tinggi ROI", 10, h_img - y0, min(h0, h_img - y0))

    # Preview panel with ROI
    preview_panel = draw_roi_on_image(panel_display, x0, y0, w0, h0)
    st.image(preview_panel, caption="Preview deteksi panel (kotak hijau = area yang diukur)", use_container_width=True)

    # ── Compute ──
    mean_dn = extract_mean_dn(panel_arr, x0, y0, w0, h0)
    factors = compute_correction_factors(mean_dn, reflectance_val)
    corrected = apply_correction(scene_arr, factors)

    # ── Results ──
    st.markdown("---")
    st.markdown('<div class="step-label">Step 4 — Results</div>', unsafe_allow_html=True)

    # Metrics
    n_ch = len(mean_dn)
    ch_names = ["R", "G", "B", "NIR", "RE"] if n_ch > 1 else ["Gray"]
    cols = st.columns(n_ch + 1)

    for i, col in enumerate(cols[:-1]):
        with col:
            st.markdown(f'''
            <div class="metric-box">
                <div class="metric-value">{factors[i]:.4f}×</div>
                <div class="metric-label">Factor {ch_names[i] if i < len(ch_names) else f"Ch{i}"}</div>
            </div>''', unsafe_allow_html=True)

    with cols[-1]:
        mean_factor = factors.mean()
        st.markdown(f'''
        <div class="metric-box">
            <div class="metric-value" style="color:#58a6ff">{mean_factor:.4f}×</div>
            <div class="metric-label">Mean Factor</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Before / After
    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown("**Sebelum koreksi** (DN mentah)")
        st.image(to_display_rgb(scene_arr), use_container_width=True)

    with col_after:
        st.markdown("**Sesudah koreksi** (Surface Reflectance)")
        st.image(to_display_rgb(corrected), use_container_width=True)

    # Histogram comparison
    with st.expander("📊 Lihat distribusi nilai piksel"):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3), facecolor="#0d1117")
        for ax, img, title in [(axes[0], scene_arr, "Sebelum"), (axes[1], corrected, "Sesudah")]:
            ax.set_facecolor("#161b22")
            colors = ["#f44747", "#3fb950", "#58a6ff"] if img.ndim == 3 else ["#c9d1d9"]
            data = img[:, :, :len(colors)] if img.ndim == 3 else img[:, :, np.newaxis]
            for c, color in enumerate(colors):
                ax.hist(data[:, :, c].flatten(), bins=100, color=color, alpha=0.7, density=True)
            ax.set_title(title, color="#c9d1d9", fontsize=10)
            ax.tick_params(colors="#8b949e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
        plt.tight_layout()
        st.pyplot(fig)

    # ── Download ──
    st.markdown("---")
    st.markdown('<div class="step-label">Step 5 — Download</div>', unsafe_allow_html=True)

    out_fmt = "tiff" if scene_type == "tiff" else "jpeg"
    out_bytes = array_to_download_bytes(corrected, fmt=out_fmt)
    out_name = scene_file.name.replace(".", "_calibrated.")

    st.download_button(
        label=f"⬇ Download citra terkoreksi ({out_fmt.upper()})",
        data=out_bytes,
        file_name=out_name,
        mime="image/tiff" if out_fmt == "tiff" else "image/jpeg",
    )

    st.markdown('<p style="color:#8b949e; font-size:0.78rem; margin-top:0.5rem;">Nilai piksel output = Surface Reflectance (skala 0–1 dipetakan ke 0–255 untuk preview/JPEG). Untuk analisis NDVI lebih lanjut, gunakan file TIFF dan baca nilai float aslinya.</p>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #8b949e;">
        <div style="font-size:2.5rem; margin-bottom:1rem;">🛰️</div>
        <div style="font-size:1rem;">Upload foto panel abu-abu dan citra lahan drone untuk memulai kalibrasi.</div>
    </div>
    """, unsafe_allow_html=True)
