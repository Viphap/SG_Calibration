import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.signal import savgol_filter
import ruptures as rpt
import torch
import torch.nn.functional as F
import h5py
import io
import zipfile
import os

st.set_page_config(page_title="Calib App", layout="wide")
st.title("Calibration App — SG Data")

# ─── PyTorch U-Net 1D (mirrors Keras architecture) ──────────────────────────
class UNet1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        p = 15 // 2
        self.enc1       = torch.nn.Conv1d(1,   32,  15, padding=p)
        self.enc2       = torch.nn.Conv1d(32,  64,  15, padding=p)
        self.bottleneck = torch.nn.Conv1d(64,  128, 15, padding=p)
        self.dec1       = torch.nn.Conv1d(192, 64,  15, padding=p)
        self.dec2       = torch.nn.Conv1d(96,  32,  15, padding=p)
        self.out        = torch.nn.Conv1d(32,  1,   1)

    def forward(self, x):
        s1 = F.relu(self.enc1(x))
        x  = F.max_pool1d(s1, 2)
        s2 = F.relu(self.enc2(x))
        x  = F.max_pool1d(s2, 2)
        x  = F.relu(self.bottleneck(x))
        x  = F.interpolate(x, scale_factor=2, mode='nearest')
        x  = torch.cat([x, s2], dim=1)
        x  = F.relu(self.dec1(x))
        x  = F.interpolate(x, scale_factor=2, mode='nearest')
        x  = torch.cat([x, s1], dim=1)
        x  = F.relu(self.dec2(x))
        return self.out(x)


def _keras_w(k):
    return torch.tensor(k.transpose(2, 1, 0), dtype=torch.float32)

def _load_conv(f, key, layer):
    layer.weight.data = _keras_w(f[f"layers/{key}/vars/0"][:])
    layer.bias.data   = torch.tensor(f[f"layers/{key}/vars/1"][:], dtype=torch.float32)

@st.cache_resource(show_spinner="Đang tải model…")
def load_model():
    model_path = os.path.join(os.path.dirname(__file__) or ".", "model", "calib_unet_model.keras")
    net = UNet1D()
    with zipfile.ZipFile(model_path) as z:
        with h5py.File(io.BytesIO(z.read("model.weights.h5")), "r") as f:
            _load_conv(f, "conv1d",   net.enc1)
            _load_conv(f, "conv1d_1", net.enc2)
            _load_conv(f, "conv1d_2", net.bottleneck)
            _load_conv(f, "conv1d_3", net.dec1)
            _load_conv(f, "conv1d_4", net.dec2)
            _load_conv(f, "conv1d_5", net.out)
    net.eval()
    return net


# ─── Step 1: auto_calib_patch_predict (giống notebook) ───────────────────────
@torch.no_grad()
def auto_calib_patch_predict(series, model, patch_size=512, stride=256, batch_size=128):
    # Lấy signal, NaN → interpolate (giống notebook: pd.to_numeric().interpolate())
    if isinstance(series, pd.Series):
        signal = pd.to_numeric(series, errors='coerce').interpolate().values.flatten().astype(np.float32)
    else:
        signal = np.array(series, dtype=np.float32)
        if np.isnan(signal).any():
            idx    = np.arange(len(signal))
            mask   = ~np.isnan(signal)
            signal = np.interp(idx, idx[mask], signal[mask]).astype(np.float32)

    n = len(signal)

    # Tách Global Trend
    x = np.arange(n, dtype=np.float64)
    slope, intercept, *_ = stats.linregress(x, signal.astype(np.float64))
    trend     = (slope * x + intercept).astype(np.float32)
    detrended = (signal - trend).astype(np.float32)

    # Tạo patches
    starts = list(np.arange(0, n - patch_size + 1, stride))
    if not starts or starts[-1] + patch_size < n:
        starts.append(int(n - patch_size))
    starts = np.array(starts)

    patches = np.stack([detrended[s : s + patch_size] for s in starts])
    t       = torch.tensor(patches[:, None, :], dtype=torch.float32)  # (N,1,512)

    # Batch predict
    outs = []
    for i in range(0, len(t), batch_size):
        outs.append(model(t[i : i + batch_size]).squeeze(1).numpy())
    preds = np.concatenate(outs, axis=0)  # (N, patch_size)

    # Reconstruction
    reconstructed = np.zeros(n, dtype=np.float64)
    weights       = np.zeros(n, dtype=np.float64)
    for i, s in enumerate(starts):
        e = s + patch_size
        reconstructed[s:e] += preds[i]
        weights[s:e]       += 1.0
    reconstructed /= np.maximum(weights, 1.0)

    # Cộng trend lại + Savitzky-Golay
    calibrated = (reconstructed + trend).astype(np.float32)
    calibrated = savgol_filter(calibrated, window_length=101, polyorder=2).astype(np.float32)
    return calibrated


# ─── Step 2: auto_calib_with_pelt ────────────────────────────────────────────
def auto_calib_with_pelt(calib_ai, pelt_penalty=2000, pre_smooth=1001):
    # Làm mượt mạnh trước PELT để loại bỏ dao động ngày đêm,
    # PELT chỉ còn thấy các bước nhảy lớn (thay đổi tải trọng thực sự)
    win = pre_smooth if pre_smooth % 2 == 1 else pre_smooth + 1
    win = min(win, len(calib_ai) - 1 if len(calib_ai) % 2 == 0 else len(calib_ai))
    smoothed = savgol_filter(calib_ai, window_length=win, polyorder=2)

    algo   = rpt.Pelt(model="l2").fit(smoothed)
    result = algo.predict(pen=pelt_penalty)

    # Piecewise constant trên signal gốc (calib_ai), không phải smoothed
    pelt_signal = np.zeros_like(calib_ai)
    start_idx   = 0
    for end_idx in result:
        pelt_signal[start_idx:end_idx] = np.mean(calib_ai[start_idx:end_idx])
        start_idx = end_idx

    return pelt_signal.astype(np.float32)


# ─── Load CSV ────────────────────────────────────────────────────────────────
_base = os.path.dirname(__file__) or "."
DEFAULT_CSV = next(
    (p for p in [
        os.path.join(_base, "PA_T38_SG.csv"),
        os.path.join(_base, "PA_T38_SG (1).csv"),
    ] if os.path.exists(p)),
    os.path.join(_base, "PA_T38_SG.csv"),
)

with st.sidebar:
    st.header("Dữ liệu")
    uploaded    = st.file_uploader("Chọn file CSV", type="csv")
    use_default = uploaded is None and os.path.exists(DEFAULT_CSV)
    if use_default:
        st.info(f"Dùng file mặc định:\n`{os.path.basename(DEFAULT_CSV)}`")


@st.cache_data(show_spinner="Đang đọc CSV…")
def read_csv(path_or_buf, _key):
    # TOA5 format: row0=metadata, row1=colnames, row2-3=units
    df = pd.read_csv(
        path_or_buf,
        low_memory=False,
        header=1,
        skiprows=[2, 3],
        na_values=["NAN", "nan", "NAN\"", ""],
    )
    df.columns = df.columns.str.strip()
    return df


if uploaded is not None:
    df = read_csv(uploaded, uploaded.name)
elif use_default:
    df = read_csv(DEFAULT_CSV, DEFAULT_CSV)
else:
    st.warning("Vui lòng upload file CSV hoặc đặt `PA_T38_SG.csv` vào thư mục app.")
    st.stop()

# ─── Sidebar controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Cột dữ liệu")
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != "RECORD"]
    selected_col = st.selectbox("Chọn cột để hiển thị", numeric_cols)

    st.header("Tùy chọn")
    do_calib  = st.checkbox("Calibration", value=False)
    do_export = st.checkbox("Export (thêm cột _calib vào CSV)", value=False)

    with st.expander("Nâng cao"):
        patch_size    = st.number_input("Patch size",    value=512, step=64,  min_value=64)
        stride        = st.number_input("Stride",        value=256, step=64,  min_value=16)
        batch_size    = st.number_input("Batch size",    value=128, step=8,   min_value=1)
        pelt_penalty  = st.number_input("PELT penalty",  value=2000, step=500, min_value=1,
                                        help="Cao → ít đoạn bậc thang (chỉ giữ biến động lớn); thấp → nhiều đoạn hơn")
        pre_smooth    = st.number_input("Pre-smooth window", value=1001, step=200, min_value=51,
                                        help="Window Savgol làm mượt trước PELT — lớn hơn = loại bỏ dao động ngày đêm tốt hơn")

# ─── Time axis ───────────────────────────────────────────────────────────────
time_col = None
for c in ["TIMESTAMP", "Timestamp", "timestamp", "Time", "DATE", "Date"]:
    if c in df.columns:
        try:
            df[c] = pd.to_datetime(df[c])
            time_col = c
        except Exception:
            pass
        break

x_vals = df[time_col] if time_col else df.index

# ─── Run calibration ─────────────────────────────────────────────────────────
calib_col_name = f"{selected_col}_calib"
calib_vals     = None

if do_calib or do_export:
    model = load_model()
    with st.spinner("Bước 1/2 — AI patch predict…"):
        calib_ai = auto_calib_patch_predict(
            df[selected_col], model,
            patch_size=int(patch_size),
            stride=int(stride),
            batch_size=int(batch_size),
        )
    with st.spinner("Bước 2/2 — PELT changepoint smoothing…"):
        calib_vals = auto_calib_with_pelt(calib_ai,
                                          pelt_penalty=float(pelt_penalty),
                                          pre_smooth=int(pre_smooth))

# ─── Plot ────────────────────────────────────────────────────────────────────
fig = go.Figure()

if do_calib and calib_vals is not None:
    fig.add_trace(go.Scatter(
        x=x_vals, y=df[selected_col],
        mode="lines",
        name=f"Raw Input ({selected_col})",
        line=dict(color="rgba(100,149,237,0.35)", width=0.8),
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=calib_vals,
        mode="lines",
        name="AI Calibrated",
        line=dict(color="red", width=2.2),
    ))
    title_text = "Comparison of Raw Data and AI Calibrated Data"
else:
    fig.add_trace(go.Scatter(
        x=x_vals, y=df[selected_col],
        mode="lines",
        name=f"Raw Input ({selected_col})",
        line=dict(color="rgba(100,149,237,0.7)", width=1),
    ))
    title_text = f"Raw Data — {selected_col}"

fig.update_layout(
    title=dict(text=title_text, font=dict(size=16), x=0.5, xanchor="center"),
    xaxis=dict(
        title="Timestamp",
        showgrid=True, gridcolor="rgba(200,200,200,0.4)",
        linecolor="lightgrey",
        tickformat="%Y-%m",
    ),
    yaxis=dict(
        title="Value",
        showgrid=True, gridcolor="rgba(200,200,200,0.4)",
        linecolor="lightgrey",
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
    height=520,
    legend=dict(
        orientation="v", x=0.99, xanchor="right", y=0.99, yanchor="top",
        bgcolor="rgba(255,255,255,0.8)", bordercolor="lightgrey", borderwidth=1,
    ),
    margin=dict(l=60, r=30, t=70, b=60),
)
st.plotly_chart(fig, use_container_width=True)

# ─── Export ──────────────────────────────────────────────────────────────────
if do_export:
    if calib_vals is None:
        st.error("Bật checkbox **Calibration** trước khi export.")
    else:
        df_out = df.copy()
        df_out[calib_col_name] = calib_vals

        if uploaded is not None:
            out_name  = uploaded.name.replace(".csv", "_calibrated.csv")
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"⬇ Tải về {out_name}",
                data=csv_bytes, file_name=out_name, mime="text/csv",
            )
        else:
            out_path = DEFAULT_CSV.replace(".csv", "_calibrated.csv")
            df_out.to_csv(out_path, index=False)
            st.success(f"Đã lưu: `{out_path}` (thêm cột `{calib_col_name}`)")

# ─── Quick stats ─────────────────────────────────────────────────────────────
with st.expander("Thống kê nhanh"):
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**{selected_col} (raw)**")
        st.dataframe(df[selected_col].describe().to_frame(), use_container_width=True)
    if calib_vals is not None:
        with c2:
            st.write(f"**{calib_col_name}**")
            st.dataframe(
                pd.Series(calib_vals, name=calib_col_name).describe().to_frame(),
                use_container_width=True,
            )
