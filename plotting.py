"""
Modul visualisasi penampang geolistrik 2D menggunakan Plotly.
Tampilan menyerupai RES2DINV: filled contour + masking area di luar coverage.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from forward_model import ForwardResult, classify_material


# ─── Skala warna geolistrik (identik dengan RES2DINV style) ──────────────────

COLORSCALE_GEO = [
    [0.000, "#1A237E"],   # Biru sangat tua
    [0.080, "#1565C0"],   # Biru tua
    [0.160, "#1976D2"],   # Biru
    [0.250, "#29B6F6"],   # Biru muda
    [0.340, "#00BCD4"],   # Cyan
    [0.430, "#00897B"],   # Teal
    [0.500, "#43A047"],   # Hijau
    [0.570, "#8BC34A"],   # Hijau kuning
    [0.640, "#CDDC39"],   # Lime
    [0.700, "#FDD835"],   # Kuning
    [0.760, "#FFB300"],   # Amber
    [0.820, "#EF9F27"],   # Oranye
    [0.870, "#E64A19"],   # Oranye tua
    [0.930, "#D84315"],   # Merah bata
    [1.000, "#B71C1C"],   # Merah tua
]


def _create_smooth_contour(
    x_positions: np.ndarray,
    depths: np.ndarray,
    values: np.ndarray,
    title_text: str,
    cb_title: str = "Resistivitas (Ω·m)",
    log_scale: bool = True
) -> go.Figure:
    """
    Fungsi internal untuk melakukan interpolasi grid data dan pembuatan plot kontur tunggal.
    Memperbaiki ketidakcocokan dimensi matriks X, Y, Z pada Plotly Contour.
    """
    fig = go.Figure()

    if len(x_positions) == 0 or len(values) == 0:
        return fig

    # 1. Transformasi logaritmik jika diaktifkan
    val_plot = np.log10(np.clip(values, 0.1, None)) if log_scale else values

    # 2. Definisikan grid rapat (dense grid) secara seragam (1D Array)
    x_dense = np.linspace(x_positions.min(), x_positions.max(), 120)
    z_dense = np.linspace(depths.min(), depths.max(), 60)

    # Buat meshgrid 2D untuk kebutuhan komputasi griddata Scipy
    X_grid, Z_grid = np.meshgrid(x_dense, z_dense)

    # 3. Proses Interpolasi Linier dari poin acak ke Grid Rapat
    points = np.vstack((x_positions, depths)).T
    Z_plot = griddata(points, val_plot, (X_grid, Z_grid), method="linear")

    # 4. Pembuatan Masking Trapesium (Khas Pseudosection Geolistrik)
    x_min, x_max = x_positions.min(), x_positions.max()
    z_min, z_max = depths.min(), depths.max()
    
    slope = 0.45 * (x_max - x_min) / (z_max - z_min + 1e-6)

    for r in range(Z_plot.shape[0]):
        current_z = z_dense[r]
        dx = (current_z - z_min) * slope
        left_bound = x_min + dx
        right_bound = x_max - dx
        Z_plot[r, (x_dense < left_bound) | (x_dense > right_bound)] = np.nan

    # 5. Kembalikan nilai asli dari skala log jika log_scale digunakan
    Z_final = 10**Z_plot if log_scale else Z_plot
    
    v_min, v_max = values.min(), values.max()
    if v_min == v_max:
        v_max += 1.0

    tick_vals = np.logspace(np.log10(max(0.1, v_min)), np.log10(max(0.2, v_max)), 8) if log_scale else np.linspace(v_min, v_max, 8)
    tick_texts = [f"{v:.1f}" if v < 100 else f"{v:.0f}" for v in tick_vals]

    # 6. Menambahkan Kontur Plotly menggunakan Koordinat 1D yang Sinkron
    fig.add_trace(go.Contour(
        x=x_dense,          # Kolom matriks Z (Panjang N)
        y=z_dense,          # Baris matriks Z (Panjang M)
        z=Z_final,          # Matriks 2D Berukuran (M x N)
        colorscale=COLORSCALE_GEO,
        colorbar=dict(
            title=cb_title,
            titleside="right",
            ticks="outside",
            tickvals=tick_vals if log_scale else None,
            ticktext=tick_texts if log_scale else None,
            thickness=20,
            len=0.85
        ),
        connectgaps=False,
        hoverinfo="x+y+z",
        hovertemplate="<b>Posisi X:</b> %{x:.1f} m<br><b>Kedalaman Z:</b> %{y:.1f} m<br><b>Resistivitas:</b> %{z:.2f} Ω·m<extra></extra>",
        line=dict(width=0.3, color="rgba(0,0,0,0.15)"),
        contours=dict(coloring="heatmap", showlines=True)
    ))

    # 7. Menambahkan marker titik data asli di atas kontur
    fig.add_trace(go.Scatter(
        x=x_positions,
        y=depths,
        mode="markers",
        marker=dict(size=4, color="black", opacity=0.4),
        name="Titik Data",
        hoverinfo="skip"
    ))

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14, color="#0a2744"), x=0.01),
        xaxis=dict(title="Posisi Lintasan / Prosedur (m)", gridcolor="#f0f0f0", zeroline=False),
        yaxis=dict(title="Kedalaman (m)", autoresize=True, autorange="reverse", gridcolor="#f0f0f0", zeroline=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=80, t=50, b=50),
        height=320,
        showlegend=False
    )
    
    return fig


def plot_pseudosection(measured_datum: np.ndarray, title: str = "Measured Apparent Resistivity Pseudosection") -> go.Figure:
    """Plot penampang apparent resistivity semu (Hasil Lapangan / Input)."""
    if measured_datum is None or len(measured_datum) == 0:
        return go.Figure()
    x = measured_datum[:, 0]
    z = measured_datum[:, 1]
    rho = measured_datum[:, 2]
    return _create_smooth_contour(x, z, rho, title, "ρa (Ω·m)")


def plot_true_section(rho_matrix: np.ndarray, depths: list, title: str = "Hasil Inversi: Model True Resistivity 2D") -> go.Figure:
    """Plot penampang model true resistivity hasil kalkulasi inversi."""
    if rho_matrix is None or len(depths) == 0:
        return go.Figure()

    x_list, z_list, rho_list = [], [], []
    for ri, d in enumerate(depths):
        if ri < rho_matrix.shape[0]:
            row_vals = rho_matrix[ri]
            n_cols = len(row_vals)
            x_coords = np.arange(n_cols) * 1.0 
            for ci, val in enumerate(row_vals):
                if val > 0:
                    x_list.append(x_coords[ci])
                    z_list.append(d)
                    rho_list.append(val)

    return _create_smooth_contour(np.array(x_list), np.array(z_list), np.array(rho_list), title, "ρ True (Ω·m)")


def plot_rms_convergence(rms_history: list) -> go.Figure:
    """Plot kurva penurunan nilai RMS Error per Iterasi inversi."""
    fig = go.Figure()
    if not rms_history:
        return fig

    iters = list(range(1, len(rms_history) + 1))
    fig.add_trace(go.Scatter(
        x=iters, y=rms_history,
        mode="lines+markers",
        line=dict(color="#185FA5", width=2.5),
        marker=dict(size=8, color="#0a2744"),
        hovertemplate="Iterasi %{x}<br>RMS Error: %{y:.2f}%<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="Konvergensi Iterasi VS Abs. RMS Error", font=dict(size=12), x=0.02),
        xaxis=dict(title="Nomor Iterasi", tickmode="linear", tick0=1, dtick=1),
        yaxis=dict(title="RMS Error (%)", side="left"),
        margin=dict(l=50, r=30, t=40, b=40),
        height=240,
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="white"
    )
    return fig


def _add_contour_subplot(fig, res, row_idx, n_total, g_min, g_max, title_text, show_scale=False):
    """
    Fungsi internal pembantu baru untuk merender grafik subplot kontur komparasi secara aman 
    tanpa memicu kegagalan parameter kontur ukuran rigid (redacted fix).
    """
    x = res.datum_points[:, 0]
    z = res.datum_points[:, 1]
    rho = res.datum_points[:, 2]

    x_dense = np.linspace(x.min(), x.max(), 100)
    z_dense = np.linspace(z.min(), z.max(), 40)
    X, Z = np.meshgrid(x_dense, z_dense)
    
    Z_rho = griddata(np.vstack((x, z)).T, np.log10(np.clip(rho, 0.1, None)), (X, Z), method="linear")

    slope = 0.45 * (x.max() - x.min()) / (z.max() - z.min() + 1e-6)
    for r_idx in range(Z_rho.shape[0]):
        dz = (z_dense[r_idx] - z.min()) * slope
        Z_rho[r_idx, (x_dense < (x.min() + dz)) | (x_dense > (x.max() - dz))] = np.nan

    Z_final = 10**Z_rho
    tick_vals = np.logspace(np.log10(g_min), np.log10(g_max), 6)
    tick_texts = [f"{v:.1f}" if v < 100 else f"{v:.0f}" for v in tick_vals]

    fig.add_trace(go.Contour(
        x=x_dense, y=z_dense, z=Z_final,
        colorscale=COLORSCALE_GEO,
        zmin=g_min,
        zmax=g_max,
        showscale=show_scale,
        colorbar=dict(
            title="ρa (Ω·m)", 
            thickness=15, 
            len=1.0 / n_total,
            y=1.0 - ((row_idx - 0.5) / n_total),
            tickvals=tick_vals,
            ticktext=tick_texts
        ),
        connectgaps=False,
        line=dict(width=0.2, color="rgba(0,0,0,0.1)"),
        contours=dict(coloring="heatmap", showlines=True)
    ), row=row_idx, col=1)


def plot_comparison_spasi(results_list: list, rho_min: float = None, rho_max: float = None) -> go.Figure:
    """Komparasi visual hasil penampang berdasarkan variasi spasi elektroda dengan proteksi limit warna kustom."""
    n = len(results_list)
    if n == 0:
        return go.Figure()

    fig = make_subplots(
        rows=n, cols=1, 
        subplot_titles=[f"Spasi Elektroda: {r.electrode_spacing} m" for r in results_list], 
        vertical_spacing=max(0.06, 0.15 / n)
    )

    all_rho = np.concatenate([r.datum_points[:, 2] for r in results_list if r.datum_points is not None and len(r.datum_points) > 0])
    g_min = rho_min if rho_min is not None else float(all_rho.min())
    g_max = rho_max if rho_max is not None else float(all_rho.max())
    
    g_min = max(0.1, g_min)
    g_max = max(g_min + 1.0, g_max)

    for i, res in enumerate(results_list, 1):
        _add_contour_subplot(fig, res, i, n, g_min, g_max, f"Spasi {res.electrode_spacing} m", show_scale=(i == 1))
        fig.update_yaxes(autorange="reverse", title="Depth (m)", row=i, col=1)
        fig.update_xaxes(title="Lintasan (m)" if i == n else None, row=i, col=1)

    fig.update_layout(
        height=240 * n + 80,
        margin=dict(l=65, r=80, t=50, b=50),
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text="Komparasi Pseudosection — Variasi Spasi Elektroda", font=dict(size=13), x=0.02),
    )
    return fig


def plot_comparison_array(results_list: list, rho_min: float = None, rho_max: float = None) -> go.Figure:
    """Komparasi visual hasil penampang berdasarkan variasi tipe konfigurasi dengan proteksi limit warna kustom."""
    n = len(results_list)
    if n == 0:
        return go.Figure()

    fig = make_subplots(
        rows=n, cols=1, 
        subplot_titles=[f"Konfigurasi: {r.array_name}" for r in results_list], 
        vertical_spacing=max(0.06, 0.15 / n)
    )

    all_rho = np.concatenate([r.datum_points[:, 2] for r in results_list if r.datum_points is not None and len(r.datum_points) > 0])
    g_min = rho_min if rho_min is not None else float(all_rho.min())
    g_max = rho_max if rho_max is not None else float(all_rho.max())
    
    g_min = max(0.1, g_min)
    g_max = max(g_min + 1.0, g_max)

    for i, res in enumerate(results_list, 1):
        _add_contour_subplot(fig, res, i, n, g_min, g_max, f"Konfigurasi {res.array_name}", show_scale=(i == 1))
        fig.update_yaxes(autorange="reverse", title="Depth (m)", row=i, col=1)
        fig.update_xaxes(title="Lintasan (m)" if i == n else None, row=i, col=1)

    fig.update_layout(
        height=260 * n + 80,
        margin=dict(l=65, r=90, t=50, b=50),
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text="Komparasi Pseudosection — Variasi Konfigurasi Elektroda", font=dict(size=13), x=0.02),
    )
    return fig


def plot_layer_bar(layer_avgs: list) -> go.Figure:
    """Plot diagram batang horizontal rata-rata nilai resistivitas tiap lapisan kedalaman."""
    if not layer_avgs:
        return go.Figure()

    labels = [f"Lap. {d['layer']}<br>(z={d['depth']:.1f}m)" for d in layer_avgs]
    values_clean = [d["avg_rho"] for d in layer_avgs]
    materials = [classify_material(v)[0] for v in values_clean]
    bar_colors = [classify_material(v)[1] for v in values_clean]

    fig = go.Figure(go.Bar(
        x=values_clean, y=labels,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="white", width=0.5)),
        text=[f"{v:.1f} Ω·m" for v in values_clean],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>ρ rata-rata = %{x:.1f} Ω·m<extra></extra>",
        customdata=materials,
    ))
    
    fig.update_layout(
        title=dict(text="Rata-rata Resistivitas Model per Lapisan Kedalaman", font=dict(size=12), x=0.02),
        xaxis=dict(title="Resistivitas Rata-rata (Ω·m)", type="log" if max(values_clean)/max(1.0, min(values_clean)) > 10 else "linear"),
        yaxis=dict(autorange="reverse"),
        margin=dict(l=90, r=50, t=40, b=40),
        height=max(180, 35 * len(layer_avgs)),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white"
    )
    return fig
