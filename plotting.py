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
    Fungsi internal untuk melakukan interpolasi grid data dan pembuatan plot kontur.
    Memperbaiki ketidakcocokan dimensi matriks X, Y, Z pada Plotly Contour.
    """
    fig = go.Figure()

    if len(x_positions) == 0 or len(values) == 0:
        return fig

    # 1. Transformasi logaritmik jika diaktifkan (agar kontur resistivitas lebih proporsional)
    val_plot = np.log10(values) if log_scale else values

    # 2. Definisikan grid rapat (dense grid) secara seragam (1D Array)
    x_dense = np.linspace(x_positions.min(), x_positions.max(), 120)
    z_dense = np.linspace(depths.min(), depths.max(), 60)

    # Buat meshgrid 2D untuk kebutuhan komputasi griddata Scipy
    X_grid, Z_grid = np.meshgrid(x_dense, z_dense)

    # 3. Proses Interpolasi Linier dari poin acak ke Grid Rapat
    points = np.vstack((x_positions, depths)).T
    Z_plot = griddata(points, val_plot, (X_grid, Z_grid), method="linear")

    # 4. Pembuatan Masking Trapesium (Khas Pseudosection Geolistrik)
    # Memotong area luar coverage data agar berbentuk trapesium terbalik seperti RES2DINV
    x_min, x_max = x_positions.min(), x_positions.max()
    z_min, z_max = depths.min(), depths.max()
    
    # Faktor kemiringan sudut pembatas pseudosection
    slope = 0.45 * (x_max - x_min) / (z_max - z_min + 1e-6)

    for r in range(Z_plot.shape[0]):
        current_z = z_dense[r]
        # Hitung batas horizontal kiri-kanan pada kedalaman z tertentu
        dx = (current_z - z_min) * slope
        left_bound = x_min + dx
        right_bound = x_max - dx
        
        # Masking nilai di luar batas trapesium menjadi NaN
        Z_plot[r, (x_dense < left_bound) | (x_dense > right_bound)] = np.nan

    # 5. Kembalikan nilai asli dari skala log jika log_scale digunakan
    Z_final = 10**Z_plot if log_scale else Z_plot
    tick_vals = np.logspace(np.log10(values.min()), np.log10(values.max()), 8) if log_scale else np.linspace(values.min(), values.max(), 8)
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
        hovertemplate="<b>Posis X:</b> %{x:.1f} m<br><b>Kedalaman Z:</b> %{y:.1f} m<br><b>Resistivitas:</b> %{z:.2f} Ω·m<extra></extra>",
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

    # Layouting Agar Sumbu Y Terbalik (Kedalaman ke bawah)
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
    # Ekstraksi koordinat grid dari bentuk matriks baris-kolom menjadi susunan array 1D
    for ri, d in enumerate(depths):
        if ri < rho_matrix.shape[0]:
            row_vals = rho_matrix[ri]
            n_cols = len(row_vals)
            # Posisikan penampang koordinat x di tengah blok grid
            x_coords = np.arange(n_cols) * (1.0) 
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


def plot_comparison_spasi(results_list: list) -> go.Figure:
    """Komparasi visual hasil pseudosection berdasarkan variasi spasi elektroda."""
    n = len(results_list)
    if n == 0:
        return go.Figure()

    fig = make_subplots(rows=n, cols=1, subplot_titles=[f"Spasi Elektroda: {r.electrode_spacing} m" for r in results_list], vertical_spacing=0.12)

    for i, res in enumerate(results_list, 1):
        x = res.datum_points[:, 0]
        z = res.datum_points[:, 1]
        rho = res.datum_points[:, 2]

        x_dense = np.linspace(x.min(), x.max(), 100)
        z_dense = np.linspace(z.min(), z.max(), 40)
        X, Z = np.meshgrid(x_dense, z_dense)
        Z_rho = griddata(np.vstack((x, z)).T, np.log10(rho), (X, Z), method="linear")

        # Masking trapesium area data kosong
        slope = 0.45 * (x.max() - x.min()) / (z.max() - z.min() + 1e-6)
        for r_idx in range(Z_rho.shape[0]):
            dz = (z_dense[r_idx] - z.min()) * slope
            Z_rho[r_idx, (x_dense < (x.min() + dz)) | (x_dense > (x.max() - dz))] = np.nan

        Z_rho = 10**Z_rho

        fig.add_trace(go.Contour(
            x=x_dense, y=z_dense, z=Z_rho,
            colorscale=COLORSCALE_GEO,
            showscale=True if i == 1 else False,
            colorbar=dict(title="ρa (Ω·m)", thickness=15, len=0.9),
            connectgaps=False,
            line=dict(width=0.2, color="rgba(0,0,0,0.1)")
        ), row=i, col=1)
        
        fig.update_yaxes(autorange="reverse", title="Depth (m)", row=i, col=1)
        fig.update_xaxes(title="Lintasan (m)" if i == n else None, row=i, col=1)

    fig.update_layout(
        height=260 * n,
        margin=dict(l=65, r=80, t=50, b=50),
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text="Komparasi Pseudosection — Variasi Spasi Elektroda", font=dict(size=13), x=0.02),
    )
    return fig


def plot_comparison_array(results_list: list) -> go.Figure:
    """Komparasi visual hasil penampang berdasarkan variasi tipe konfigurasi."""
    n = len(results_list)
    if n == 0:
        return go.Figure()

    fig = make_subplots(rows=n, cols=1, subplot_titles=[f"Konfigurasi: {r.array_name}" for r in results_list], vertical_spacing=0.12)

    for i, res in enumerate(results_list, 1):
        x = res.datum_points[:, 0]
        z = res.datum_points[:, 1]
        rho = res.datum_points[:, 2]

        x_dense = np.linspace(x.min(), x.max(), 100)
        z_dense = np.linspace(z.min(), z.max(), 40)
        X, Z = np.meshgrid(x_dense, z_dense)
        Z_rho = griddata(np.vstack((x, z)).T, np.log10(rho), (X, Z), method="linear")

        slope = 0.45 * (x.max() - x.min()) / (z.max() - z.min() + 1e-6)
        for r_idx in range(Z_rho.shape[0]):
            dz = (z_dense[r_idx] - z.min()) * slope
            Z_rho[r_idx, (x_dense < (x.min() + dz)) | (x_dense > (x.max() - dz))] = np.nan

        Z_rho = 10**Z_rho

        fig.add_trace(go.Contour(
            x=x_dense, y=z_dense, z=Z_rho,
            colorscale=COLORSCALE_GEO,
            showscale=True if i == 1 else False,
            colorbar=dict(title="ρa (Ω·m)", thickness=15, len=0.9),
            connectgaps=False,
            line=dict(width=0.2, color="rgba(0,0,0,0.1)")
        ), row=i, col=1)
        
        fig.update_yaxes(autorange="reverse", title="Depth (m)", row=i, col=1)
        fig.update_xaxes(title="Lintasan (m)" if i == n else None, row=i, col=1)

    fig.update_layout(
        height=290 * n,
        margin=dict(l=65, r=90, t=50, b=50),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11),
        title=dict(text="Komparasi Pseudosection — Variasi Konfigurasi Elektroda", font=dict(size=13), x=0.02),
    )
    return fig


def plot_layer_bar(layer_avgs: list) -> go.Figure:
    """Plot diagram batang horizontal rata-rata nilai resistivitas tiap lapisan kedalaman."""
    if not layer_avgs:
        return go.Figure()

    labels = [f"Lap. {d['layer']}<br>(z={d['depth']:.1f}m)" for d in layer_avgs]
    values = [d["avg_rho"]] for d in layer_avgs
    
    # Alternatif ekstraksi jika struktur data list of dict biasa
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
        xaxis=dict(title="Resistivitas Rata-rata (Ω·m)", type="log" if max(values_clean)/min(values_clean) > 10 else "linear"),
        yaxis=dict(autorange="reverse"),
        margin=dict(l=90, r=50, t=40, b=40),
        height=max(180, 35 * len(layer_avgs)),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white"
    )
    return fig
