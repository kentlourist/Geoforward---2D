"""
Parser untuk file hasil inversi RES2DINV (.INV)
Mendukung format output RES2DINV versi 3.x - 10.x

Struktur file .INV RES2DINV:
  Baris 1        : Nama lintasan
  Baris 2        : Spasi elektroda minimum (m)
  Baris 3        : Kode tipe array (1=Wenner, 3=DD, 7=WS, dsb.)
  Baris 4        : Jumlah datum
  Baris 5        : Flag (biasanya 1)
  Baris 6        : Flag topografi (0=flat)
  Baris 7..N     : Data apparent resistivity terukur
                   format: x_mid, a, n, rho_a
  ---
  INVERSION RESULTS
  Initial RMS error / INITIAL RMS ERROR
  <nilai>
  NUMBER OF LAYERS
  <n_layer>
  NUMBER OF BLOCKS
  <n_blok>

  [Diulang tiap iterasi:]
  ITERATION <k>
  MODEL RESISTIVITY
  LAYER <l>
  <n_blok_layer>,  <kedalaman_layer>    <- depth bisa kosong di layer terakhir
      <x_pos>,  <rho_true>
      ...
                    <rho_sisi_kanan>    <- indent >= 10 spasi, hanya 1 angka
  CALCULATED APPARENT RESISTIVITY
      x_mid, a, n, rho_calc
      ...
   <rms_persen>                         <- satu angka setelah data calc

  [akhir file:]
  Iteration  Time ...  Total Time   % Abs. Error  /  Abs. Error
       1        ...       ...          29.512
"""

import numpy as np
import re


# ─── Konstanta & mapping ───────────────────────────────────────────────────────

ARRAY_NAMES = {
    1: "Wenner",
    2: "Pole-Pole",
    3: "Dipole-Dipole",
    4: "Wenner-Schlumberger",
    5: "Pole-Dipole",
    6: "Dipole-Dipole (Azimuthal)",
    7: "Wenner-Schlumberger",
    8: "Gradient",
    11: "Pole-Pole (non-conventional)",
}


def _array_name(code: int) -> str:
    return ARRAY_NAMES.get(code, f"Array-{code}")


def get_array_name(code: int) -> str:
    return _array_name(code)


# ─── Pseudo-depth & faktor geometri ───────────────────────────────────────────

def _pseudo_depth(array_code: int, a: float, n: float) -> float:
    """Pseudo-depth dari spasi a dan level n."""
    if array_code == 1:          # Wenner
        return 0.500 * n * a
    elif array_code == 3:        # Dipole-Dipole
        return 0.300 * n * a
    else:                        # Wenner-Schlumberger (default)
        return 0.519 * n * a


def geometric_factor(array_code: int, a: float, n: float) -> float:
    """
    Hitung faktor geometri K untuk setiap konfigurasi elektroda.

    Wenner-Schlumberger (kode 7):
        Susunan: A --- M --- N --- B
        AM = na, MN = a, NB = na  →  AB = (2n+1)a
        K = π · n · (n+1) · a

    Wenner (kode 1):
        AM = MN = NB = a  →  K = 2π · a

    Dipole-Dipole (kode 3):
        AM = a, MN = na, NB = a (dipole separation)
        K = π · n · (n+1) · (n+2) · a

    Returns K dalam satuan meter (a harus dalam meter).
    """
    if array_code == 1:          # Wenner
        return 2.0 * np.pi * a
    elif array_code == 3:        # Dipole-Dipole
        return np.pi * n * (n + 1) * (n + 2) * a
    else:                        # Wenner-Schlumberger (default, kode 7 & 4)
        return np.pi * n * (n + 1) * a


def compute_apparent_resistivity(rho_true: float, K: float) -> float:
    """
    ρa = K × (ΔV / I)
    Dalam konteks forward modelling sederhana 1D:
    ρa ≈ K × rho_true   (untuk model homogen)
    Untuk model berlapis, perhitungan ini dilakukan via fungsi
    compute_apparent_from_model() di bawah.
    """
    return K * rho_true


def compute_apparent_from_model(
    rho_matrix: np.ndarray,
    x_positions: list,
    depths: list,
    array_code: int,
    electrode_spacing: float,
    n_levels: int = None,
) -> np.ndarray:
    """
    Hitung apparent resistivity dari model true resistivity 2D
    menggunakan pendekatan weighted average (kernel depth weighting).

    Metode: untuk setiap titik datum (x_mid, n_level), integrasikan
    kontribusi resistivitas tiap blok model dengan bobot kernel
    berdasarkan kedalaman. Ini adalah pendekatan semi-analitik
    yang umum dipakai untuk estimasi forward dari model 2D.

    Parameters
    ----------
    rho_matrix  : np.ndarray (n_layers, n_cols)  — resistivitas true (Ω·m)
    x_positions : list of float  — posisi kolom model (m)
    depths      : list of float  — kedalaman tiap layer (m)
    array_code  : int  — kode array (1=Wenner, 3=DD, 7=WS)
    electrode_spacing : float  — spasi elektroda dasar a (m)
    n_levels    : int  — jumlah level n max (default: auto dari data)

    Returns
    -------
    datum_points : np.ndarray (N, 3) — [x_mid, pseudo_depth, rho_apparent]
    """
    if rho_matrix is None or len(x_positions) == 0 or len(depths) == 0:
        return np.array([]).reshape(0, 3)

    a = electrode_spacing
    x_arr = np.array(x_positions, dtype=float)
    z_arr = np.array(depths, dtype=float)
    n_layers, n_cols = rho_matrix.shape

    # Tentukan n_max dari geometri survei
    x_span = x_arr.max() - x_arr.min()
    if n_levels is None:
        if array_code == 1:       # Wenner: n_max = (n_elec - 1) / 2
            n_max = max(1, int(x_span / a) // 2)
        elif array_code == 3:     # Dipole-Dipole
            n_max = max(1, int(x_span / a) - 2)
        else:                     # Wenner-Schlumberger
            n_max = max(1, int(x_span / (2 * a)) - 1)
    else:
        n_max = n_levels

    datum_points = []

    for n_val in range(1, n_max + 1):
        # Rentang x_mid yang valid untuk level n ini
        if array_code == 1:       # Wenner: AM=MN=NB=a → AB=3a → x_mid margin = 1.5a
            x_margin = (n_val + 0.5) * a
        elif array_code == 3:     # Dipole-Dipole: margin = (n+1)*a
            x_margin = (n_val + 1.0) * a
        else:                     # Wenner-Schlumberger: margin = n*a + 0.5a
            x_margin = (n_val + 0.5) * a

        x_min_valid = x_arr.min() + x_margin
        x_max_valid = x_arr.max() - x_margin

        if x_min_valid > x_max_valid:
            break

        # Titik-titik x_mid yang valid
        x_mids = x_arr[(x_arr >= x_min_valid) & (x_arr <= x_max_valid)]

        K = geometric_factor(array_code, a, float(n_val))
        z_pseudo = _pseudo_depth(array_code, a, float(n_val))

        for x_mid in x_mids:
            rho_a = _kernel_weighted_rho(
                rho_matrix, x_arr, z_arr,
                x_mid, z_pseudo, a, n_val, array_code
            )
            if rho_a is not None and rho_a > 0:
                datum_points.append([x_mid, z_pseudo, rho_a])

    if not datum_points:
        return np.array([]).reshape(0, 3)
    return np.array(datum_points)


def _kernel_weighted_rho(
    rho_matrix, x_arr, z_arr,
    x_mid, z_pseudo, a, n_val, array_code
):
    """
    Hitung ρa di titik (x_mid, n_val) dengan kernel depth weighting.

    Kernel sensitivity Wenner-Schlumberger (Barker 1989):
        w(z, x) ∝ 1 / [1 + (2z / L)²]^(3/2)
    di mana L = setengah panjang bentangan = (n+0.5)*a untuk WS.

    Untuk model 2D, bobot juga mempertimbangkan jarak lateral
    antara x_mid dan posisi kolom model.
    """
    n_layers, n_cols = rho_matrix.shape

    # Panjang karakteristik bentangan
    if array_code == 1:
        L = a                          # Wenner: L = a
    elif array_code == 3:
        L = (n_val + 1) * a            # Dipole-Dipole
    else:
        L = (n_val + 0.5) * a          # Wenner-Schlumberger

    total_weight = 0.0
    weighted_rho = 0.0

    for li in range(n_layers):
        z = z_arr[li]

        # Kernel kedalaman (Wenner-Schlumberger sensitivity kernel)
        depth_weight = 1.0 / (1.0 + (2.0 * z / L) ** 2) ** 1.5

        for ci in range(n_cols):
            rho = rho_matrix[li, ci]
            if np.isnan(rho) or rho <= 0:
                continue

            x_col = x_arr[ci]
            dx = abs(x_col - x_mid)

            # Bobot lateral: Gaussian dengan lebar L
            lateral_weight = np.exp(-0.5 * (dx / L) ** 2)

            w = depth_weight * lateral_weight
            weighted_rho += w * rho
            total_weight += w

    if total_weight <= 0:
        return None
    return weighted_rho / total_weight


# ─── Helper umum ──────────────────────────────────────────────────────────────

def _parse_floats(s: str) -> list:
    """Ambil semua angka float dari string."""
    return [float(x) for x in re.findall(r"-?[\d]+(?:\.[\d]+)?(?:[eE][+-]?\d+)?", s)]


def _is_section_header(line: str) -> bool:
    """True jika baris adalah penanda section baru."""
    ln = line.strip().upper()
    patterns = [
        r"^LAYER\s+\d+",
        r"^ITERATION\s+\d+",
        r"^CALCULATED APPARENT",
        r"^MODEL RESISTIVITY",
        r"^REFERENCE MODEL",
        r"^INVERSION RESULTS",
        r"^ERROR DISTRIBUTION",
        r"^MEAN ERROR",
        r"^AVERAGE ABSOLUTE",
        r"^STANDARD DEVIATION",
        r"^RES2DINV",
        r"^WINDOWS",
        r"^KEY ID",
        r"^ITERATION\s+TIME",
        r"^BLOCKS SENSITIVITY",
        r"^AVERAGE SENSITIVITY",
        r"^MODEL INVERSION",
        r"^DATA INVERSION",
        r"^INITIAL DAMPING",
        r"^MINIMUM DAMPING",
        r"^VERTICAL TO HORIZONTAL",
        r"^FINITE-ELEMENT",
        r"^APPARENT RESISTIVITY VALUES",
        r"^LIMIT IMPOSED",
        r"^RATIO OF MAXIMUM",
    ]
    return any(re.match(p, ln) for p in patterns)


# ─── Parser utama ──────────────────────────────────────────────────────────────

def parse_inv_file(content: str) -> dict:
    """
    Parse file .INV dari RES2DINV.
    Mengembalikan dict lengkap berisi metadata, matriks resistivitas,
    datum terukur, datum terhitung, dan apparent resistivity dari model.
    """
    lines = [l.rstrip() for l in content.replace("\r\n", "\n").replace("\r", "\n").splitlines()]

    result = {
        "success": False,
        "error": None,
        "metadata": {},
        "survey_name": "",
        "electrode_spacing": 1.0,
        "array_type": 7,
        "array_name": "Wenner-Schlumberger",
        "n_datum": 0,
        "n_electrodes": 0,
        # Measured apparent resistivity (dari data input file)
        "measured_datum": None,       # np.array (N,3): [x_mid, pseudo_depth, rho_a]
        # Calculated apparent resistivity (dari section CALCULATED APPARENT di file)
        "calculated_datum": None,     # np.array (N,3): [x_mid, pseudo_depth, rho_a]
        # Apparent resistivity dihitung ulang dari model true resistivity
        "forward_datum": None,        # np.array (N,3): [x_mid, pseudo_depth, rho_a]
        # Model true resistivity
        "rho_matrix": None,           # np.array (n_layers, n_cols)
        "depths": [],
        "x_positions": [],
        "n_rows": 0,
        "n_cols": 0,
        "rms_history": [],
        "final_rms": None,
        "iteration_used": 0,
        "n_iterations": 0,
        "initial_rms": None,
        "iterations": [],
        "layer_averages": [],
    }

    try:
        n_lines = len(lines)
        idx = 0

        # ── Baris 1–6: header ───────────────────────────────────────────────
        result["survey_name"] = lines[idx].strip(); idx += 1

        m = re.search(r"([\d.]+)", lines[idx])
        if m: result["electrode_spacing"] = float(m.group(1))
        idx += 1

        m = re.search(r"(\d+)", lines[idx])
        if m: result["array_type"] = int(m.group(1))
        result["array_name"] = _array_name(result["array_type"])
        idx += 1

        m = re.search(r"(\d+)", lines[idx])
        if m: result["n_datum"] = int(m.group(1))
        idx += 1

        idx += 2  # lewati 2 baris flag

        # ── Data apparent resistivity terukur ───────────────────────────────
        measured = []
        while idx < n_lines:
            ln = lines[idx].strip()
            if "INVERSION RESULTS" in ln.upper():
                break
            vals = _parse_floats(ln)
            if len(vals) >= 4:
                x_mid, a_val, n_val, rho_a = vals[0], vals[1], vals[2], vals[3]
                z = _pseudo_depth(result["array_type"], a_val, n_val)
                measured.append([x_mid, z, rho_a])
            idx += 1

        if measured:
            result["measured_datum"] = np.array(measured)
            x_arr = result["measured_datum"][:, 0]
            a = result["electrode_spacing"]
            result["n_electrodes"] = int(round((x_arr.max() - x_arr.min()) / a)) + 3

        # ── Cari dan lewati "INVERSION RESULTS" ────────────────────────────
        while idx < n_lines and "INVERSION RESULTS" not in lines[idx].upper():
            idx += 1
        idx += 1

        # ── Initial RMS ─────────────────────────────────────────────────────
        while idx < n_lines and "INITIAL RMS" not in lines[idx].upper():
            idx += 1
        idx += 1
        if idx < n_lines:
            m = re.search(r"([\d.]+)", lines[idx])
            if m: result["initial_rms"] = float(m.group(1))
            idx += 1

        # ── NUMBER OF LAYERS ────────────────────────────────────────────────
        while idx < n_lines and "NUMBER OF LAYERS" not in lines[idx].upper():
            idx += 1
        idx += 1
        if idx < n_lines:
            m = re.search(r"(\d+)", lines[idx])
            idx += 1

        # ── NUMBER OF BLOCKS ────────────────────────────────────────────────
        while idx < n_lines and "NUMBER OF BLOCKS" not in lines[idx].upper():
            idx += 1
        idx += 2  # lewati header + nilai

        # ── Baca semua ITERATION ────────────────────────────────────────────
        iterations = []

        while idx < n_lines:
            ln = lines[idx].strip()

            iter_match = re.match(r"ITERATION\s+(\d+)", ln, re.IGNORECASE)
            if not iter_match:
                idx += 1
                continue

            iter_num = int(iter_match.group(1))
            idx += 1

            # Lewati ke "MODEL RESISTIVITY"
            while idx < n_lines and "MODEL RESISTIVITY" not in lines[idx].upper():
                if re.match(r"ITERATION\s+\d+", lines[idx].strip(), re.IGNORECASE):
                    break
                idx += 1
            if idx >= n_lines:
                break
            idx += 1

            # ── Baca semua LAYER ────────────────────────────────────────────
            layers = {}
            layer_depths = {}

            while idx < n_lines:
                ln2 = lines[idx].strip()

                if "CALCULATED APPARENT" in ln2.upper():
                    break
                if re.match(r"ITERATION\s+\d+", ln2, re.IGNORECASE):
                    break
                if "REFERENCE MODEL" in ln2.upper():
                    break

                layer_match = re.match(r"LAYER\s+(\d+)", ln2, re.IGNORECASE)
                if not layer_match:
                    idx += 1
                    continue

                layer_idx = int(layer_match.group(1))
                idx += 1

                # Header layer: "n_blok, depth" atau "n_blok,"
                layer_depth = None
                if idx < n_lines:
                    depth_vals = _parse_floats(lines[idx].strip())
                    if len(depth_vals) >= 2:
                        layer_depth = depth_vals[1]
                    # depth_vals == 1: hanya n_blok, depth tidak tersedia
                    idx += 1

                # Baca blok data layer
                layer_data = []
                while idx < n_lines:
                    raw_ln = lines[idx]
                    stripped = raw_ln.strip()

                    if _is_section_header(stripped):
                        break

                    vals = _parse_floats(stripped)

                    if len(vals) == 0:
                        idx += 1
                        continue
                    elif len(vals) >= 2:
                        # x_pos, rho_true
                        layer_data.append((vals[0], vals[1]))
                        idx += 1
                    elif len(vals) == 1:
                        # Cek apakah ini boundary sisi kanan (indent >= 10 spasi)
                        leading = len(raw_ln) - len(raw_ln.lstrip())
                        if leading >= 10:
                            idx += 1  # nilai boundary — lewati
                        else:
                            break
                    else:
                        idx += 1

                layers[layer_idx] = layer_data
                layer_depths[layer_idx] = layer_depth

            # ── CALCULATED APPARENT RESISTIVITY ─────────────────────────────
            calc = []
            rms_val = None

            if idx < n_lines and "CALCULATED APPARENT" in lines[idx].upper():
                idx += 1
                while idx < n_lines:
                    raw_ln = lines[idx]
                    stripped = raw_ln.strip()

                    # Berhenti di section header baru (kecuali dirinya sendiri)
                    if _is_section_header(stripped) and "CALCULATED" not in stripped.upper():
                        break

                    vals = _parse_floats(stripped)

                    if len(vals) >= 4:
                        x_mid, a_val, n_val, rho_c = vals[0], vals[1], vals[2], vals[3]
                        z = _pseudo_depth(result["array_type"], a_val, n_val)
                        calc.append([x_mid, z, rho_c])
                        idx += 1
                    elif len(vals) == 1 and stripped:
                        # Nilai RMS di akhir section
                        rms_val = vals[0]
                        idx += 1
                        break
                    elif len(vals) == 0:
                        idx += 1
                    else:
                        idx += 1

            iterations.append({
                "iter": iter_num,
                "rms": rms_val if rms_val is not None else 99.0,
                "layers": layers,
                "layer_depths": layer_depths,
                "calculated_datum": np.array(calc) if calc else None,
            })

        # ── RMS history dari tabel konvergensi di akhir file ────────────────
        rms_from_table = _parse_convergence_table(lines)
        if rms_from_table:
            result["rms_history"] = rms_from_table
        else:
            rms_list = []
            if result["initial_rms"]:
                rms_list.append(result["initial_rms"])
            for it in iterations:
                if it["rms"] < 99.0:
                    rms_list.append(it["rms"])
            result["rms_history"] = rms_list

        # ── Pilih iterasi terbaik ───────────────────────────────────────────
        result["iterations"] = iterations
        result["n_iterations"] = len(iterations)

        if not iterations:
            result["error"] = "Tidak ada data iterasi yang berhasil diparsing."
            return result

        valid_iters = [it for it in iterations if it["layers"]]
        if not valid_iters:
            result["error"] = "Semua iterasi tidak memiliki data layer."
            return result

        best = min(valid_iters, key=lambda it: it["rms"])
        result["final_rms"] = best["rms"]
        result["iteration_used"] = best["iter"]
        result["calculated_datum"] = best["calculated_datum"]

        # ── Bangun matriks resistivitas ─────────────────────────────────────
        _build_rho_matrix(best, result)

        # ── Hitung apparent resistivity dari model (forward calculation) ────
        if result["rho_matrix"] is not None:
            # Deteksi n_max dari data terukur
            n_max = None
            if result["measured_datum"] is not None and len(result["measured_datum"]) > 0:
                n_levels_arr = result["measured_datum"][:, 1] / (
                    0.519 * result["electrode_spacing"]
                    if result["array_type"] not in [1, 3]
                    else (0.5 if result["array_type"] == 1 else 0.3) * result["electrode_spacing"]
                )
                n_max = max(1, int(round(n_levels_arr.max())))

            result["forward_datum"] = compute_apparent_from_model(
                rho_matrix=result["rho_matrix"],
                x_positions=result["x_positions"],
                depths=result["depths"],
                array_code=result["array_type"],
                electrode_spacing=result["electrode_spacing"],
                n_levels=n_max,
            )

        result["success"] = result["rho_matrix"] is not None
        if not result["success"]:
            result["error"] = "Matriks resistivitas tidak dapat dibangun."

        return result

    except Exception as e:
        import traceback
        result["error"] = f"Error parsing file: {str(e)}\n{traceback.format_exc()}"
        return result


# ─── Parse tabel konvergensi ───────────────────────────────────────────────────

def _parse_convergence_table(lines: list) -> list:
    """
    Parse tabel konvergensi di akhir file.
    Mendukung header '% Abs. Error' maupun 'Abs. Error'.
    Format: iter, time_iter, total_time, rms%
    """
    for i, ln in enumerate(lines):
        # Deteksi header tabel: "Iteration  Time ... Error"
        if re.search(r"Iteration\s+Time", ln, re.IGNORECASE) and \
           re.search(r"Error", ln, re.IGNORECASE):
            rms_vals = []
            j = i + 1
            while j < len(lines):
                stripped = lines[j].strip()
                if not stripped:
                    j += 1
                    continue
                vals = _parse_floats(stripped)
                # Baris tabel: 4 kolom (iter, time_iter, total_time, rms%)
                if len(vals) == 4 and vals[0] == int(vals[0]) and vals[0] >= 1:
                    rms_vals.append(vals[3])
                    j += 1
                else:
                    break
            if rms_vals:
                return rms_vals
    return []


# ─── Bangun matriks 2D ────────────────────────────────────────────────────────

def _build_rho_matrix(iteration: dict, result: dict):
    """
    Susun matriks 2D [n_layers × n_cols] dari data per-layer.
    """
    layers = iteration["layers"]
    layer_depths_raw = iteration["layer_depths"]

    if not layers:
        return

    n_layers = max(layers.keys())

    all_x = set()
    for ldata in layers.values():
        for (xp, _) in ldata:
            all_x.add(round(xp, 3))
    all_x = sorted(all_x)

    if not all_x:
        return

    n_cols = len(all_x)
    x_map = {x: i for i, x in enumerate(all_x)}

    # Kedalaman tiap layer
    depths = []
    prev_depth = 0.0
    for li in range(1, n_layers + 1):
        d = layer_depths_raw.get(li)
        if d is not None and d > 0:
            depths.append(round(d, 4))
            prev_depth = d
        else:
            est = round(prev_depth * 1.3, 4) if prev_depth > 0 else result["electrode_spacing"] * 0.5
            depths.append(est)
            prev_depth = est

    # Bangun matriks
    mat = np.full((n_layers, n_cols), np.nan)
    for li in range(1, n_layers + 1):
        for (xp, rho) in layers.get(li, []):
            xi = x_map.get(round(xp, 3))
            if xi is not None and rho > 0:
                mat[li - 1, xi] = rho

    result["rho_matrix"] = mat
    result["depths"] = depths
    result["x_positions"] = all_x
    result["n_rows"] = n_layers
    result["n_cols"] = n_cols

    # Statistik per layer
    layer_avgs = []
    for ri, depth in enumerate(depths):
        row = mat[ri]
        valid = row[~np.isnan(row)]
        valid = valid[valid > 0]
        avg = float(np.mean(valid)) if len(valid) > 0 else 0.0
        layer_avgs.append({
            "layer": ri + 1,
            "depth": depth,
            "avg_rho": round(avg, 2),
            "min_rho": round(float(valid.min()), 2) if len(valid) > 0 else 0.0,
            "max_rho": round(float(valid.max()), 2) if len(valid) > 0 else 0.0,
            "n_blocks": int(len(valid)),
        })
    result["layer_averages"] = layer_avgs


# ─── Demo INV generator ────────────────────────────────────────────────────────

def generate_demo_inv() -> str:
    """Generate contoh file .INV sintetik untuk demo/testing."""
    np.random.seed(42)
    name = "Demo_Lintasan_WS"
    a = 1.0
    array_code = 7
    n_elec = 24
    n_layers = 6
    n_max = 8

    layer_depths = [round(0.519 * (i + 1) * a, 6) for i in range(n_layers)]

    rho_model = [
        [8, 9, 10, 10, 11, 12, 15, 18, 20, 25, 30, 35, 42, 50, 55, 60, 65, 70, 72, 75, 70],
        [7, 8, 9, 10, 12, 14, 18, 22, 28, 35, 45, 55, 65, 75, 80, 85, 90, 95, 90, 85],
        [6, 7, 9, 11, 14, 18, 24, 32, 42, 55, 70, 90, 110, 130, 140, 145, 140, 135],
        [6, 7, 9, 12, 16, 22, 32, 45, 60, 80, 105, 130, 155, 165, 165, 155],
        [7, 9, 12, 17, 24, 36, 52, 72, 98, 130, 162, 185, 195],
        [9, 13, 20, 32, 50, 78, 115, 165, 230, 310],
    ]

    lines = [name, f"    {a:.6f}", f"{array_code}"]
    total_datum = sum(max(0, n_elec - (2 * nv + 1)) for nv in range(1, n_max + 1))
    lines += [str(total_datum), "1", "0"]

    for nv in range(1, n_max + 1):
        n_pts = n_elec - (2 * nv + 1)
        if n_pts <= 0:
            break
        for i in range(n_pts):
            x_mid = (i + nv + 0.5) * a
            rho_a = 10 * nv * (1 + 0.1 * np.random.randn())
            lines.append(f"    {x_mid:.6f},     {a:.6f},   {float(nv):.5f},      {rho_a:.4f}")

    lines += ["INVERSION RESULTS", "Initial RMS error", " 45.00",
              "NUMBER OF LAYERS", str(n_layers),
              "NUMBER OF BLOCKS", str(sum(len(r) for r in rho_model))]

    rms_seq = [45.0, 25.0, 15.0, 10.0]
    for it, rms in enumerate(rms_seq, 1):
        lines += [f"ITERATION {it}", "MODEL RESISTIVITY"]
        for li, (depth, row) in enumerate(zip(layer_depths, rho_model), 1):
            n_blok = len(row) - 1
            lines.append(f"LAYER {li}")
            lines.append(f"{n_blok},     {depth:.6f}")
            for ci, rho in enumerate(row[:-1]):
                x_pos = (ci + 2) * a
                noise = 1 + 0.05 * np.random.randn()
                lines.append(f"    {x_pos:.6f},      {rho * noise:.4f}")
            lines.append(f"                   {row[-1]:.4f}")
        lines.append("CALCULATED APPARENT RESISTIVITY")
        for nv in range(1, n_max + 1):
            n_pts = n_elec - (2 * nv + 1)
            if n_pts <= 0:
                break
            for i in range(n_pts):
                x_mid = (i + nv + 0.5) * a
                rho_c = 10 * nv * (1 + 0.05 * np.random.randn())
                lines.append(f"    {x_mid:.6f},     {a:.6f},   {float(nv):.5f},      {rho_c:.4f}")
        lines.append(f" {rms:.3f}")

    lines += ["", "Iteration  Time for this iteration  Total Time   % Abs. Error"]
    total_t = 0.0
    for it, rms in enumerate(rms_seq, 1):
        t = round(0.5 + 0.3 * np.random.rand(), 2)
        total_t = round(total_t + t, 2)
        lines.append(f"     {it}               {t:.2f}             {total_t:.2f}       {rms:.3f}")
    lines += ["", "RES2DINVx64 ver. 4.10.20"]

    return "\n".join(lines)
