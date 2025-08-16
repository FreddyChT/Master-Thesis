# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 17:30:28 2025

@author: fredd
Generate summary reports for SU2 runs.

Disclaimer: GPT-o3 & Codex were heavily used for the elaboration of this script
"""
    
import re
from pathlib import Path
import tkinter as tk
from tkinter import simpledialog
import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.collections import LineCollection


# ────────── Matplotlib / LaTeX setup ─────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble": r"\usepackage{helvet}"
})

HEADERS = ["Geometry Preprocessing", "Performance Summary"]


ORDER_RIGHT_TO_LEFT = [
    
    #Ordered by agressiveness definition (largest to lowest)
    "Blade 6", "Blade 8", "Blade 13", "Blade 19", "Blade 22", "Blade 16",
    "Blade 26", "Blade 2", "Blade 3", "Blade 4", "Blade 5", "Blade 14",
    "Blade 15", "Blade 18", "Blade 21", "Blade 17", "Blade 23", "Blade 1",
    "Blade 24", "Blade 11", "Blade 12", "Blade 20", "Blade 25", "Blade 7",
    "Blade 9", "Blade 10", "Blade 0"          # keep Blade 0 last
    
    #Ordered by Inlet flow angle (largest to lowest)
#    "Blade 13", "Blade 1", "Blade 19", "Blade 6", "Blade 8", "Blade 15",
#    "Blade 2", "Blade 4", "Blade 5", "Blade 14", "Blade 23", "Blade 18",
#    "Blade 3", "Blade 26", "Blade 22", "Blade 21", "Blade 12", "Blade 11",
#    "Blade 7", "Blade 10", "Blade 16", "Blade 17", "Blade 20", "Blade 24",
#    "Blade 25", "Blade 9", "Blade 0"          # keep Blade 0 last
    
    #Ordered by Outlet flow angle (largest to lowest)
#    "Blade 6", "Blade 8", "Blade 18", "Blade 16", "Blade 17", "Blade 26",
#    "Blade 19", "Blade 13", "Blade 22", "Blade 21", "Blade 15", "Blade 2",
#    "Blade 4", "Blade 5", "Blade 14", "Blade 3", "Blade 20", "Blade 24",
#    "Blade 25", "Blade 7", "Blade 23", "Blade 1", "Blade 12", "Blade 11",
#    "Blade 10", "Blade 9", "Blade 0"          # keep Blade 0 last
    
    #Ordered by Turning angle (largest to lowest)
#    "Blade 9", "Blade 10", "Blade 24", "Blade 25", "Blade 7", "Blade 20",
#    "Blade 12", "Blade 11", "Blade 16", "Blade 17", "Blade 22", "Blade 21",
#    "Blade 23", "Blade 3", "Blade 26", "Blade 1", "Blade 15", "Blade 2",
#    "Blade 4", "Blade 5", "Blade 14", "Blade 18", "Blade 19", "Blade 6",
#    "Blade 8", "Blade 13", "Blade 0"          # keep Blade 0 last

    #Ordered by Pitch-to-chord ratio (largest to lowest)
#    "Blade 6", "Blade 8", "Blade 13", "Blade 16", "Blade 22", "Blade 15",
#    "Blade 26", "Blade 24", "Blade 2", "Blade 18", "Blade 19", "Blade 4",
#    "Blade 5", "Blade 14", "Blade 3", "Blade 12", "Blade 17", "Blade 25",
#    "Blade 21", "Blade 9", "Blade 10", "Blade 1", "Blade 23", "Blade 20",
#    "Blade 7", "Blade 11", "Blade 0"          # keep Blade 0 last
]

# ────────── TK helpers ───────────────────────────────────────────────────────
class DualEntryDialog(simpledialog.Dialog):
    def body(self, master):
        tk.Label(master, text="Date string (e.g., 03-07-2025)").grid(row=0, column=0, sticky="w")
        self.date_var = tk.StringVar()
        self.date_entry = tk.Entry(master, textvariable=self.date_var)
        self.date_entry.grid(row=0, column=1)

        tk.Label(master, text="Test number (e.g., 7)").grid(row=1, column=0, sticky="w")
        self.test_var = tk.StringVar()
        tk.Entry(master, textvariable=self.test_var).grid(row=1, column=1)
        return self.date_entry

    def apply(self):
        self.result = self.date_var.get().strip(), self.test_var.get().strip()


def ask_inputs():
    root = tk.Tk()
    root.withdraw()
    dialog = DualEntryDialog(root, title="Select run")
    root.destroy()
    if not dialog.result[0] or not dialog.result[1]:
        raise SystemExit("Inputs required")
    return dialog.result


# ────────── SU2‑log parsing helpers ─────────────────────────────────────────
def find_header(lines, header):
    for i, line in enumerate(lines):
        if header in line:
            return i
    return -1


def get_section(lines, header):
    start = find_header(lines, header)
    if start == -1:
        return []
    end = len(lines)
    for other in HEADERS:
        if other == header:
            continue
        idx = find_header(lines[start + 1:], other)
        if idx != -1:
            end = min(end, idx + start + 1)
    return lines[start + 1:end]


NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_geometry(section):
    stats, out_lines, capture_quality = {}, [], False
    for line in section:
        if m := re.search(rf"(\d+)\s+grid points before partitioning", line, re.I):
            stats["grid_points"] = int(m.group(1))
        if m := re.search(rf"(\d+)\s+interior elements including halo cells", line, re.I):
            stats["elements"] = int(m.group(1))

        if re.search(r"Orthogonality Angle", line, re.I):
            nums = re.findall(NUM_RE, line)
            if nums:
                stats["min_orth_angle"] = float(nums[0])
        if re.search(r"CV Face Area Aspect Ratio", line, re.I):
            nums = re.findall(NUM_RE, line)
            if nums:
                stats["max_face_area_ar"] = float(nums[-1])
        if re.search(r"CV Sub-Volume Ratio", line, re.I):
            nums = re.findall(NUM_RE, line)
            if nums:
                stats["max_subvol_ratio"] = float(nums[-1])

        stripped = line.strip()
        if stripped.startswith("Max K"):
            out_lines.append(stripped)
        if re.search(r"computing mesh quality", line, re.I):
            capture_quality = True
            out_lines.append(stripped)
            continue
        if capture_quality:
            if re.search(r"finding max control volume width", line, re.I):
                capture_quality = False
                continue
            out_lines.append(stripped)

    return stats, out_lines


def parse_performance(section):
    perf, start_idx, end_idx = {}, None, None
    for i, line in enumerate(section):
        if m := re.search(r"Wall-clock time \(hrs\):\s*([\d.eE+-]+)", line):
            perf["wall_hours"] = float(m.group(1))
        if m := re.search(r"Cores:\s*(\d+)", line):
            perf["cores"] = int(m.group(1))
        if m := re.search(r"Iteration count:\s*(\d+)", line):
            perf["iterations"] = int(m.group(1))
        if m := re.search(r"Core-hrs:\s*([\d.eE+-]+)", line):
            perf["core_hours"] = float(m.group(1))
        if start_idx is None and line.strip().lower().startswith("simulation totals"):
            start_idx = i
        if start_idx is not None and end_idx is None and line.strip().lower().startswith("restart aggr"):
            end_idx = i
    lines = (section[start_idx:end_idx] if start_idx is not None
             else [l for l in section if l.strip()])
    perf["performance_lines"] = [l.strip() for l in lines]
    return perf


def parse_last_iteration(lines):
    for line in reversed(lines):
        if m := re.search(r"\|\s*(\d+)\|\s*[\d.eE+-]+\|\s*([-\d.eE+]+)\|", line):
            return int(m.group(1)), float(m.group(2))
    for line in reversed(lines):
        if m := re.search(r"Last iteration:\s*(\d+)", line):
            return int(m.group(1)), None
    return None, None


def parse_log(log_path):
    text = Path(log_path).read_text().splitlines()
    data = {}
    if geom := get_section(text, "Geometry Preprocessing"):
        mesh, lines = parse_geometry(geom)
        data.update(mesh=mesh, geometry_lines=lines)
    if perf := get_section(text, "Performance Summary"):
        data.update(parse_performance(perf))
    iters, resid = parse_last_iteration(text)
    data["last_iteration"], data["final_residual"] = iters, resid
    for line in text:
        if "Mach RMS error" in line:
            try:
                data["mach_rms"] = float(line.split(":", 1)[1])
            except (IndexError, ValueError):
                pass
        if "wake" in line.lower() and "loss" in line.lower() and "error" in line.lower():
            nums = re.findall(NUM_RE, line)
            if nums:
                data["wake_loss_error"] = float(nums[0])
        if "c_f mismatch" in line.lower() or "cf mismatch" in line.lower():
            nums = re.findall(NUM_RE, line)
            if nums:
                data["cf_mismatch_band"] = float(nums[0])
    data["success"] = any("Exit Success" in line for line in text)
    return data


# ────────── Plot helper ──────────────────────────────────────────────────────
def plot_metric(blades, values, colors, ylabel, title, filename, out_dirs):
    """
    Bar‑plot a metric, with:
      * mean & gradient ignoring diverged blades and Blade 0,
      * Blade 0 displayed as SPLEEN,
      * horizontal compression,
      * colour‑bar explaining gradient (label “Aggressiveness”, ticks “–”, “+”).
    """
    blades = list(blades)
    values = np.asarray(values, float)
    colors = list(colors) if len(colors) != 1 else list(colors) * len(values)
    if len(blades) != len(values):
        raise ValueError("blades and values length mismatch")

    converged = np.array([c != "red" for c in colors])
    spleen = np.array([b.endswith(" 0") or b == "Blade 0" for b in blades])
    valid = converged & ~spleen & np.isfinite(values)

    mean_val = np.mean(values[valid]) if valid.any() else np.nan

    fig_w = max(6, 0.35 * len(blades))      # compressed width
    x = np.arange(len(blades))
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    ax.bar(x, values, color=colors, width=0.4)

    if np.isfinite(mean_val):
        ax.axhline(mean_val, color="gray", linestyle="--", linewidth=1)

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    def tick(b):
        if b.endswith(" 0") or b == "Blade 0":
            return "EXP"
        m = re.search(r"(\d+)$", b)
        return m.group(1) if m else b

    ax.set_xticks(x)
    ax.set_xticklabels([tick(b) for b in blades], rotation=0)
    ax.set_xlabel("Blades")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="blue"),
        plt.Rectangle((0, 0), 1, 1, color="red"),
    ]
    labels = ["Converged", "Diverged"]
    if np.isfinite(mean_val):
        handles.append(ax.lines[-1])
        labels.append(f"Mean = {mean_val:.3g}")
    ax.legend(handles, labels, frameon=True)

    # Gradient & colour‑bar
    cmap = plt.get_cmap("RdYlGn_r")
    if valid.sum() >= 2:
        xs, ys = x[valid], values[valid]
        segs = np.concatenate([np.column_stack([xs, ys]).reshape(-1, 1, 2)[:-1],
                               np.column_stack([xs, ys]).reshape(-1, 1, 2)[1:]], axis=1)
        lc = LineCollection(segs, array=np.linspace(0, 1, len(xs)), cmap=cmap)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        ax.scatter(xs, ys, color=cmap(np.linspace(0, 1, len(xs))), zorder=3)

        # ScalarMappable for colour‑bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Aggressiveness", rotation=90)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["–", "+"])

    ax.margins(x=0.05)
    plt.tight_layout()

    for d in out_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(d) / filename, bbox_inches="tight", dpi=200)
    plt.show()
    plt.close(fig)


# ────────── Main ────────────────────────────────────────────────────────────
def main():
    date_str, test_num = ask_inputs()
    base = Path(__file__).resolve().parent
    blades_root = base / "Blades"
    reports_dir = base / "reports"
    reports_dir.mkdir(exist_ok=True)
    run_reports_dir = reports_dir / f"{date_str}_Test_{test_num}"
    run_reports_dir.mkdir(exist_ok=True)

    summary, report_entries = [], []

    def blade_sort_key(p: Path):
        m = re.search(r"(\d+)$", p.name)
        return -(int(m.group(1)) if m else 0)

    for bdir in sorted(blades_root.iterdir(), key=blade_sort_key):
        if not bdir.is_dir():
            continue
        run_dir = bdir / "results" / f"Test_{test_num}_{date_str}"
        if not run_dir.exists():
            continue
        log_file = run_dir / "su2.log"
        if not log_file.is_file():
            log_file = run_dir / "run_summary.txt"
        if not log_file.is_file():
            continue

        data = parse_log(log_file)
        blade_name = bdir.name.replace("_", " ")
        data.update(blade=blade_name, run_dir=run_dir)

        # Iterations to reach residual 1e-7 from history file
        hist_file = run_dir / f"history_{date_str}_{bdir.name}.csv"
        if hist_file.is_file():
            try:
                hist = np.genfromtxt(hist_file, delimiter=",", names=True)
                resid = hist['RMS_DENSITY']
                itnum = hist['INNER_ITER']
                mask = resid <= 1e-7
                if mask.any():
                    data['iter_1e7'] = int(itnum[mask][0])
            except Exception:
                pass

        summary.append(data)

        rl = [f"Blade: {blade_name}"]
        if lines := data.get("geometry_lines"):
            rl.append("Geometry Preprocessing:"); rl.extend(lines); rl.append("")
        if lines := data.get("performance_lines"):
            rl.append("Performance Summary:"); rl.extend(lines); rl.append("")
        mesh = data.get("mesh", {})
        if mesh:
            rl.append(f"Mesh points: {mesh.get('grid_points','N/A')} "
                      f"elements: {mesh.get('elements','N/A')}")
        wall = data.get("wall_hours", 0) * 60
        iters = data.get("iterations", data.get("last_iteration"))
        rl.append(f"Wall time [min]: {wall:.1f}  Iterations: {iters}  "
                  f"Cores: {data.get('cores','N/A')}  Success: {data['success']}")
        if data.get("final_residual") is not None:
            rl.append(f"Last iteration: {data['last_iteration']}  "
                      f"Final residual: {data['final_residual']}")
        if data.get("mach_rms") is not None:
            rl.append(f"Mach RMS error: {data['mach_rms']:.4f}")
        report_entries.append("\n".join(rl))

    if not summary:
        print("No runs found."); return

    order_lr = list(reversed(ORDER_RIGHT_TO_LEFT))
    s_map = {d["blade"]: d for d in summary}
    ordered = [s_map[b] for b in order_lr if b in s_map]

    def bnum(name): m = re.search(r"(\d+)$", name); return int(m.group(1)) if m else -1
    ordered += sorted([d for d in summary if d["blade"] not in order_lr],
                      key=lambda d: bnum(d["blade"]))
    summary = ordered or summary

    divider = "\n" + "-" * 40 + "\n"
    (run_reports_dir / f"{date_str}_Test_{test_num}_report.txt") \
        .write_text(divider.join(report_entries))

    # Summary table per blade
    table_path = run_reports_dir / f"{date_str}_Test_{test_num}_summary.csv"
    headers = ["Blade", "Mach_RMS", "Wake_Loss_Error", "C_f_mismatch_band",
               "Iter_to_1e-7", "Wall_time_s", "Core_h"]
    with table_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for d in summary:
            wall_s = d.get("wall_hours", 0.0) * 3600
            core_h = d.get("core_hours", d.get("wall_hours", 0.0) * d.get("cores", 0))
            writer.writerow([
                d["blade"],
                d.get("mach_rms", np.nan),
                d.get("wake_loss_error", np.nan),
                d.get("cf_mismatch_band", np.nan),
                d.get("iter_1e7", np.nan),
                wall_s,
                core_h,
            ])

    blades = [d["blade"] for d in summary]
    times  = np.array([d.get("wall_hours", 0) * 60 for d in summary], float)
    iters  = np.array([d.get("iterations", d.get("last_iteration")) or np.nan
                       for d in summary], float)
    mesh   = [d.get("mesh", {}) for d in summary]
    points = np.array([m.get("grid_points", np.nan) for m in mesh], float)
    angles = np.array([m.get("min_orth_angle", np.nan) for m in mesh], float)
    face_ar= np.array([m.get("max_face_area_ar", np.nan) for m in mesh], float)
    subvol = np.array([m.get("max_subvol_ratio", np.nan) for m in mesh], float)

    colors = ["blue" if d.get("success") else "red" for d in summary]
    out_dirs = [run_reports_dir]
    any_fin = lambda a: np.isfinite(a).any()

    if any_fin(times):
        plot_metric(blades, times, colors, "Wall-clock time [min]",
                    "Convergence time", "convergence_time.png", out_dirs)
    if any_fin(iters):
        plot_metric(blades, iters, colors, "Iterations",
                    "Iteration count", "iterations.png", out_dirs)
    if any_fin(points):
        plot_metric(blades, points, colors, "Grid points",
                    "Mesh size", "grid_points.png", out_dirs)
    if any_fin(angles):
        plot_metric(blades, angles, colors, "Min orthogonality angle",
                    "Minimum Orthogonality", "min_orth_angle.png", out_dirs)
    if any_fin(face_ar):
        plot_metric(blades, face_ar, colors, "Max CV Face Area AR",
                    "Face Area Aspect Ratio", "max_face_area_ar.png", out_dirs)
    if any_fin(subvol):
        plot_metric(blades, subvol, colors, "Max CV sub-volume ratio",
                    "Sub-volume Ratio", "max_subvol_ratio.png", out_dirs)


if __name__ == "__main__":
    main()