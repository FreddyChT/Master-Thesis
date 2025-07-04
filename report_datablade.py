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

# Section headers used in SU2 logs
HEADERS = ["Geometry Preprocessing", "Performance Summary"]


class DualEntryDialog(simpledialog.Dialog):
    """Simple dialog with two text entries."""

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
        self.result = (self.date_var.get().strip(), self.test_var.get().strip())


def ask_inputs():
    """Prompt the user for date string and test number in one window."""
    root = tk.Tk()
    root.withdraw()
    dialog = DualEntryDialog(root, title="Select run")
    root.destroy()
    if not dialog.result or not dialog.result[0] or not dialog.result[1]:
        raise SystemExit("Inputs required")
    return dialog.result


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
        idx = find_header(lines[start + 1 :], other)
        if idx != -1:
            idx += start + 1
            if idx < end:
                end = idx
    return lines[start + 1 : end]


NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_geometry(section):
    """Parse mesh statistics and extract quality-related lines."""
    stats = {}
    lines = []
    capture_quality = False

    for line in section:
        # basic mesh size information
        if m := re.search(rf"(\d+)\s+grid points before partitioning", line, re.I):
            stats["grid_points"] = int(m.group(1))
        if m := re.search(rf"(\d+)\s+interior elements including halo cells", line, re.I):
            stats["elements"] = int(m.group(1))

        # quality metrics
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
            lines.append(stripped)
        if re.search(r"computing mesh quality", line, re.I):
            capture_quality = True
            lines.append(stripped)
            continue
        if capture_quality:
            if re.search(r"finding max control volume width", line, re.I):
                capture_quality = False
                continue
            lines.append(stripped)

    return stats, lines


def parse_performance(section):
    """Parse wall time, core count, iteration info and capture summary lines."""
    perf = {}
    start_idx = None
    end_idx = None
    for i, line in enumerate(section):
        m = re.search(r"Wall-clock time \(hrs\):\s*([\d.eE+-]+)", line)
        if m:
            perf["wall_hours"] = float(m.group(1))
        m = re.search(r"Cores:\s*(\d+)", line)
        if m:
            perf["cores"] = int(m.group(1))
        m = re.search(r"Iteration count:\s*(\d+)", line)
        if m:
            perf["iterations"] = int(m.group(1))
        if start_idx is None and line.strip().lower().startswith("simulation totals"):
            start_idx = i
        if start_idx is not None and end_idx is None and line.strip().lower().startswith("restart aggr"):
            end_idx = i
    if start_idx is not None:
        if end_idx is None:
            end_idx = len(section)
        perf_lines = [section[j].strip() for j in range(start_idx, end_idx)]
    else:
        perf_lines = [l.strip() for l in section if l.strip()]
    perf["performance_lines"] = perf_lines
    return perf


def parse_last_iteration(lines):
    # Search from bottom for the last iteration entry
    for line in reversed(lines):
        m = re.search(r"\|\s*(\d+)\|\s*[\d.eE+-]+\|\s*([-\d.eE+]+)\|", line)
        if m:
            return int(m.group(1)), float(m.group(2))
    # Fallback line in run_summary
    for line in reversed(lines):
        m = re.search(r"Last iteration:\s*(\d+)", line)
        if m:
            return int(m.group(1)), None
    return None, None


def parse_log(log_path):
    text = Path(log_path).read_text().splitlines()
    data = {}
    geom_sec = get_section(text, "Geometry Preprocessing")
    if geom_sec:
        mesh, lines = parse_geometry(geom_sec)
        data["mesh"] = mesh
        data["geometry_lines"] = lines
    perf_sec = get_section(text, "Performance Summary")
    if perf_sec:
        data.update(parse_performance(perf_sec))
    iter_count, resid = parse_last_iteration(text)
    data["last_iteration"] = iter_count
    data["final_residual"] = resid
    data["success"] = any("Exit Success" in line for line in text)
    return data


def plot_metric(blades, values, colors, ylabel, title, filename, out_dirs):
    """Create a bar plot for a given metric and save to all run directories."""
    fig, ax = plt.subplots(figsize=(max(6, len(blades)), 4))
    ax.bar(blades, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=90)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color="blue"),
                      plt.Rectangle((0, 0), 1, 1, color="red")]
    ax.legend(legend_handles, ["Converged", "Diverged"], frameon=False)
    plt.tight_layout()
    for d in out_dirs:
        fig.savefig(d / filename)
    plt.show()
    plt.close(fig)


def main():
    date_str, test_num = ask_inputs()
    base = Path(__file__).resolve().parent
    blades_root = base / "Blades"
    reports_dir = base / "reports"
    reports_dir.mkdir(exist_ok=True)

    run_reports_dir = reports_dir / f"{date_str}_Test_{test_num}"
    run_reports_dir.mkdir(exist_ok=True)

    summary = []
    report_entries = []

    for blade_dir in sorted(blades_root.iterdir()):
        if not blade_dir.is_dir():
            continue
        run_dir = blade_dir / "results" / f"Test_{test_num}_{date_str}"
        if not run_dir.exists():
            continue
        log_file = run_dir / "su2.log"
        if not log_file.is_file():
            log_file = run_dir / "run_summary.txt"
        if not log_file.is_file():
            continue
        data = parse_log(log_file)
        data["blade"] = blade_dir.name
        data["run_dir"] = run_dir
        summary.append(data)
        report_lines = [f"Blade: {blade_dir.name}"]
        if data.get("geometry_lines"):
            report_lines.append("Geometry Preprocessing:")
            report_lines.extend(data["geometry_lines"])
            report_lines.append("")
        if data.get("performance_lines"):
            report_lines.append("Performance Summary:")
            report_lines.extend(data["performance_lines"])
            report_lines.append("")
        mesh = data.get("mesh", {})
        if mesh:
            report_lines.append(
                "Mesh points: {} elements: {}".format(
                    mesh.get("grid_points", "N/A"), mesh.get("elements", "N/A")
                )
            )
        wall = data.get("wall_hours", 0) * 60
        iters = data.get("iterations", data.get("last_iteration"))
        report_lines.append(
            f"Wall time [min]: {wall:.1f}  Iterations: {iters}  "
            f"Cores: {data.get('cores', 'N/A')}  "
            f"Success: {data['success']}"
        )
        if data.get("final_residual") is not None:
            report_lines.append(
                f"Last iteration: {data.get('last_iteration')}  Final residual: {data.get('final_residual')}"
            )
        report_text = "\n".join(report_lines)
        report_entries.append(report_text)

    if not summary:
        print("No runs found.")
        return

    divider = "\n" + ("-" * 40) + "\n"
    overall_text = divider.join(report_entries)
    report_file = run_reports_dir / f"{date_str}_Test_{test_num}_report.txt"
    report_file.write_text(overall_text)
    print(f"Report saved to {report_file}")

    blades = [d["blade"] for d in summary]
    times = [d.get("wall_hours", 0) * 60 for d in summary]
    iters = [d.get("iterations", d.get("last_iteration")) for d in summary]
    points = [d.get("mesh", {}).get("grid_points") for d in summary]
    angles = [d.get("mesh", {}).get("min_orth_angle") for d in summary]
    face_ar = [d.get("mesh", {}).get("max_face_area_ar") for d in summary]
    subvol = [d.get("mesh", {}).get("max_subvol_ratio") for d in summary]
    colors = ["blue" if d.get("success") else "red" for d in summary]
    out_dirs = [run_reports_dir]

    plot_metric(blades, times, colors, "Wall-clock time [min]", "Convergence time",
                "convergence_time.png", out_dirs)
    plot_metric(blades, iters, colors, "Iterations", "Iteration count",
                "iterations.png", out_dirs)
    plot_metric(blades, points, colors, "Grid points", "Mesh size",
                "grid_points.png", out_dirs)

    if any(a is not None for a in angles):
        plot_metric(blades, angles, colors, "Min orthogonality angle",
                    "Minimum Orthogonality", "min_orth_angle.png", out_dirs)

    if any(a is not None for a in face_ar):
        plot_metric(blades, face_ar, colors, "Max CV Face Area AR",
                    "Face Area Aspect Ratio", "max_face_area_ar.png", out_dirs)

    if any(a is not None for a in subvol):
        plot_metric(blades, subvol, colors, "Max CV sub-volume ratio",
                    "Sub-volume Ratio", "max_subvol_ratio.png", out_dirs)


if __name__ == "__main__":
    main()