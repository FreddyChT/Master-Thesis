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

# Section headers that appear in SU2 logs
HEADERS = [
    "Physical Case Definition",
    "Space Numerical Integration",
    "Time Numerical Integration",
    "Convergence Criteria",
    "Geometry Preprocessing",
    "Solver Preprocessing",
    "Performance Summary",
]


def condense(section, max_len=200):
    """Return a condensed single line from a section."""
    text = " ".join(line.strip() for line in section if line.strip())
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def ask_inputs():
    """Prompt the user for date string and test number."""
    root = tk.Tk()
    root.withdraw()
    date_str = simpledialog.askstring("Input", "Date string (e.g., 03-07-2025)")
    if not date_str:
        raise SystemExit("No date string provided")
    test_num = simpledialog.askstring("Input", "Test number (e.g., 7)")
    if not test_num:
        raise SystemExit("No test number provided")
    root.destroy()
    return date_str, test_num


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


def parse_geometry(section):
    stats = {}
    patterns = {
        "grid_points": r"(\d+) grid points before partitioning",
        "volume_elements": r"(\d+) volume elements before partitioning",
        "vertices": r"(\d+) vertices including ghost points",
        "elements": r"(\d+) interior elements including halo cells",
        "triangles": r"(\d+) triangles",
        "quads": r"(\d+) quadrilaterals",
    }
    for line in section:
        for key, pat in patterns.items():
            m = re.search(pat, line)
            if m:
                stats[key] = int(m.group(1))
    return stats


def parse_performance(section):
    perf = {}
    for line in section:
        m = re.search(r"Wall-clock time \(hrs\):\s*([\d.eE+-]+)", line)
        if m:
            perf["wall_hours"] = float(m.group(1))
        m = re.search(r"Cores:\s*(\d+)", line)
        if m:
            perf["cores"] = int(m.group(1))
        m = re.search(r"Iteration count:\s*(\d+)", line)
        if m:
            perf["iterations"] = int(m.group(1))
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
    for header in HEADERS:
        section = get_section(text, header)
        if section:
            data[header] = section
    if "Geometry Preprocessing" in data:
        data["mesh"] = parse_geometry(data["Geometry Preprocessing"])
    if "Performance Summary" in data:
        data.update(parse_performance(data["Performance Summary"]))
    iter_count, resid = parse_last_iteration(text)
    data["last_iteration"] = iter_count
    data["final_residual"] = resid
    data["success"] = any("Exit Success" in line for line in text)
    return data


def main():
    date_str, test_num = ask_inputs()
    base = Path(__file__).resolve().parent
    blades_root = base / "Blades"
    summary = []

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
        report_lines = [f"Blade {blade_dir.name}"]
        for h in HEADERS:
            if h in data:
                report_lines.append(f"{h}: {condense(data[h])}")
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
        print(report_text)
        (run_dir / "run_report.txt").write_text(report_text)

    if not summary:
        print("No runs found.")
        return

    blades = [d["blade"] for d in summary]
    times = [d.get("wall_hours", 0) * 60 for d in summary]
    iters = [d.get("iterations", d.get("last_iteration")) for d in summary]
    points = [d.get("mesh", {}).get("grid_points") for d in summary]

    fig, ax = plt.subplots()
    ax.bar(blades, times)
    ax.set_ylabel("Wall-clock time [min]")
    ax.set_title("Convergence time")
    plt.tight_layout()
    for d in summary:
        fig.savefig(d["run_dir"] / "convergence_time.png")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(blades, iters)
    ax.set_ylabel("Iterations")
    ax.set_title("Iteration count")
    plt.tight_layout()
    for d in summary:
        fig.savefig(d["run_dir"] / "iterations.png")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(blades, points)
    ax.set_ylabel("Grid points")
    ax.set_title("Mesh size")
    plt.tight_layout()
    for d in summary:
        fig.savefig(d["run_dir"] / "grid_points.png")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
