"""File reading utilities used across the project."""

import numpy as np
import os
import shutil


def extract_from_ises(file_path):
    """Parse ``ises`` file extracting alpha and Reynolds values.

    This is invoked by ``DataBladeAnalysis v8.py`` prior to geometry
    computation.
    """
    with open(file_path, 'r') as f:
        next(f); next(f)
        tokens = f.readline().split()
        alpha1 = np.float64(tokens[2])
        tokens = f.readline().split()
        alpha2 = np.float64(tokens[2])
        next(f)
        tokens = f.readline().split()
        reynolds = np.float64(tokens[0])
    print("Inlet flow angle (deg):", int(np.degrees(np.arctan(alpha1))))
    print("Outlet flow angle (deg):", int(np.degrees(np.arctan(alpha2))))
    print("Reynolds number:", reynolds)
    return alpha1, alpha2, reynolds


def extract_from_blade(file_path):
    """Return pitch extracted from ``blade`` file."""
    with open(file_path, 'r') as f:
        next(f)
        tokens = f.readline().split()
        pitch = np.float64(tokens[4])
    print("Blade pitch:", pitch)
    return pitch


def extract_from_outlet(file_path):
    """Extract reference Mach and pressure ratio from ``outlet`` file."""
    with open(file_path, 'r') as f:
        for _ in range(19):
            next(f)
        tokens = f.readline().split()
        M1_ref = np.float64(tokens[2])
        for _ in range(3):
            next(f)
        tokens = f.readline().split()
        P21_ratio = np.float64(tokens[2])
    print("Inlet Mach number:", M1_ref)
    print("P2/P1:", P21_ratio)
    return M1_ref, P21_ratio


def copy_blade_file(original_filename):
    """Create a ``.databladeValidation`` copy of *original_filename*."""
    current_directory = os.path.dirname(os.path.abspath(__file__))
    original_filepath = os.path.join(current_directory, original_filename)
    new_filename = original_filename + ".databladeValidation"
    new_filepath = os.path.join(current_directory, new_filename)
    shutil.copyfile(original_filepath, new_filepath)

