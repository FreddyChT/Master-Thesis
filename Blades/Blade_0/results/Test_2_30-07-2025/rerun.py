
#!/usr/bin/env python3
#Created on 30-07-2025, 21:37:19
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[4]))
import mesh_datablade
import configSU2_datablade
import post_processing_datablade

bladeName = 'Blade_0'
no_cores = 12
string = 'databladeVALIDATION'
fileExtension = 'csv'

run_dir = Path(__file__).resolve().parent
base_dir = Path(__file__).resolve().parents[4]
blade_dir = base_dir / 'Blades' / bladeName
isesFilePath = blade_dir / f'ises.databladeVALIDATION'
bladeFilePath = blade_dir / f'blade.databladeVALIDATION'
outletFilePath = blade_dir / f'outlet.databladeVALIDATION'
