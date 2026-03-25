"""
run_gwo.py — Experimento GWO-IPP sobre el Lago Ypacaraí

Condiciones idénticas a run_aco.py y run_pso.py:
- Misma semilla inicial, mismas posiciones, misma distancia total
- Mismas métricas calculadas desde Core/metrics.py
- Resultados y gráficas guardados en tiempo real por GT en Results/GWO/
"""

import matplotlib
matplotlib.use('Agg')  # sin ventanas — guardar directo a archivo
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import time
import os
import sys

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# Raíz del proyecto autónomo — un nivel arriba de Experiments/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Experiments.config import (
    RESOLUTION, XS, YS, INITIAL_SEED, N_GROUND_TRUTHS,
    N_VEHICLES, TOTAL_DISTANCE, INITIAL_POSITIONS
)
from Algorithms.gwo.gwo_environment import GWOEnvironment
from Environment.plot import Plots

# ─────────────────────────────────────────────
# Directorios de salida
# ─────────────────────────────────────────────
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Results/GWO'))
PLOTS_DIR  = os.path.join(OUTPUT_DIR, 'Plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ─────────────────────────────────────────────
# Inicialización
# ─────────────────────────────────────────────
gwo_env = GWOEnvironment(
    resolution=RESOLUTION,
    ys=YS,
    initial_seed=INITIAL_SEED,
    initial_position=INITIAL_POSITIONS,
    number_of_vehicles=N_VEHICLES,
    total_distance=TOTAL_DISTANCE
)

# ─────────────────────────────────────────────
# Loop de experimento
# ─────────────────────────────────────────────
results = []

for i in range(N_GROUND_TRUTHS):
    time_init = time.time()
    gwo_env.reset()

    done = False
    while not done:
        done = gwo_env.step()

    elapsed   = time.time() - time_init
    seed_used = gwo_env.seed
    mse_map   = gwo_env.map_mse[-1]
    r2        = gwo_env.r2_map[-1]
    mse_peak  = gwo_env.peak_mse[-1]

    print(f"GT {i} | seed {seed_used} | MSE map: {mse_map:.5f} | "
          f"R2: {r2:.5f} | MSE peaks: {mse_peak:.5f} | t: {elapsed:.1f}s")

    # Guardar métricas en tiempo real
    row = {
        'gt':        i,
        'seed':      seed_used,
        'mse_map':   mse_map,
        'r2_map':    r2,
        'mse_peaks': mse_peak,
        'time_s':    elapsed
    }
    results.append(row)
    pd.DataFrame([row]).to_csv(
        os.path.join(OUTPUT_DIR, f'result_gt_{i}.csv'), index=False
    )

    # Guardar gráficas
    X_test, secure, bench_function, grid_min, sigma, \
        mu, part_ant, bench_array, grid_or, bench_max = gwo_env.data_out()

    plot = Plots(XS, YS, X_test, grid_or, bench_function, grid_min, grid_or)

    # Trayectorias + sigma + mu
    plot.plot_classic(mu, sigma, part_ant)
    plt.savefig(os.path.join(PLOTS_DIR, f'trayectoria_gt_{i}.png'),
                dpi=150, bbox_inches='tight')
    plt.close('all')

    # Ground truth
    plot.benchmark()
    plt.savefig(os.path.join(PLOTS_DIR, f'benchmark_gt_{i}.png'),
                dpi=150, bbox_inches='tight')
    plt.close('all')

# ─────────────────────────────────────────────
# Resumen final
# ─────────────────────────────────────────────
df = pd.DataFrame(results)

summary = {
    'gt':        'PROMEDIO',
    'seed':      '-',
    'mse_map':   df['mse_map'].mean(),
    'r2_map':    df['r2_map'].mean(),
    'mse_peaks': df['mse_peaks'].mean(),
    'time_s':    df['time_s'].mean()
}
std_row = {
    'gt':        'STD_95',
    'seed':      '-',
    'mse_map':   df['mse_map'].std() * 1.96,
    'r2_map':    df['r2_map'].std() * 1.96,
    'mse_peaks': df['mse_peaks'].std() * 1.96,
    'time_s':    df['time_s'].std() * 1.96
}

df_out = pd.concat([df, pd.DataFrame([summary, std_row])], ignore_index=True)
df_out.to_excel(os.path.join(OUTPUT_DIR, 'Resultados_GWO.xlsx'), index=False)

print("\n=== Resultados GWO-IPP ===")
gwo_env.print_error()