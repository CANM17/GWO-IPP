"""
config.py — Parámetros compartidos para comparación justa entre ACO, PSO y GWO-IPP
Todos los algoritmos consumen este archivo. Nada que deba ser igual está fuera de aquí.
"""

import numpy as np
from sklearn.gaussian_process.kernels import RBF

# ─────────────────────────────────────────────
# Entorno
# ─────────────────────────────────────────────
RESOLUTION       = 1
XS               = 100
YS               = 150
# Mapa leído via Data/data_path.py → map_path_classic() → Image/snazzy-image-prueba.png

# ─────────────────────────────────────────────
# Experimento
# ─────────────────────────────────────────────
INITIAL_SEED     = 1_000_000
N_GROUND_TRUTHS  = 10
N_VEHICLES       = 4
TOTAL_DISTANCE   = 200        # px — igual para los tres algoritmos

INITIAL_POSITIONS = np.array([
    [8,  56],
    [37, 16],
    [78, 81],
    [74, 124]
])

# ─────────────────────────────────────────────
# Proceso Gaussiano — idéntico para los tres
# ─────────────────────────────────────────────
GP_LENGTH_SCALE        = 10
GP_LENGTH_SCALE_BOUNDS = (1e-1, 10)
GP_ALPHA               = 1e-6

def make_gp_kernel():
    return RBF(
        length_scale=GP_LENGTH_SCALE,
        length_scale_bounds=GP_LENGTH_SCALE_BOUNDS
    )

# ─────────────────────────────────────────────
# ACO — parámetros propios, no se tocan
# ─────────────────────────────────────────────
ACO_ALPHA              = 1.0   # peso feromona
ACO_BETA               = 2.0   # peso heurística
ACO_RHO                = 0.7   # evaporación
ACO_Q                  = 1.0   # feromona depositada
ACO_N_ANTS             = 20
ACO_N_ITER             = 30
ACO_EXPLORATION_DIST   = 50    # px — umbral exploración/explotación propio del ACO

# ─────────────────────────────────────────────
# PSO — parámetros propios, no se tocan
# ─────────────────────────────────────────────
PSO_EXPLORATION_DIST   = 50    # px — umbral exploración/explotación propio del PSO

# ─────────────────────────────────────────────
# GWO-IPP — parámetros propios
# ─────────────────────────────────────────────
GWO_N_WOLVES           = 20    # tamaño de la manada virtual por ASV
GWO_N_ITER             = 30    # iteraciones de la manada por re-planificación
GWO_W3_MIN             = 0.1   # piso mínimo del peso de diversidad f3 (peor caso 1)
GWO_A_EXP              = 2   # exponente de a(t) = 2·(1-t/T)^a_exp (1=lineal, 2=cóncavo)

# ─────────────────────────────────────────────
# Métricas — umbrales compartidos
# ─────────────────────────────────────────────
COVERAGE_SIGMA_THRESHOLD = None  # se calcula como 0.1 * sigma_max en t=0 por cada corrida
