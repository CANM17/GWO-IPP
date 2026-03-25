"""
optimize_gwo.py — Optimización Bayesiana de hiperparámetros GWO-IPP
                  con scikit-optimize (skopt) / gp_minimize

Referencia metodológica:
    Snoek, J., Larochelle, H., & Adams, R.P. (2012).
    Practical Bayesian Optimization of Machine Learning Algorithms.
    Advances in Neural Information Processing Systems (NeurIPS), 25.

    Scikit-optimize: https://scikit-optimize.github.io/

Objetivo (minimizar):
    J = 0.5 · MSE_picos + 0.5 · (1 − R²_mapa)

Espacio de búsqueda:
    n_wolves : Integer [5,  50]   — tamaño de la manada virtual por ASV
    n_iter   : Integer [10, 60]   — iteraciones por re-planificación
    w3_min   : Real    [0.0, 0.3] — piso mínimo del peso de diversidad f3
    a_exp    : Real    [1.0, 3.0] — exponente de a(t) = 2·(1 - t/T)^a_exp
                                    1.0 → lineal (GWO estándar, Mirjalili 2014)
                                    2.0 → cóncavo (GWO-IPP v3)
                                    >2  → mayor exploración inicial

Uso:
    cd YpacaraiIPP/Experiments
    pip install scikit-optimize
    python optimize_gwo.py

Salida:
    Results/GWO_optim/bo_trials.csv        — historial completo de trials
    Results/GWO_optim/best_params.txt      — mejores hiperparámetros
    Results/GWO_optim/bo_convergence.png   — curva de convergencia y J vs params
    Results/GWO_optim/bo_w3min.png         — J vs w3_min
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import types
import time
import os
import sys

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# ─── scikit-optimize ──────────────────────────────────────────────────────────
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

# ─── Raíz del proyecto ────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Experiments.config import (
    RESOLUTION, YS, INITIAL_SEED, N_GROUND_TRUTHS,
    N_VEHICLES, TOTAL_DISTANCE, INITIAL_POSITIONS
)
from Algorithms.gwo.gwo_environment import GWOEnvironment

# ─── Directorio de salida ─────────────────────────────────────────────────────
OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../Results/GWO_optim')
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Configuración ────────────────────────────────────────────────────────────
N_CALLS          = 50
N_INITIAL_POINTS = 10   # puntos aleatorios antes de usar el surrogate GP (~20% de N_CALLS)
RANDOM_STATE     = 42

# ─── Espacio de búsqueda ──────────────────────────────────────────────────────
space = [
    Integer(5,   50,  name='n_wolves'),
    Integer(10,  60,  name='n_iter'),
    Real   (0.0, 0.3, name='w3_min', prior='uniform'),
    Real   (1.0, 3.0, name='a_exp',  prior='uniform'),
]

# Punto de arranque: GWO-IPP v3 (parámetros actuales conocidos)
X0 = [20, 30, 0.1, 2.0]


# ─── Parche para a_exp generalizado ───────────────────────────────────────────
def _make_patched_run(a_exp_val):
    def _run(self, candidates, mu_candidates, sigma_candidates,
             assigned_waypoints, w1, w2, w3):
        M = len(candidates)
        if M <= 1:
            return 0, candidates[0]
        F = self._compute_fitness(
            candidates, mu_candidates, sigma_candidates,
            assigned_waypoints, w1, w2, w3
        )
        wolf_indices = np.random.choice(M, size=min(self.n_wolves, M), replace=True)
        wolves = candidates[wolf_indices].astype(float)
        alpha_pos, beta_pos, delta_pos = self._get_leaders(candidates, F)
        for t in range(self.n_iter):
            a = 2.0 * (1.0 - t / self.n_iter) ** a_exp_val
            for k in range(len(wolves)):
                wolves[k] = self._update_position(
                    wolves[k], alpha_pos, beta_pos, delta_pos, a
                )
                wolves[k] = self._clamp_to_bounds(wolves[k], candidates)
            wolf_candidate_indices = self._project_to_candidates(wolves, candidates)
            wolf_fitness = F[wolf_candidate_indices]
            sorted_idx = np.argsort(-wolf_fitness)
            alpha_pos = wolves[sorted_idx[0]].copy()
            beta_pos  = wolves[sorted_idx[1]].copy() if len(sorted_idx) > 1 else alpha_pos
            delta_pos = wolves[sorted_idx[2]].copy() if len(sorted_idx) > 2 else beta_pos
        best_idx = self._project_to_candidates(alpha_pos.reshape(1, 2), candidates)[0]
        return best_idx, candidates[best_idx]
    return _run


# ─── Historial ────────────────────────────────────────────────────────────────
history      = []
call_counter = [0]


# ─── Función objetivo ─────────────────────────────────────────────────────────
@use_named_args(space)
def objective(n_wolves, n_iter, w3_min, a_exp):
    call_counter[0] += 1
    t0 = time.time()

    gwo_env = GWOEnvironment(
        resolution=RESOLUTION,
        ys=YS,
        initial_seed=INITIAL_SEED,
        initial_position=INITIAL_POSITIONS,
        number_of_vehicles=N_VEHICLES,
        total_distance=TOTAL_DISTANCE,
        n_wolves=int(n_wolves),
        n_iter=int(n_iter),
        w3_min=float(w3_min)
    )
    gwo_env.gwo.run = types.MethodType(_make_patched_run(float(a_exp)), gwo_env.gwo)

    mse_peaks_list, r2_list = [], []
    for i in range(N_GROUND_TRUTHS):
        try:
            gwo_env.reset()
            done = False
            while not done:
                done = gwo_env.step()
            mse_peaks_list.append(gwo_env.peak_mse[-1])
            r2_list.append(gwo_env.r2_map[-1])
        except Exception as e:
            print(f"  [Call {call_counter[0]}] GT {i} error: {e}")
            mse_peaks_list.append(1.0)
            r2_list.append(0.0)

    mean_mse = np.mean(mse_peaks_list)
    mean_r2  = np.mean(r2_list)
    J        = 0.5 * mean_mse + 0.5 * (1.0 - mean_r2)
    elapsed  = time.time() - t0

    print(
        f"  [{call_counter[0]:>3}/{N_CALLS}] "
        f"wolves={int(n_wolves):>2} iter={int(n_iter):>2} "
        f"w3={w3_min:.3f} a_exp={a_exp:.2f} | "
        f"MSEpk={mean_mse:.5f} R²={mean_r2:.5f} "
        f"J={J:.5f} | {elapsed:.0f}s"
    )

    history.append({
        'call':      call_counter[0],
        'n_wolves':  int(n_wolves),
        'n_iter':    int(n_iter),
        'w3_min':    float(w3_min),
        'a_exp':     float(a_exp),
        'mse_peaks': mean_mse,
        'r2_map':    mean_r2,
        'J':         J,
        'time_s':    elapsed
    })

    return J


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    print("=" * 70)
    print("  Optimización Bayesiana — GWO-IPP (scikit-optimize / gp_minimize)")
    print(f"  Calls: {N_CALLS} | Iniciales: {N_INITIAL_POINTS} | GTs: {N_GROUND_TRUTHS}")
    print(f"  Objetivo: J = 0.5·MSE_picos + 0.5·(1 − R²)")
    print(f"  Referencia: Snoek et al. (2012), NeurIPS")
    print("=" * 70)
    print(f"  Espacio:  n_wolves[5-50]  n_iter[10-60]  w3_min[0-0.3]  a_exp[1-3]")
    print(f"  Arranque: {X0}  (GWO-IPP v3)\n")

    t_total = time.time()

    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=N_CALLS,
        n_initial_points=N_INITIAL_POINTS,
        x0=X0,
        acq_func='EI',          # Expected Improvement — estándar en BO
        random_state=RANDOM_STATE,
        noise=1e-10,
        verbose=False
    )

    elapsed_total = time.time() - t_total

    best_params = dict(zip(['n_wolves','n_iter','w3_min','a_exp'], result.x))
    best_J      = result.fun

    # ── Imprimir resultado ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MEJORES HIPERPARÁMETROS")
    print("=" * 70)
    print(f"  J óptimo  : {best_J:.6f}")
    print(f"  n_wolves  : {best_params['n_wolves']}")
    print(f"  n_iter    : {best_params['n_iter']}")
    print(f"  w3_min    : {best_params['w3_min']:.4f}")
    print(f"  a_exp     : {best_params['a_exp']:.4f}")
    print(f"  Tiempo    : {elapsed_total/60:.1f} min")
    print("=" * 70)

    # ── Guardar historial ─────────────────────────────────────────────────────
    df = pd.DataFrame(history)
    df['best_J'] = df['J'].cummin()
    df.to_csv(os.path.join(OUTPUT_DIR, 'bo_trials.csv'), index=False)

    # ── Guardar mejores parámetros ────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, 'best_params.txt'), 'w') as f:
        f.write("# Mejores hiperparámetros GWO-IPP\n")
        f.write("# Método: Optimización Bayesiana con GP surrogate (scikit-optimize)\n")
        f.write("# Referencia: Snoek, J., Larochelle, H., Adams, R.P. (2012). NeurIPS.\n")
        f.write(f"# J = {best_J:.6f}  (0.5·MSE_picos + 0.5·(1-R²))\n")
        f.write(f"# N_CALLS={N_CALLS} | N_GROUND_TRUTHS={N_GROUND_TRUTHS}\n\n")
        f.write(f"GWO_N_WOLVES = {best_params['n_wolves']}\n")
        f.write(f"GWO_N_ITER   = {best_params['n_iter']}\n")
        f.write(f"GWO_W3_MIN   = {best_params['w3_min']:.4f}\n")
        f.write(f"GWO_A_EXP    = {best_params['a_exp']:.4f}\n")

    # ── Gráfica principal: convergencia + J vs params ─────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        'Optimización Bayesiana GWO-IPP — scikit-optimize / gp_minimize',
        fontweight='bold', fontsize=12
    )

    # (0,0) Convergencia
    ax = axes[0, 0]
    ax.plot(df['call'], df['J'],      'o', alpha=0.45, ms=4,
            color='steelblue', label='Evaluación')
    ax.plot(df['call'], df['best_J'], '-', lw=2,
            color='crimson',   label='Mejor acumulado')
    ax.axhline(best_J, color='crimson', ls='--', alpha=0.35)
    ax.set_xlabel('Evaluación')
    ax.set_ylabel('J')
    ax.set_title('Convergencia del objetivo')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

    # (0,1), (1,0), (1,1) — J vs n_wolves, n_iter, a_exp
    param_list = [('n_wolves','n_wolves'), ('n_iter','n_iter'), ('a_exp','a_exp')]
    positions  = [(0,1), (1,0), (1,1)]
    for (r,c), (col, lbl) in zip(positions, param_list):
        ax = axes[r, c]
        sc = ax.scatter(df[col], df['J'], c=df['call'],
                        cmap='viridis', alpha=0.7, s=30)
        ax.axvline(best_params[col], color='crimson', ls='--', lw=1.5,
                   label=f"óptimo={best_params[col]:.2f}")
        ax.set_xlabel(lbl)
        ax.set_ylabel('J')
        ax.set_title(f'J vs {lbl}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines[['top','right']].set_visible(False)
        plt.colorbar(sc, ax=ax, label='Nº evaluación', pad=0.01)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bo_convergence.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Gráfica secundaria: J vs w3_min
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sc2 = ax2.scatter(df['w3_min'], df['J'], c=df['call'],
                      cmap='viridis', alpha=0.75, s=40)
    ax2.axvline(best_params['w3_min'], color='crimson', ls='--', lw=1.5,
                label=f"óptimo={best_params['w3_min']:.4f}")
    ax2.set_xlabel('w3_min')
    ax2.set_ylabel('J')
    ax2.set_title('J vs w3_min')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.spines[['top','right']].set_visible(False)
    plt.colorbar(sc2, ax=ax2, label='Nº evaluación')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bo_w3min.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Guardado en: {OUTPUT_DIR}/")
    print(f"    bo_trials.csv  |  best_params.txt")
    print(f"    bo_convergence.png  |  bo_w3min.png")
    print(f"\n  Siguiente paso: copiar best_params.txt a config.py")
    print(f"  y agregar GWO_A_EXP como nuevo parámetro.")