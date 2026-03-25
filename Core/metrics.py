"""
metrics.py — Cálculo de métricas idéntico para ACO, PSO y GWO-IPP
Todos los algoritmos llaman estas funciones. Nada de métricas fuera de aquí.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def compute_map_metrics(bench_array, mu):
    """
    Métricas sobre el mapa completo.

    Parámetros
    ----------
    bench_array : array-like (N,)
        Valores del ground truth en todos los puntos de X_test
    mu : array-like (N,)
        Media del GP predicha en todos los puntos de X_test

    Retorna
    -------
    dict con mse_map y r2_map
    """
    mu_flat = np.array(mu).flatten()
    bench_flat = np.array(bench_array).flatten()

    mse = mean_squared_error(y_true=bench_flat, y_pred=mu_flat)
    r2  = r2_score(y_true=bench_flat, y_pred=mu_flat)

    return {'mse_map': mse, 'r2_map': r2}


def compute_peak_metrics(max_bench_list, index_a, mu):
    """
    Métricas sobre los picos de contaminación.

    Parámetros
    ----------
    max_bench_list : list
        Valores del ground truth en los picos (uno por pico)
    index_a : list
        Índices en X_test de cada pico del ground truth
    mu : array-like (N,)
        Media del GP predicha en todos los puntos de X_test

    Retorna
    -------
    dict con mse_peaks y mean_abs_error_peaks
    """
    mu_flat = np.array(mu).flatten()

    errors = []
    for i in range(len(index_a)):
        mu_peak = mu_flat[round(index_a[i])]
        errors.append(abs(max_bench_list[i] - mu_peak))

    mse_peaks = np.mean(np.array(errors) ** 2)
    mae_peaks = np.mean(errors)

    return {'mse_peaks': mse_peaks, 'mae_peaks': mae_peaks}


def compute_all_metrics(bench_array, mu, max_bench_list, index_a):
    """
    Calcula todas las métricas en una sola llamada.

    Retorna
    -------
    dict con mse_map, r2_map, mse_peaks, mae_peaks
    """
    map_m  = compute_map_metrics(bench_array, mu)
    peak_m = compute_peak_metrics(max_bench_list, index_a, mu)
    return {**map_m, **peak_m}