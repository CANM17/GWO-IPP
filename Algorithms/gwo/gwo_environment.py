"""
gwo_environment.py — Entorno GWO-IPP para monitoreo del Lago Ypacaraí

Entorno similar al ACOEnvironment para garantizar comparación justa:
- Misma lógica de movimiento de ASVs
- Misma lógica de toma de mediciones
- Misma actualización del GP
- Misma generación de candidatos (create_action_zones)
- Diferencia: la selección del waypoint usa GWO-IPP en lugar de ACO

Aporte del algoritmo:
- ACO usa umbral exploration_distance para política binaria exploración/explotación
- GWO-IPP no usa umbral: el balance emerge de los pesos adaptativos del GP
"""

import numpy as np
import copy
import random

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score

import sys, os

# Raíz del proyecto autónomo — dos niveles arriba de Algorithms/gwo/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Módulos del proyecto original (viven en la raíz de YpacaraiIPP/)
from Data.limits import Limits
from Data.utils import Utils
from Environment.map import Map
from Environment.bounds import Bounds
from Environment.contamination_areas import DetectContaminationAreas
from Environment.plot import Plots
from Benchmark.benchmark_functions import Benchmark_function

# GWO-IPP
from Algorithms.gwo.gwo_path import GWO

# Métricas compartidas
from Core.metrics import compute_all_metrics

# Config compartida
from Experiments.config import (
    make_gp_kernel, GP_ALPHA, GWO_N_WOLVES, GWO_N_ITER, GWO_W3_MIN, GWO_A_EXP
)


class GWOEnvironment:

    def __init__(self, resolution, ys, initial_seed, initial_position,
                 number_of_vehicles=4, total_distance=200,
                 n_wolves=None, n_iter=None, w3_min=None, a_exp=None):

        self.resolution         = resolution
        self.number_of_vehicles = number_of_vehicles
        self.total_distance     = total_distance
        self.initial_position   = initial_position
        self.initial_seed       = initial_seed
        self.seed               = initial_seed

        self.xs = int(10000 / (15000 / ys))
        self.ys = ys

        # GP — misma configuración que ACO y PSO
        self.gpr = GaussianProcessRegressor(
            kernel=make_gp_kernel(), alpha=GP_ALPHA
        )

        # Hiperparámetros GWO — usan config.py por defecto, sobreescribibles
        _n_wolves = n_wolves if n_wolves is not None else GWO_N_WOLVES
        _n_iter   = n_iter   if n_iter   is not None else GWO_N_ITER
        _w3_min   = w3_min   if w3_min   is not None else GWO_W3_MIN
        _a_exp    = a_exp    if a_exp    is not None else GWO_A_EXP

        # GWO
        self.gwo = GWO(
            initial_seed=initial_seed,
            n_wolves=_n_wolves,
            n_iter=_n_iter,
            w3_min=_w3_min,
            a_exp=_a_exp
        )

        # Movimiento
        self.vel_max = 2
        self.lam     = 0.3

        # Métricas acumuladas por corrida
        self.map_mse  = []
        self.peak_mse = []
        self.r2_map   = []

        # Mapa
        self.grid_or = Map(self.xs, ys).black_white()
        self.grid_min = 0
        self.df_bounds, self.X_test, self.bench_limits = Bounds(
            resolution, self.xs, ys, load_file=False
        ).map_bound()
        self.secure, self.df_bounds = Bounds(resolution, self.xs, ys).interest_area()

        self.util = Utils(number_of_vehicles)

    # ─────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────

    def reset(self):
        self._reset_variables()

        self.bench_function, self.bench_array, self.num_of_peaks, self.index_a = \
            Benchmark_function(
                self.grid_or, self.resolution, self.xs, self.ys,
                self.X_test, self.seed, self.number_of_vehicles
            ).create_new_map()

        self.bench_max, self.coordinate_bench_max = self._obtain_max(self.bench_array)

        random.seed(self.seed)
        self._peaks_bench()

        self.detect_areas = DetectContaminationAreas(
            self.X_test, self.bench_array,
            vehicles=self.number_of_vehicles, area=self.xs
        )
        self.centers_bench, self.dict_index_bench, self.dict_bench, \
        self.dict_coord_bench, self.center_peaks_bench, self.max_bench_list, \
        self.dict_limits_bench, self.action_zone_bench, self.dict_impo_bench, \
        self.index_center_bench = self.detect_areas.benchmark_areas()

        self.max_peaks = self.detect_areas.real_peaks()
        self._first_measurement()

    def _reset_variables(self):
        self.x_h               = []
        self.y_h               = []
        self.measurement       = 0
        self.measurements_vector = []
        self.mu                = []
        self.sigma             = []
        self.post_array        = np.ones(self.number_of_vehicles)
        self.distances         = np.zeros(self.number_of_vehicles)
        self.part_ant          = np.zeros((1, self.number_of_vehicles * 2))
        self.array_part        = np.zeros((1, self.number_of_vehicles * 2))
        self.duplicate         = False
        self.seed             += 1
        self.bench_array       = []
        self.num_of_peaks      = 0
        self.index_a           = []
        self.vehicles          = copy.copy(self.initial_position[:self.number_of_vehicles])
        self.samples           = 0
        self.g                 = 0
        self.max_peaks_bench   = []
        self.s_n               = np.full(self.number_of_vehicles, True)
        self.n_data            = 0
        self.s_ant             = np.zeros(self.number_of_vehicles)
        self.f                 = 0
        self.last_sample       = 0
        self.next_vehicle_positions = []
        self.error_map         = 0
        self.error_peak        = []
        self.mean_error_peak   = 0
        self.epsilon_coverage  = None  # se fija en la primera llamada a pesos

    # ─────────────────────────────────────────────
    # Loop principal
    # ─────────────────────────────────────────────

    def step(self):
        return self._step_stage()

    def _step_stage(self):
        self.dist_pre = np.mean(self.distances)
        self.n_data   = 0
        self.f       += 1
        done          = False

        while not done:
            self.dist_pre = np.mean(self.distances)
            self.n_data   = 0

            # — Mover ASVs un paso hacia su destino —
            for i, vehicle_position in enumerate(self.vehicles):
                pos_actual  = np.array(vehicle_position)
                pos_destino = np.array(self.next_vehicle_positions[i])
                direccion   = pos_destino - pos_actual
                distancia_total = np.linalg.norm(direccion)

                if distancia_total == 0:
                    velocidad_aplicada = 0
                else:
                    dir_unit = direccion / distancia_total
                    velocidad_aplicada = (
                        np.minimum(self.vel_max, np.abs(direccion)) * np.sign(dir_unit)
                    )

                nueva_pos = pos_actual + velocidad_aplicada
                self.vehicles[int(self.n_data)] = nueva_pos.tolist()

                self.part_ant, self.distances = self.util.distance_part(
                    self.g, self.n_data, vehicle_position,
                    self.part_ant, self.distances, self.array_part, dfirst=False
                )
                self.n_data += 1
                if self.n_data > self.number_of_vehicles - 1:
                    self.n_data = 0

            self.g += 1

            # — Tomar medición si corresponde —
            if (np.mean(self.distances) - self.last_sample) >= (
                np.min(self.post_array) * self.lam
            ):
                self.last_sample = np.mean(self.distances)
                for i, vehicle_position in enumerate(self.vehicles):
                    vehicle_position, measurement = self._new_measurement(vehicle_position)
                    self._check_duplicate(vehicle_position, measurement)
                    self.post_array = self._gp_regression()
                    self.samples   += 1
                    self.n_data    += 1
                    if self.n_data > self.number_of_vehicles - 1:
                        self.n_data = 0

            # — Re-planificar si algún ASV llegó a su destino —
            vehiculos_llegaron = [
                (np.array(self.vehicles[i]) == np.array(self.next_vehicle_positions[i])).all()
                for i in range(len(self.vehicles))
            ]

            if any(vehiculos_llegaron):
                self._plan_next_waypoints()

            self.n_data += 1
            if self.n_data > self.number_of_vehicles - 1:
                self.n_data = 0

            # — Criterio de parada —
            if (np.max(self.distances) >= self.total_distance) or \
               (np.mean(self.distances) == self.dist_pre):
                self._error_calculation()
                done = True

        return done

    # ─────────────────────────────────────────────
    # Planificación GWO-IPP
    # ─────────────────────────────────────────────

    def _plan_next_waypoints(self):
        """
        Re-planificación con GWO-IPP.
        Espejo de la lógica ACO en first_measurement y step_stage,
        con GWO reemplazando la selección del waypoint.
        """
        self.next_vehicle_positions = []

        # Candidatos desde el GP actualizado
        candidates = self._create_action_zones()

        # Peor caso 3: si no hay candidatos suficientes, completar con max sigma global
        candidates = self._ensure_candidates(candidates)

        # Pesos adaptativos desde el estado global del GP
        # epsilon_coverage se fija una sola vez en t=0 como 0.1 * sigma_max inicial
        if self.epsilon_coverage is None and len(self.sigma) > 0:
            self.epsilon_coverage = 0.1 * float(np.max(self.sigma))

        w1, w2, w3 = self.gwo.compute_adaptive_weights(
            self.mu, self.sigma, self.epsilon_coverage
        )

        # mu y sigma en los candidatos
        mu_cand, sigma_cand = self._get_gp_at_candidates(candidates)

        # Asignación secuencial — igual que ACO
        assigned_waypoints = []
        candidates_remaining = candidates.copy()
        mu_remaining         = mu_cand.copy()
        sigma_remaining      = sigma_cand.copy()

        for i in range(self.number_of_vehicles):
            if len(candidates_remaining) == 0:
                # Fallback: quedarse en posición actual
                self.next_vehicle_positions.append(
                    np.array(self.vehicles[i])
                )
                continue

            best_idx, best_wp = self.gwo.run(
                candidates=candidates_remaining,
                mu_candidates=mu_remaining,
                sigma_candidates=sigma_remaining,
                assigned_waypoints=assigned_waypoints,
                w1=w1, w2=w2, w3=w3
            )

            self.next_vehicle_positions.append(best_wp)
            assigned_waypoints.append(best_wp)

            # Eliminar candidato elegido (igual que ACO)
            candidates_remaining = np.delete(candidates_remaining, best_idx, axis=0)
            mu_remaining         = np.delete(mu_remaining, best_idx)
            sigma_remaining      = np.delete(sigma_remaining, best_idx)

    def _create_action_zones(self):
        """
        Genera candidatos desde el GP — igual que ACOEnvironment.create_action_zones()
        pero SIN el umbral exploration_distance.

        GWO-IPP usa siempre candidatos de mu Y sigma combinados.
        El balance exploración/explotación emerge de los pesos w1, w2, w3.
        """
        coord_mu, _    = self.detect_areas.areas_levels(self.mu,    self.X_test, True)
        coord_sigma, _ = self.detect_areas.areas_levels(self.sigma, self.X_test, False)
        return np.vstack([coord_mu, coord_sigma])

    def _ensure_candidates(self, candidates):
        """
        Peor caso 3: si hay menos candidatos que ASVs,
        completar con puntos de máxima sigma global.
        """
        if len(candidates) >= self.number_of_vehicles:
            return candidates

        sigma_arr = np.array(self.sigma).flatten()
        sorted_idx = np.argsort(-sigma_arr)
        extra = []
        for idx in sorted_idx:
            pt = np.array(self.X_test[idx])
            already = any(
                np.linalg.norm(pt - c) < 1e-3 for c in candidates
            )
            if not already:
                extra.append(pt)
            if len(candidates) + len(extra) >= self.number_of_vehicles:
                break

        if extra:
            candidates = np.vstack([candidates, np.array(extra)])
        return candidates

    def _get_gp_at_candidates(self, candidates):
        """
        Extrae mu y sigma del GP para cada candidato.
        """
        mu_arr    = np.array(self.mu).flatten()
        sigma_arr = np.array(self.sigma).flatten()
        X_test_arr = np.array(self.X_test)

        mu_cand    = np.zeros(len(candidates))
        sigma_cand = np.zeros(len(candidates))

        for j, cand in enumerate(candidates):
            dists = np.linalg.norm(X_test_arr - cand, axis=1)
            nearest = np.argmin(dists)
            mu_cand[j]    = mu_arr[nearest]
            sigma_cand[j] = sigma_arr[nearest]

        return mu_cand, sigma_cand

    # ─────────────────────────────────────────────
    # Medición y GP — idéntico a ACOEnvironment
    # ─────────────────────────────────────────────

    def _first_measurement(self):
        for i, vehicle_position in enumerate(self.vehicles):
            vehicle_position, measurement = self._new_measurement(vehicle_position)
            self.part_ant, self.distances = self.util.distance_part(
                self.g, self.n_data, vehicle_position,
                self.part_ant, self.distances, self.array_part, dfirst=True
            )
            self._check_duplicate(vehicle_position, measurement)
            self.post_array = self._gp_regression()
            self.samples   += 1
            self.n_data    += 1
            if self.n_data > self.number_of_vehicles - 1:
                self.n_data = 0

        self._plan_next_waypoints()

    def _new_measurement(self, vehicle_position):
        vehicle_position, self.s_n = Limits(
            self.secure, self.xs, self.ys
        ).new_limit(self.g, vehicle_position, self.s_n, self.n_data,
                    self.s_ant, self.part_ant)

        x_bench = int(vehicle_position[0])
        y_bench = int(vehicle_position[1])
        measurement = [self.bench_function[x_bench][y_bench]]
        self._check_duplicate(vehicle_position, measurement)
        return vehicle_position, measurement

    def _gp_regression(self):
        x_a = np.array(self.x_h).reshape(-1, 1)
        y_a = np.array(self.y_h).reshape(-1, 1)
        x_train = np.concatenate([x_a, y_a], axis=1)
        y_train = np.array(self.measurements_vector).reshape(-1, 1)

        self.gpr.fit(x_train, y_train)
        self.mu, self.sigma = self.gpr.predict(self.X_test, return_std=True)

        post_ls = round(np.min(np.exp(self.gpr.kernel_.theta[0])), 1)
        r = self.n_data
        self.post_array[r] = post_ls
        return self.post_array

    def _check_duplicate(self, vehicle_position, measurement):
        x = int(vehicle_position[0])
        y = int(vehicle_position[1])
        self.duplicate = False
        for i in range(len(self.x_h)):
            if self.x_h[i] == x and self.y_h[i] == y:
                self.duplicate = True
                self.measurements_vector[i] = measurement
                return
        self.x_h.append(x)
        self.y_h.append(y)
        self.measurements_vector.append(measurement)

    # ─────────────────────────────────────────────
    # Métricas — usa Core/metrics.py
    # ─────────────────────────────────────────────

    def _error_calculation(self):
        results = compute_all_metrics(
            self.bench_array, self.mu,
            self.max_bench_list, self.index_a
        )
        self.map_mse.append(results['mse_map'])
        self.r2_map.append(results['r2_map'])
        self.peak_mse.append(results['mse_peaks'])

    def print_error(self):
        print("MSE peaks:", np.mean(self.peak_mse), '+-', np.std(self.peak_mse) * 1.96)
        print("MSE map:",   np.mean(self.map_mse),  '+-', np.std(self.map_mse)  * 1.96)
        print("R2 map:",    np.mean(self.r2_map),   '+-', np.std(self.r2_map)   * 1.96)

    # ─────────────────────────────────────────────
    # Utilidades
    # ─────────────────────────────────────────────

    def _obtain_max(self, array_function):
        max_value = np.max(array_function)
        index_1   = np.where(array_function == max_value)
        index_x2  = index_1[0][0]
        index_x   = int(self.X_test[index_x2][0])
        index_y   = int(self.X_test[index_x2][1])
        return max_value, np.array([index_x, index_y])

    def _peaks_bench(self):
        self.max_peaks_bench = []
        for i in range(len(self.index_a)):
            self.max_peaks_bench.append(self.bench_array[round(self.index_a[i])])

    def data_out(self):
        return (self.X_test, self.secure, self.bench_function, self.grid_min,
                self.sigma, self.mu, self.part_ant, self.bench_array,
                self.grid_or, self.bench_max)

    def return_seed(self):
        return self.seed