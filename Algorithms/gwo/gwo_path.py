"""
gwo_path.py — Grey Wolf Optimizer adaptado a Informative Path Planning (GWO-IPP)

Referencia base: Mirjalili, S., Mirjalili, S.M., Lewis, A. (2014).
"Grey Wolf Optimizer." Advances in Engineering Software, 69, 46–61.

Adaptación IPP:
- Los lobos son waypoints candidatos en el espacio del lago (no soluciones continuas)
- La fitness F es una agregación ponderada de tres objetivos derivados del GP:
    f1 = sigma_j  (maximizar incertidumbre — exploración)
    f2 = mu_j     (maximizar contaminación estimada — explotación)
    f3 = min dist a waypoints ya asignados (diversidad geográfica)
- Los pesos w1, w2, w3 son adaptativos según el estado global del GP
- La manada virtual opera por ASV: cada ASV tiene su propia instancia de búsqueda
"""

import numpy as np
import copy


class GWO:
    def __init__(self, initial_seed, n_wolves=37, n_iter=60, w3_min=0.2, a_exp=2.6):
        """
        Parámetros
        ----------
        initial_seed : int
        n_wolves     : int  — tamaño de la manada virtual (análogo a n_ants en ACO)
        n_iter       : int  — iteraciones por re-planificación (análogo a ACO)
        w3_min       : float — piso mínimo del peso de diversidad (peor caso 1)
        """
        np.random.seed(initial_seed)
        self.n_wolves = n_wolves
        self.n_iter   = n_iter
        self.w3_min   = w3_min
        self.a_exp    = a_exp

    # ─────────────────────────────────────────────
    # Interfaz pública
    # ─────────────────────────────────────────────

    def compute_adaptive_weights(self, mu_global, sigma_global, epsilon_coverage=None):
        """
        Calcula los pesos adaptativos w1, w2, w3 desde el estado global del GP.

        Parámetros
        ----------
        mu_global    : array (N,) — media GP sobre todo X_test
        sigma_global : array (N,) — desv. estándar GP sobre todo X_test
        epsilon_coverage : float o None — umbral de σ para considerar un punto
                           como 'cubierto'. Si None, usa 0.1 * max(sigma_global)

        Retorna
        -------
        w1, w2, w3 : floats normalizados que suman 1, con w3 >= w3_min
        """
        mu_arr    = np.array(mu_global).flatten()
        sigma_arr = np.array(sigma_global).flatten()

        # w1: media global de sigma — cuánta incertidumbre queda en el lago
        w1_raw = float(np.mean(sigma_arr))

        # w2: máximo de mu — cuán fuerte es la señal de contaminación detectada
        # Si el GP tiene pocas muestras, mu ≈ 0 → w2 ≈ 0 → f2 no influye (comportamiento emergente)
        w2_raw = float(np.max(mu_arr))

        # w3: fracción del lago sin cubrir — 1 - cobertura
        if epsilon_coverage is None:
            epsilon_coverage = 0.1 * float(np.max(sigma_arr)) if np.max(sigma_arr) > 0 else 0.01
        covered = np.sum(sigma_arr < epsilon_coverage) / len(sigma_arr)
        w3_raw  = float(1.0 - covered)

        # Normalizar
        total = w1_raw + w2_raw + w3_raw
        if total == 0:
            w1, w2, w3 = 1/3, 1/3, 1/3
        else:
            w1 = w1_raw / total
            w2 = w2_raw / total
            w3 = w3_raw / total

        # Piso de diversidad — peor caso 1: sigma colapsada, todos al mismo pico
        if w3 < self.w3_min:
            deficit = self.w3_min - w3
            w3 = self.w3_min
            # redistribuir el deficit proporcionalmente entre w1 y w2
            w1_w2_total = w1 + w2
            if w1_w2_total > 0:
                w1 -= deficit * (w1 / w1_w2_total)
                w2 -= deficit * (w2 / w1_w2_total)
            else:
                w1 = (1 - self.w3_min) / 2
                w2 = (1 - self.w3_min) / 2

        return w1, w2, w3

    def run(self, candidates, mu_candidates, sigma_candidates,
            assigned_waypoints, w1, w2, w3):
        """
        Ejecuta la manada virtual para un ASV y retorna el mejor waypoint.

        Parámetros
        ----------
        candidates         : np.array (M, 2) — coordenadas de waypoints candidatos
        mu_candidates      : np.array (M,)   — mu GP en cada candidato
        sigma_candidates   : np.array (M,)   — sigma GP en cada candidato
        assigned_waypoints : list de np.array — waypoints ya asignados a ASVs anteriores
                             (vacío para el primer ASV)
        w1, w2, w3         : floats — pesos adaptativos

        Retorna
        -------
        best_idx  : int — índice en candidates del waypoint elegido
        best_wolf : np.array (2,) — posición del lobo alpha al final
        """
        M = len(candidates)
        if M == 0:
            return 0, candidates[0]

        # Peor caso 3: si hay menos candidatos que ASVs, usar todos disponibles
        if M == 1:
            return 0, candidates[0]

        # Calcular fitness F para cada candidato
        F = self._compute_fitness(
            candidates, mu_candidates, sigma_candidates,
            assigned_waypoints, w1, w2, w3
        )

        # Inicializar posiciones de los lobos como puntos continuos en el espacio del lago
        # Cada lobo parte de un candidato aleatorio
        wolf_indices = np.random.choice(M, size=min(self.n_wolves, M), replace=True)
        wolves = candidates[wolf_indices].astype(float)  # (n_wolves, 2)

        # Identificar α, β, δ iniciales por fitness
        alpha_pos, beta_pos, delta_pos = self._get_leaders(candidates, F)

        # Ciclo principal de la manada
        for t in range(self.n_iter):

            # Factor de convergencia generalizado: a(t) = 2·(1 - t/T)^a_exp
            # a_exp optimizado por BO (Snoek et al. 2012): a_exp=2.6
            a = 2.0 * (1.0 - t / self.n_iter) ** self.a_exp

            # Actualizar posición de cada lobo ω
            for k in range(len(wolves)):
                wolves[k] = self._update_position(
                    wolves[k], alpha_pos, beta_pos, delta_pos, a
                )
                # Clamp al bounding box del lago (los candidatos definen el espacio válido)
                wolves[k] = self._clamp_to_bounds(wolves[k], candidates)

            # Re-evaluar fitness: cada lobo se proyecta al candidato más cercano
            wolf_candidate_indices = self._project_to_candidates(wolves, candidates)
            wolf_fitness = F[wolf_candidate_indices]

            # Actualizar α, β, δ con los mejores lobos de esta iteración
            sorted_idx = np.argsort(-wolf_fitness)  # descendente (maximizar)
            alpha_pos = wolves[sorted_idx[0]].copy()
            beta_pos  = wolves[sorted_idx[1]].copy() if len(sorted_idx) > 1 else alpha_pos
            delta_pos = wolves[sorted_idx[2]].copy() if len(sorted_idx) > 2 else beta_pos

        # Selección final: candidato más cercano al lobo α
        best_idx = self._project_to_candidates(alpha_pos.reshape(1, 2), candidates)[0]

        return best_idx, candidates[best_idx]

    # ─────────────────────────────────────────────
    # Métodos internos
    # ─────────────────────────────────────────────

    def _compute_fitness(self, candidates, mu_candidates, sigma_candidates,
                         assigned_waypoints, w1, w2, w3):
        """
        F(xj) = w1*f1_norm(xj) + w2*f2_norm(xj) + w3*f3_norm(xj)

        f1 = sigma_j
        f2 = mu_j
        f3 = min dist a waypoints ya asignados (o dist al candidato más lejano si no hay asignados)
        """
        M = len(candidates)
        f1 = np.array(sigma_candidates).flatten()
        f2 = np.array(mu_candidates).flatten()

        # f3: diversidad geográfica
        if len(assigned_waypoints) == 0:
            # Primer ASV: f3 = distancia al candidato más cercano (maximizar dispersión interna)
            f3 = np.zeros(M)
            for j in range(M):
                dists = [np.linalg.norm(candidates[j] - candidates[k])
                         for k in range(M) if k != j]
                f3[j] = min(dists) if dists else 0.0
        else:
            # ASVs siguientes: f3 = distancia mínima a waypoints ya asignados
            assigned = np.array(assigned_waypoints)  # (n_assigned, 2)
            f3 = np.array([
                np.min(np.linalg.norm(assigned - candidates[j], axis=1))
                for j in range(M)
            ])

        # Normalización min-max por objetivo (Ec. normalización)
        f1_norm = self._minmax_norm(f1)
        f2_norm = self._minmax_norm(f2)
        f3_norm = self._minmax_norm(f3)

        F = w1 * f1_norm + w2 * f2_norm + w3 * f3_norm
        return F

    def _minmax_norm(self, arr):
        """Normalización min-max. Si todos iguales, retorna array de 0.5."""
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.full_like(arr, 0.5, dtype=float)
        return (arr - mn) / (mx - mn)

    def _get_leaders(self, candidates, F):
        """Retorna posiciones de α, β, δ según fitness."""
        sorted_idx = np.argsort(-F)  # descendente
        alpha = candidates[sorted_idx[0]].astype(float)
        beta  = candidates[sorted_idx[1]].astype(float) if len(sorted_idx) > 1 else alpha.copy()
        delta = candidates[sorted_idx[2]].astype(float) if len(sorted_idx) > 2 else beta.copy()
        return alpha, beta, delta

    def _update_position(self, wolf, alpha, beta, delta, a):
        """
        Actualización estándar GWO (Mirjalili 2014, Ec. 3–13).

        X(t+1) = (X1 + X2 + X3) / 3
        """
        r1, r2 = np.random.rand(2), np.random.rand(2)
        A1 = 2 * a * r1 - a
        C1 = 2 * np.random.rand(2)
        D_alpha = np.abs(C1 * alpha - wolf)
        X1 = alpha - A1 * D_alpha

        r1, r2 = np.random.rand(2), np.random.rand(2)
        A2 = 2 * a * r1 - a
        C2 = 2 * np.random.rand(2)
        D_beta = np.abs(C2 * beta - wolf)
        X2 = beta - A2 * D_beta

        r1, r2 = np.random.rand(2), np.random.rand(2)
        A3 = 2 * a * r1 - a
        C3 = 2 * np.random.rand(2)
        D_delta = np.abs(C3 * delta - wolf)
        X3 = delta - A3 * D_delta

        return (X1 + X2 + X3) / 3.0

    def _project_to_candidates(self, wolves, candidates):
        """
        Proyecta cada lobo al candidato más cercano.
        wolves: (K, 2), candidates: (M, 2)
        Retorna indices: (K,)
        """
        wolves = np.atleast_2d(wolves)
        dists = np.linalg.norm(
            wolves[:, np.newaxis, :] - candidates[np.newaxis, :, :], axis=2
        )  # (K, M)
        return np.argmin(dists, axis=1)

    def _clamp_to_bounds(self, wolf, candidates):
        """Clamp la posición del lobo al bounding box de los candidatos."""
        mins = candidates.min(axis=0)
        maxs = candidates.max(axis=0)
        return np.clip(wolf, mins, maxs)