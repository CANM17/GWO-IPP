
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# 8. Asignar puntos iniciales a vehículos
# ==========================
def asignar_puntos_a_vehiculos(best_path, num_vehiculos):
    posicion_actual = best_path[:num_vehiculos]
    punto_intermedio = best_path[num_vehiculos:]
    return posicion_actual, punto_intermedio

# ==========================
# 9. Clase Vehiculo
# ==========================

class Vehiculo:
    def __init__(self, punto_inicial):
        self.punto_actual = punto_inicial
        self.tiempo = 0
        self.distancia = 0


def simular_movimiento_vehiculos_mru(vehiculos, punto_intermedio, distance_matrix, velocidad=2.0, delta_t=1.0):
        for vehiculo, destino in zip(vehiculos, punto_intermedio):
            pos_actual = np.array(vehiculo.punto_actual)
            pos_destino = np.array(destino)

            direccion = pos_destino - pos_actual
            distancia_total = np.linalg.norm(direccion)
            if distancia_total == 0:
                continue

            direccion_unitaria = direccion / distancia_total
            pasos = int(np.ceil(distancia_total / (velocidad * delta_t)))

            for _ in range(pasos):
                avance = velocidad * delta_t
                nueva_pos = np.array(vehiculo.punto_actual) + avance * direccion_unitaria

                if np.linalg.norm(nueva_pos - pos_actual) >= distancia_total:
                    nueva_pos = pos_destino

                vehiculo.punto_actual = nueva_pos.tolist()
                vehiculo.distancia += avance
                vehiculo.tiempo += delta_t

                if np.allclose(nueva_pos, pos_destino):
                    break

