from Data.limits import Limits
from Environment.map import Map
from Benchmark.benchmark_functions import Benchmark_function
# from Benchmark.bench_functions import *
from Environment.bounds import Bounds
from Data.utils import Utils
from Environment.contamination_areas import DetectContaminationAreas
from Environment.plot import Plots

from PathPlanning.aco_path import ACO
from PathPlanning.mov_vehiculo import simular_movimiento_vehiculos_mru
from PathPlanning.mov_vehiculo import asignar_puntos_a_vehiculos
from PathPlanning.mov_vehiculo import Vehiculo

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
import random
import math

import copy

from statistics import mean
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker


from deap.benchmarks import shekel


import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

"""[https://deap.readthedocs.io/en/master/examples/pso_basic.html]"""


class ACOEnvironment():

    def __init__(self, resolution, ys, method, initial_seed, initial_position, number_of_vehicles=4, exploration_distance=200,
                 total_distance=200,type_error='all_map', stage='exploration'):
        self.measurement = 0
        self.measurements_vector = []
        self.vehicle_position = []
        self.resolution = resolution
        self.number_of_vehicles = number_of_vehicles
        self.exploration_distance = exploration_distance
        self.map_mse = []
        self.peak_mse = []
        self.xs = int(10000 / (15000 / ys))
        self.ys = ys
        ker = RBF(length_scale=10, length_scale_bounds=(1e-1, 10))
        self.gpr = GaussianProcessRegressor(kernel=ker, alpha=1e-6)  # optimizer=None)
        self.x_h = []
        self.y_h = []
        self.array_part = np.zeros((1, self.number_of_vehicles * 2))
        self.initial_position = initial_position
        self.type_error = type_error
        self.initial_stage = stage
        self.exploration_distance_initial = exploration_distance

        self.exploration_distance = exploration_distance

        self.stage = stage
        self.vel_min = 0
        self.vel_max = 2
        self.size_position = 2
        self.seed = initial_seed
        self.initial_seed = initial_seed
        self.mu = []
        self.mean_error_peak = 0
        self.sigma = []
        self.post_array = np.ones((1, self.number_of_vehicles))
        self.distances = np.zeros(self.number_of_vehicles)
        self.lam = 0.3
        self.duplicate = False
        self.bench_array, self.num_of_peaks, self.index_a = [], 0, []
        self.vehicles = []
        self.bench_function = None
        self.error_map = 0
        self.error_peak = []
        self.total_distance = total_distance
        self.r2_map = []

        self.s_n = np.full(self.number_of_vehicles, True)
        self.n_data = 0
        self.s_ant = np.zeros(self.number_of_vehicles)
        self.n_plot = float(1)
        self.f = None
        self.last_sample = 0
        self.next_vehicle_positions = []


        self.grid_or = Map(self.xs, ys).black_white()
        self.grid_min, self.grid_max, self.grid_max_x, self.grid_max_y = 0, self.ys, self.xs, self.ys
        self.df_bounds, self.X_test, self.bench_limits = Bounds(self.resolution, self.xs, self.ys,
                                                                load_file=False).map_bound()
        self.secure, self.df_bounds = Bounds(self.resolution, self.xs, self.ys).interest_area()

        self.plot = Plots(self.xs, self.ys, self.X_test, self.secure, self.bench_function, self.grid_min, self.grid_or,
                          self.stage)
        self.util = Utils(self.number_of_vehicles)
        self.aco = ACO(initial_seed)

    def reset(self):

        """
        Inicializa los mapas de benchmark, áreas contaminadas para obtener los picos de contaminación y las variables
        """
        self.reset_variables()
        self.bench_function, self.bench_array, self.num_of_peaks, self.index_a = Benchmark_function(self.grid_or,
                                                                                                    self.resolution,
                                                                                                    self.xs, self.ys,
                                                                                                    self.X_test,
                                                                                                    self.seed, self.number_of_vehicles).create_new_map()


        self.max_contamination()


        random.seed(self.seed)

        self.peaks_bench()
        self.detect_areas = DetectContaminationAreas(self.X_test, self.bench_array, vehicles=self.number_of_vehicles,
                                                     area=self.xs)
        self.centers_bench, self.dict_index_bench, self.dict_bench, self.dict_coord_bench, self.center_peaks_bench, \
        self.max_bench_list, self.dict_limits_bench, self.action_zone_bench, self.dict_impo_bench, \
        self.index_center_bench = self.detect_areas.benchmark_areas()
        self.max_peaks = self.detect_areas.real_peaks()
        self.first_measurement()

    def reset_variables(self):

        """
        Resetea las variables
        """

        self.x_h = []
        self.y_h = []
        self.stage = self.initial_stage
        self.exploration_distance = self.exploration_distance_initial

        self.measurement = 0
        self.measurements_vector = []
        self.vehicle_position = []
        self.mu = []
        self.sigma = []
        self.post_array = np.ones(self.number_of_vehicles)
        self.distances = np.zeros(self.number_of_vehicles)
        self.part_ant = np.zeros((1, self.number_of_vehicles * 2))
        self.duplicate = False
        self.array_part = np.zeros((1, self.number_of_vehicles * 2))
        self.seed += 1
        self.bench_array, self.num_of_peaks, self.index_a = [], 0, []
        self.vehicles = copy.copy(self.initial_position[:self.number_of_vehicles])
        self.samples = 0
        self.g = 0
        self.max_peaks_bench = list()
        self.s_n = np.full(self.number_of_vehicles, True)
        self.n_data = 0
        self.s_ant = np.zeros(self.number_of_vehicles)
        self.n_plot = float(1)
        self.f = 0
        self.last_sample = 0
        self.next_vehicle_positions = []
        self.error_map = 0
        self.error_peak = []
        self.mean_error_peak = 0


    def max_contamination(self):

        """
        Obtiene el punto máximo de contaminación del graund truth
        """
        self.bench_max, self.coordinate_bench_max = self.obtain_max(self.bench_array)

    def new_measurement(self, vehicle_position):

        """
        Obtiene las mediciones de los vehículos en la posición (x,y)
        """

        vehicle_position, self.s_n = Limits(self.secure, self.xs, self.ys).new_limit(self.g, vehicle_position, self.s_n, self.n_data,
                                                                         self.s_ant, self.part_ant)
        self.x_bench = int(vehicle_position[0])
        self.y_bench = int(vehicle_position[1])

        measurement = [self.bench_function[self.x_bench][self.y_bench]]
        self.check_duplicate(vehicle_position, measurement)
        # fit = self.toolbox.evaluate(part)
        return vehicle_position, measurement

    def gp_regression(self):

        """
        Ajusta y actualiza el proceso Gaussiano
        """

        x_a = np.array(self.x_h).reshape(-1, 1)
        y_a = np.array(self.y_h).reshape(-1, 1)
        x_train = np.concatenate([x_a, y_a], axis=1).reshape(-1, 2)
        y_train = np.array(self.measurements_vector).reshape(-1, 1)

        self.gpr.fit(x_train, y_train)
        self.gpr.get_params()

        self.mu, self.sigma = self.gpr.predict(self.X_test, return_std=True)
        post_ls = round(np.min(np.exp(self.gpr.kernel_.theta[0])), 1)
        r = self.n_data
        self.post_array[r] = post_ls
        if self.post_array[-1] == 0.1:
            self.len_scale += 1
            self.fail = True
            print("Warning: reached the minimum value of length scale (Exploration)")
        else:
            self.len_scale = 0

        return self.post_array

    def sort_index(self, array, rev=True):

        """
        Calcula un índice aleatorio
        """

        index = range(len(array))
        s = sorted(index, reverse=rev, key=lambda i: array[i])
        return s

    def peaks_bench(self):

        """
        Obtiene las posiciones de los picos de contaminación del ground truth
        """

        for i in range(len(self.index_a)):
            self.max_peaks_bench.append(self.bench_array[round(self.index_a[i])])

    def peaks_mu(self):

        """
        Obtiene los valores estimados de los modelos del GP en los picos de contaminación
        """

        self.max_peaks_mu = list()
        for i in range(len(self.index_a)):
            self.max_peaks_mu.append(self.mu[round(self.index_a[i])])

    def check_duplicate(self, vehicle_position, measurement):

        """
        Valida las mediciones controlando si hay datos duplicados
        """

        self.duplicate = False
        for i in range(len(self.x_h)):
            if self.x_h[i] == self.x_bench and self.y_h[i] == self.y_bench:
                self.duplicate = True
                self.measurements_vector[i] = measurement
                break
            else:
                self.duplicate = False
        if self.duplicate:
            pass
        else:
            self.x_h.append(int(vehicle_position[0]))
            self.y_h.append(int(vehicle_position[1]))
            self.measurements_vector.append(measurement)

    def first_measurement(self):
        """
        Primera medición en la posición inicial
        """
        # Realizamos la medición inicial para cada vehículo
        for i, vehicle_position in enumerate(self.vehicles):
            vehicle_position, measurement = self.new_measurement(vehicle_position)

            if self.n_plot > self.number_of_vehicles:
                self.n_plot = float(1)

            self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, vehicle_position,
                                                                    self.part_ant,
                                                                    self.distances, self.array_part, dfirst=True)

            self.check_duplicate(vehicle_position, measurement)

            self.post_array = self.gp_regression()

            self.samples += 1

            self.n_data += 1
            if self.n_data > self.number_of_vehicles - 1:
                self.n_data = 0

        # Crear zonas de acción llamando a la función que ya está definida en create_action_zones
        center_action_zones_coord = self.create_action_zones()  # Esto crea las zonas de acción para los vehículos

        # Aplicar ACO para planificar las rutas de los vehículos
        self.next_vehicle_positions = []
        next_index = None

        for i, vehicle_position in enumerate(self.vehicles):
            # Concatenamos la posición actual del vehículo con las zonas de acción generadas
            points_for_aco = np.vstack([vehicle_position,
                                        center_action_zones_coord])  # Concatena el punto actual del vehículo con los puntos de la zona de acción para el ACO
            #print(f"Vehículo {i + 1} va a los nodos: {points_for_aco}")

            self.aco.initialize_aco_params(points_for_aco)  # Inicializa ACO con los puntos de acción
            best_path, best_cost = self.aco.run()  # Ejecuta ACO para encontrar el mejor camino

            # Asegúrate de que next_index esté dentro del rango válido (de 0 a len(center_action_zones_coord) - 1)
            next_index = best_path[1]  # El primer índice siempre es 0, así que tomamos el siguiente (índice 1)
            #next_index = next_index % len(
             #   center_action_zones_coord)  # Asegúrate de que next_index esté dentro del rango válido

            # Determina la próxima posición del vehículo según el mejor camino encontrado
            next_position = points_for_aco[next_index]
            #print(f"Vehículo {i + 1} irá a {next_position}")

            # Almacenamos la siguiente posición para cada vehículo
            self.next_vehicle_positions.append(next_position)

            # Eliminamos el punto elegido para evitar duplicados en los siguientes vehículos
            center_action_zones_coord = np.delete(center_action_zones_coord, next_index - 1, axis=0)

        # Aquí puedes agregar más lógica si es necesario

    def obtain_max(self, array_function):

        """
        Obtiene los valores máximos y su coordenada
        """

        max_value = np.max(array_function)
        index_1 = np.where(array_function == max_value)
        index_x1 = index_1[0]

        index_x2 = index_x1[0]
        index_x = int(self.X_test[index_x2][0])
        index_y = int(self.X_test[index_x2][1])

        index_xy = [index_x, index_y]
        coordinate_max = np.array(index_xy)

        return max_value, coordinate_max

    def calculate_error(self, dfirts=False):
        if self.type_error == 'all_map':
            self.error_map = mean_squared_error(y_true=self.bench_array, y_pred=self.mu)
            r2 = r2_score(y_true=self.bench_array, y_pred=self.mu)

        elif self.type_error == 'contamination':
            self.peaks_mu()
            for i in range(len(self.index_a)):
                self.error_peak.append(abs(self.max_bench_list[i] - self.max_peaks_mu[i]))
            self.mean_error_peak = np.mean(self.error_peak)


    def allocate_vehicles(self):
        """
        Asignar los vehículos a los puntos de las zonas de acción
        """

    def create_action_zones(self):

        # Creación de zonas de acción

        self.coord_centers_mu, self.data_zone_mu = self.detect_areas.areas_levels(self.mu, self.X_test,
                                                                                  True)  # Zonas de acción de
        # la media (mu)
        self.coord_centers_sigma, self.data_zone_sigma = self.detect_areas.areas_levels(self.sigma, self.X_test,
                                                                                        False)  # Zonas de
        # acción de la incertidumbre (sigma)
        self.coord_centers_sigma = self.coord_centers_sigma
        #print(self.coord_centers_mu, self.coord_centers_sigma)
        if np.mean(self.distances) >= self.exploration_distance:
            center_action_zones_coord = np.vstack([self.coord_centers_mu, self.coord_centers_sigma])  #
        else:
            center_action_zones_coord = copy.copy(self.coord_centers_sigma)
            # Concatena los dos vectores para tener en uno solo
        if self.seed == 1000010:
            self.plot.plot_classic(self.mu, self.sigma, self.part_ant)
        return center_action_zones_coord

    def step_stage(self):

        """
        Sección principal del código
        """

        dis_steps = 0
        dist_ant = np.mean(self.distances)
        self.dist_pre = np.mean(self.distances)
        self.n_data = 0
        self.f += 1
        done = False

        while not done: #Cada 1000 metros recorrido
            self.dist_pre = np.mean(self.distances)
            previous_dist = np.mean(self.distances)
            self.n_data = 0
            # Mover vehículo a la posición que se calculo con el ACO
            ### Agregar algoritmo de movimiento de vehiculo---------------------------

            for i, vehicle_position in enumerate(self.vehicles):

                pos_actual = np.array(vehicle_position)  # Posición actual del vehículo
                pos_destino = np.array(self.next_vehicle_positions[i])  # La siguiente posición (destino)

                # Calcular la dirección y la distancia
                direccion = pos_destino - pos_actual
                distancia_total = np.linalg.norm(direccion)

                # Si la distancia total no es 0 (es decir, hay un destino)

                # Normalizar la dirección para obtener el vector unitario
                if distancia_total == 0:
                    velocidad_aplicada = 0
                else:
                    direccion_unitaria = direccion / distancia_total

                    # Aplicar el límite de velocidad en ambas dimensiones (x y y)
                    velocidad_aplicada = np.minimum(self.vel_max, np.abs(direccion)) * np.sign(direccion_unitaria)

                # Calcular la nueva posición, sumando el avance al punto actual
                nueva_pos = pos_actual + velocidad_aplicada

                # Actualizar la posición en self.vehicles (actualización de la posición del vehículo i)
                self.vehicles[int(self.n_data)] = nueva_pos.tolist()
                print(self.distances)

                self.part_ant, self.distances = self.util.distance_part(self.g, self.n_data, vehicle_position,
                                                                        self.part_ant,
                                                                        self.distances, self.array_part, dfirst=False)

                self.n_data += 1
                if self.n_data > self.number_of_vehicles - 1:
                    self.n_data = 0

            self.g += 1

                #mover vehiculo
                #-----------------------------------------------------------------------------
                # Asegurar que part_ant tiene suficientes filas para acceder con índice `g`
               # if self.part_ant.shape[0] <= self.g:
                #    nueva_fila = self.part_ant[-1:].copy()  # duplicar la última fila
                 #   self.part_ant = np.vstack([self.part_ant, nueva_fila])
                    # Calcula
                # la distancia recorrida por los vehículos

            #-------------------------------------------------------------------------
            if (np.mean(self.distances) - self.last_sample) >= (np.min(self.post_array) * self.lam): #Calcular la distancia
                #para toma de medidas
                self.ok = True
                self.last_sample = np.mean(self.distances)

                for i, vehicle_position in enumerate(self.vehicles):
                    vehicle_position, measurement = self.new_measurement(vehicle_position) #toma de medidas

                    self.check_duplicate(vehicle_position, measurement)

                    self.post_array = self.gp_regression() #actualización de gp

                    self.samples += 1

                    self.n_data += 1
                    if self.n_data > self.number_of_vehicles - 1:
                        self.n_data = 0

            #if any  # los vehiculos ya llegaron al siguiente punto? (ya llegaron a los centros de las zonas de accion)
            vehiculos_llegaron = []
            # Verificar si algún vehículo ya llegó a su destino
            for i, vehicle_position in enumerate(self.vehicles):
                if (vehicle_position == self.next_vehicle_positions[i]).all():
                    vehiculos_llegaron.append(True)
                else:
                    vehiculos_llegaron.append(False)

            if any(vehiculos_llegaron):

                 # Ejecutar nueva planificación ACO, # si ya llegaron, calcular nuevamente al aco

                self.next_vehicle_positions = []

                # Crear zonas de acción
                center_action_zones_coord = self.create_action_zones()

                # Aplicar ACO

                for i, vehicle_position in enumerate(self.vehicles):
                    points_for_aco = []
                    points_for_aco = np.vstack([vehicle_position, center_action_zones_coord])  # Concatena el
                    # punto actual del vehículo con los puntos de la zona de acción para el ACO
                    self.aco.initialize_aco_params(points_for_aco)
                    best_path, best_cost = self.aco.run()

                    # best_path --- el elemento del indice 1
                    # Siguiente posicion al que debe ir el vehiculo points for aco [dato de linea 298]
                    # elimino el punto donde debe ir el vehiculo n
                    # vector con next_vehicle_position  ---- agregar los puntos a los que deben ir los vehiculos
                    #if len(best_path) < 2:
                     #   print(
                      #      f"[AVISO] El camino generado para el vehículo {i + 1} es demasiado corto: {best_path}")
                      #  continue  # Salta a la siguiente iteración del bucle

                    next_index = best_path[1]

                    next_index = next_index % len(center_action_zones_coord) # Asegúrate de que next_index esté dentro del rango válido (de 0 a len(center_action_zones_coord) - 1)
                    next_position = points_for_aco[next_index]

                    #print(f"Vehículo {i + 1} irá a {next_position}")

                    self.next_vehicle_positions.append(next_position)

                    # Eliminamos el punto elegido para evitar duplicados en los siguientes vehículos
                    center_action_zones_coord = np.delete(center_action_zones_coord, next_index, axis=0)

            # Crear 8 nodos de acción para que los vehículos los recorran


            # Determinar punto (x,y) donde ira el vehiculo i
            # Agregar (x,y) a un vector self.siguientes_puntos
            # Eliminar el punto (x,y) del vector center_action_zones_coord

            self.n_data += 1
            if self.n_data > self.number_of_vehicles - 1:
                self.n_data = 0



            if (np.max(self.distances) >= self.total_distance) or np.mean(self.distances) == self.dist_pre:
                self.error_calculation()
                done = True
                #if max(self.mu) < 0.33:
                    #done = True
                 #   self.exploration_distance = self.exploration_distance + self.exploration_distance_initial
                  #  self.exploitation_distance = self.exploitation_distance_initial - self.exploration_distance_initial
                #else:
                    #self.plot.movement_exploration(self.mu, self.sigma, self.part_ant_explore)
        return done

    def step(self):
        done = self.step_stage()
            # if self.init_fig:
            #     self.fig = plt.figure(figsize=(8, 8))
            #
            # dict_matrix_mu = {}
            # dict_matrix_sigma = {}
            # for i in range(len(self.dict_centers)):
            #     dict_matrix_sigma["action_zone%s" % i], dict_matrix_mu["action_zone%s" % i] = self.plot.Z_var_mean(
            #         self.dict_mu["action_zone%s" % i], self.dict_sigma["action_zone%s" % i])
            # for j in range(len(self.assig_centers)):
            #     x = 2 * j
            #     y = 2 * j + 1
            #     zone = int(self.assig_centers[j])
            #     cols = round(self.vehicles / 2)
            #     rows = self.vehicles // cols
            #     rows += self.vehicles % cols
            #     position = range(1, self.vehicles + 1)
            #     initial_x = self.part_ant_exploit[0, x]
            #     final_x = self.part_ant_exploit[-1, x]
            #     initial_y = self.part_ant_exploit[0, y]
            #     final_y = self.part_ant_exploit[-1, y]
            #
            #     self.axs = self.fig.add_subplot(rows, cols, position[zone])
            #     if self.init_fig:
            #         cols = round(self.vehicles / 2)
            #         rows = self.vehicles // cols
            #         rows += self.vehicles % cols
            #         position = range(1, self.vehicles + 1)
            #
            #
            #         # self.axs.plot(initial_x, initial_y, 'x', color='black', markersize=4,
            #         #               label='Exploitation initial position')
            #
            #         # self.axs.legend(loc=3, fontsize=6)
            #
            #         self.axs.set_xlabel("x [m]")
            #         self.axs.set_ylabel("y [m]")
            #         self.axs.set_title("Action Zone %s" % str(zone))
            #         self.axs.set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
            #         self.axs.set_xticks([0, 50, 100])
            #         self.axs.set_aspect('equal')
            #         self.axs.set_ylim([self.ys, 0])
            #         self.axs.grid(True)
            #         ticks_x = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
            #         self.axs.xaxis.set_major_formatter(ticks_x)
            #
            #         ticks_y = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
            #         self.axs.yaxis.set_major_formatter(ticks_y)
            #         bottom, top = 0.1, 1.5
            #         left, right = 0.1, 2
            #     # self.axs.plot(final_x, final_y, 'X', color='red', markersize=3, label='Exploitation final position')
            #     # self.plot.plot_trajectory_classic(self.axs, self.part_ant_exploit[:, x], self.part_ant_exploit[:, y],
            #     #                              colormap=self.plot.colors[j])
            #     matrix_sigma = copy.copy(dict_matrix_sigma["action_zone%s" % zone])
            #     matrix_mu = copy.copy(dict_matrix_mu["action_zone%s" % zone])
            #     # im = self.axs.imshow(matrix_sigma.T, interpolation='bilinear', origin='lower', cmap="gist_yarg", vmin=0,
            #     #                  vmax=1.0)
            #     im = self.axs.imshow(matrix_mu.T, interpolation='bilinear', origin='lower', cmap=self.plot.cmapmean, vmin=0,
            #                     vmax=1.0)
            #
            # if self.init_fig:
            #
            #
            #     self.fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
            #     cbar_ax = self.fig.add_axes([0.85, bottom, 0.025, 0.85])
            #     # self.fig.colorbar(im, cax=cbar_ax, label='σ', shrink=1.0)
            #     self.fig.colorbar(im, cax=cbar_ax, label='µ', shrink=1.0)
            #
            #     plt.tight_layout()
            #     self.init_fig = False
            #     plt.pause(6)
            # else:
            #     plt.pause(0.5)
            #
            # plt.show()


            # self.final_gaussian()
            # self.plot.movement_exploration_(self.final_mu, self.final_sigma, self.x_h, self.y_h)

            # Z_var, Z_mean = self.plot.Z_var_mean(self.final_mu, self.final_sigma)
            # initial_x = list()
            # initial_y = list()
            # final_x = list()
            # final_y = list()
            # for i in range(self.part_ant.shape[1]):
            #     if i % 2 == 0:
            #         initial_x.append(self.part_ant[0, i])
            #         final_x.append(self.part_ant[-1, i])
            #     else:
            #         initial_y.append(self.part_ant[0, i])
            #         final_y.append(self.part_ant[-1, i])
            # vehicles = int(self.part_ant.shape[1] / 2)
            # # print(vehicles)
            # for i in range(vehicles):
            #     self.plot.plot_trajectory_classic(self.axs, self.part_ant[:, 2 * i], self.part_ant[:, 2 * i + 1],
            #                                  colormap=self.plot.colors[i])
            # if self.init_fig:
            #     self.axs.plot(initial_x, initial_y, 'o', color='black', markersize=3, label='ASVs initial positions')
            #     # self.axs.plot(final_x, final_y, 'x', color='red', markersize=4,
            #     #               label='ASVs exploration final positions')
            #     self.axs.legend(loc=3, fontsize=6)
            # # else:
            #     # self.axs.plot(final_x, final_y, 'x', color='red', markersize=4)
            #
            # im2 = self.axs.imshow(Z_var.T, interpolation='bilinear', origin='lower', cmap="gist_yarg", vmin=0, vmax=1.0)
            # # im3 = self.axs.imshow(Z_mean.T, interpolation='bilinear', origin='lower', cmap=self.plot.cmapmean, vmin=0,
            # #                     vmax=1.0)
            # if self.init_fig:
            #     plt.colorbar(im2, ax=self.axs, label='σ', shrink=1.0)
            #     # plt.colorbar(im3, ax=self.axs, label='µ', shrink=1.0)
            #     self.axs.set_xlabel("x [m]")
            #     self.axs.set_ylabel("y [m]")
            #     self.axs.set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
            #     self.axs.set_xticks([0, 50, 100])
            #     self.axs.set_aspect('equal')
            #     self.axs.set_ylim([self.ys, 0])
            #     self.axs.grid(True)
            #     self.init_fig = False
            #     ticks_x = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
            #     self.axs.xaxis.set_major_formatter(ticks_x)
            #
            #     ticks_y = ticker.FuncFormatter(lambda x, pos: format(int(x * 100), ','))
            #     self.axs.yaxis.set_major_formatter(ticks_y)
            #     plt.pause(6)
            # else:
            #     plt.pause(0.5)
            # plt.show()
            # self.ims.append([im2])

        return done

    def data_out(self):

        """
        Return the first and the last position of the particles (drones).
        """

        return self.X_test, self.secure, self.bench_function, self.grid_min, self.sigma, \
            self.mu, self.part_ant, self.bench_array, self.grid_or, self.bench_max

    def error_calculation(self):

        """
        Return the first and the last position of the particles (drones).
        """
        self.type_error = 'all_map'
        self.calculate_error()
        self.map_mse.append(self.error_map)
        self.r2_map.append(r2_score(y_true=self.bench_array, y_pred=self.mu))

        self.type_error = 'contamination'
        self.calculate_error()
        self.peak_mse.append(self.mean_error_peak)


    def return_bench(self):
        return self.centers_bench, self.dict_limits_bench, self.center_peaks_bench

    def print_error(self):
        print("MSE peaks:", np.mean(np.array(self.peak_mse)), '+-', np.std(np.array(self.peak_mse)) * 1.96)
        print("MSE map:", np.mean(np.array(self.map_mse)), '+-', np.std(np.array(self.map_mse)) * 1.96)
        print("R2 map:", np.mean(np.array(self.r2_map)), '+-', np.std(np.array(self.r2_map)) * 1.96)
        print("Cantidad de valores R2:", len(self.r2_map))
        print("Valores R2:", self.r2_map)



    def return_seed(self):
        return self.seed