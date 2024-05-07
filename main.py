import sys
import os
import laspy
import numpy as np
from osgeo import gdal
import time
from progressbar import ProgressBar



las = laspy.read('data.laz')


print(set(list(las.classification)))

tree_points = las.points[las.classification == 5]
building_points = las.points[las.classification == 6]

import numpy as np
from scipy.spatial import KDTree

def calculate_distance(p1, p2):
    # p1 i p2 są teraz oczekiwane jako krotki (x, y, z)
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

las = laspy.read('data.laz')
print(set(list(las.classification)))

tree_points = las.points[las.classification == 5]
building_points = las.points[las.classification == 6]

# Przygotowanie danych punktów jako tablice NumPy
tree_coords = np.vstack((tree_points.x, tree_points.y, tree_points.z)).T
building_coords = np.vstack((building_points.x, building_points.y, building_points.z)).T

# Użycie KDTree dla efektywnego znajdowania najbliższych punktów
building_tree = KDTree(building_coords)

# Obliczenie minimalnej odległości dla każdego punktu drzewa do najbliższego budynku
distances, _ = building_tree.query(tree_coords)
average_distance = np.mean(distances)
print(f"Średnia minimalna odległość od drzew do budynków: {average_distance}")

