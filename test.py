import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import numpy as np
import laspy

# Wczytanie danych
las = laspy.read('data.laz')
tree_points = las.points[las.classification == 5]
building_points = las.points[las.classification == 6]

# Przygotowanie danych punktów jako tablice NumPy
tree_coords = np.vstack((tree_points.x, tree_points.y)).T
building_coords = np.vstack((building_points.x, building_points.y)).T

# Użycie KDTree dla efektywnego znajdowania najbliższych punktów
building_tree = KDTree(building_coords)

# Wyszukiwanie najbliższych budynków dla każdego drzewa w zasięgu 10 metrów
distances, indices = building_tree.query(tree_coords, distance_upper_bound=10)

# Filtracja wyników, aby uwzględnić tylko te poniżej 10 metrów
valid_distances = distances < 10

# Rysowanie punktów
plt.figure(figsize=(10, 10))
plt.scatter(tree_coords[:, 0], tree_coords[:, 1], c='green', label='Trees')
plt.scatter(building_coords[:, 0], building_coords[:, 1], c='red', label='Buildings')

# Rysowanie linii między drzewami a budynkami z odległością mniejszą niż 10 metrów
for tree, building_idx, dist in zip(tree_coords[valid_distances], indices[valid_distances], distances[valid_distances]):
    if dist < 10:  # Dodatkowe sprawdzenie na wypadek, gdyby 'distance_upper_bound' nie działało jak trzeba
        building = building_coords[building_idx]
        plt.plot([tree[0], building[0]], [tree[1], building[1]], 'b--', linewidth=0.5)

plt.legend()
plt.title('Connections between trees and buildings within 10 meters')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.show()
