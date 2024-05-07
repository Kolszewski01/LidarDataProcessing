import plotly.graph_objects as go
import numpy as np
import laspy
from scipy.spatial import KDTree

# Wczytanie danych z pliku LAS
las = laspy.read('data.laz')
tree_points = las.points[las.classification == 5]
building_points = las.points[las.classification == 6]

# Przygotowanie danych punktów jako tablice NumPy (włączając Z)
# Zapisujemy współrzędne drzew i budynków jako tablice (x, y, z) dla dalszego przetwarzania
tree_coords = np.vstack((tree_points.x, tree_points.y, tree_points.z)).T
building_coords = np.vstack((building_points.x, building_points.y, building_points.z)).T

# Użycie KDTree dla efektywnego znajdowania najbliższych punktów
building_tree = KDTree(building_coords)

# Wyszukiwanie najbliższych budynków dla każdego drzewa w zasięgu 10 jednostek
distances, indices = building_tree.query(tree_coords, distance_upper_bound=10)

# Filtracja wyników, aby uwzględnić tylko te poniżej 10 jednostek
valid_distances = distances < 10

# Utworzenie wykresu
fig = go.Figure()

# Dodanie punktów drzew
fig.add_trace(go.Scatter3d(
    x=tree_coords[:, 0],
    y=tree_coords[:, 1],
    z=tree_coords[:, 2],
    mode='markers',
    marker=dict(size=2, color='green'),
    name='Trees'
))

# Dodanie punktów budynków
fig.add_trace(go.Scatter3d(
    x=building_coords[:, 0],
    y=building_coords[:, 1],
    z=building_coords[:, 2],
    mode='markers',
    marker=dict(size=2, color='red'),
    name='Buildings'
))

# Dodanie linii łączących bliskie punkty
for tree, building_idx, dist in zip(tree_coords[valid_distances], indices[valid_distances], distances[valid_distances]):
    if dist < 10:  # Sprawdzanie ponowne, na wypadek problemów z 'distance_upper_bound'
        building = building_coords[building_idx]
        fig.add_trace(go.Scatter3d(
            x=[tree[0], building[0]],
            y=[tree[1], building[1]],
            z=[tree[2], building[2]],
            mode='lines',
            line=dict(color='blue', width=2)
        ))

# Konfiguracja wyglądu wykresu
fig.update_layout(
    title='3D Visualization of Trees and Buildings within 10 Units Distance',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

# Wyświetlenie wykresu
fig.show()
