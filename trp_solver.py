
import numpy as np
from scipy.spatial import KDTree
import time

def sweep_sort_for_trp(points, center):
    # Sortiert Punkte so, dass nahe und radial nah zuerst kommen
    dists = np.linalg.norm(points - center, axis=1)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sort_index = np.lexsort((dists, angles))  # radial + distanz
    return points[sort_index]

def compute_arrival_time(path):
    total_time = 0.0
    arrival_sum = 0.0
    for i in range(1, len(path)):
        step = np.linalg.norm(path[i] - path[i - 1])
        total_time += step
        arrival_sum += total_time
    return arrival_sum

def trp_solver(num_points=1000, max_cluster_size=20):
    start_time = time.time()
    coords = np.random.rand(num_points, 2).astype(np.float32)
    grid_size = int(np.ceil(np.sqrt(num_points / max_cluster_size)))
    step = 1.0 / grid_size

    clusters = {}
    for point in coords:
        gx, gy = int(point[0] / step), int(point[1] / step)
        clusters.setdefault((gx, gy), []).append(point)

    cluster_routes = []
    centroids = []

    for points in clusters.values():
        points = np.array(points, dtype=np.float32)
        center = np.mean(points, axis=0)
        centroids.append(center)
        if len(points) > 1:
            sorted_points = sweep_sort_for_trp(points, center)
            cluster_routes.append(sorted_points)
        else:
            cluster_routes.append(points)

    centroids = np.array(centroids, dtype=np.float32)
    used = np.zeros(len(centroids), dtype=bool)
    tree = KDTree(centroids)
    path_order = []

    start = np.array([0.5, 0.5], dtype=np.float32)
    _, current_idx = tree.query(start)
    path_order.append(current_idx)
    used[current_idx] = True

    for _ in range(1, len(centroids)):
        _, idx = tree.query(centroids[current_idx], k=10)
        if isinstance(idx, int):
            idx = [idx]
        for i in idx:
            if not used[i]:
                current_idx = i
                used[i] = True
                path_order.append(i)
                break

    full_path = []
    for idx in path_order:
        full_path.extend(cluster_routes[idx])

    arrival_total = compute_arrival_time(full_path)
    duration = round(time.time() - start_time, 3)

    return {
        "Punkte": num_points,
        "Cluster": len(clusters),
        "Rechenzeit (s)": duration,
        "Summe Ankunftszeiten": round(arrival_total, 2),
        "Pfadlänge": len(full_path)
    }

if __name__ == "__main__":
    result = trp_solver(1000)
    print(result)
