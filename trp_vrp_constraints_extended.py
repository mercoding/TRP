
import numpy as np
from scipy.spatial import KDTree

def sweep_sort_for_trp(points, center):
    dists = np.linalg.norm(points - center, axis=1)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sort_index = np.lexsort((dists, angles))
    return points[sort_index]

def compute_arrival_time(path):
    total_time = 0.0
    arrival_sum = 0.0
    for i in range(1, len(path)):
        step = np.linalg.norm(path[i] - path[i - 1])
        total_time += step
        arrival_sum += total_time
    return arrival_sum

def trp_vrp_constraints_extended(num_points=1000000, max_cluster_size=20, max_capacity=100,
                                  n_depots=3, service_time_per_point=2.0, max_route_time=1200.0,
                                  break_after=600.0, break_duration=15.0):
    coords = np.random.rand(num_points, 2).astype(np.float32)
    demands = np.random.randint(1, 5, size=num_points)
    time_windows = np.column_stack((
        np.random.randint(0, 500, size=num_points),
        np.random.randint(500, 1000, size=num_points)
    ))

    depots = np.random.rand(n_depots, 2).astype(np.float32)
    assigned_depots = KDTree(depots).query(coords)[1]

    depot_clusters = {i: {} for i in range(n_depots)}
    grid_size = int(np.ceil(np.sqrt(num_points / max_cluster_size)))
    step = 1.0 / grid_size

    for i, point in enumerate(coords):
        depot_idx = assigned_depots[i]
        gx, gy = int(point[0] / step), int(point[1] / step)
        key = (gx, gy)
        depot_clusters[depot_idx].setdefault(key, []).append((point, demands[i], time_windows[i]))

    total_arrival_time = 0.0
    total_segments = 0
    total_clusters = 0
    exceeded_routes = 0

    for depot_idx, clusters in depot_clusters.items():
        cluster_routes = []
        centroids = []

        for values in clusters.values():
            points = np.array([v[0] for v in values], dtype=np.float32)
            centroids.append(np.mean(points, axis=0))
            if len(points) > 1:
                sorted_points = sweep_sort_for_trp(points, np.mean(points, axis=0))
                cluster_routes.append(sorted_points)
            else:
                cluster_routes.append(points)

        centroids = np.array(centroids, dtype=np.float32)
        used = np.zeros(len(centroids), dtype=bool)
        tree = KDTree(centroids)
        path_order = []

        start = depots[depot_idx]
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

        route_time = 0.0
        service_counter = 0
        last_break = 0.0

        for i in range(1, len(full_path)):
            step_time = np.linalg.norm(full_path[i] - full_path[i - 1])
            route_time += step_time
            service_counter += 1
            route_time += service_time_per_point

            if route_time - last_break > break_after:
                route_time += break_duration
                last_break = route_time

        if route_time > max_route_time:
            exceeded_routes += 1

        total_arrival_time += route_time
        total_segments += len(full_path)
        total_clusters += len(clusters)

    return {
        "Punkte": num_points,
        "Depots": n_depots,
        "Cluster": total_clusters,
        "Summe Ankunftszeiten (inkl. Service, Pause)": round(total_arrival_time, 2),
        "Pfadlänge": total_segments,
        "Routen über Zeitlimit": exceeded_routes,
        "Fahrzeitlimit": max_route_time
    }
