
import numpy as np
import time
from scipy.spatial.distance import cdist
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# 1. Punktgenerierung
np.random.seed(42)
coords = np.random.rand(100, 2).astype(np.float32) * 100

# 2. Sweep-Algorithmus (HiTSP)
def tsp_sweep(coords):
    start = time.time()
    depot = np.mean(coords, axis=0)
    angles = np.arctan2(coords[:, 1] - depot[1], coords[:, 0] - depot[0])
    order = np.argsort(angles)
    total_length = sum(np.linalg.norm(coords[order[i]] - coords[order[i+1]]) for i in range(len(order)-1))
    total_length += np.linalg.norm(coords[order[-1]] - coords[order[0]])
    end = time.time()
    return {
        "Modus": "HiTSP Sweep",
        "Distanz": round(total_length, 2),
        "Rechenzeit (s)": round(end - start, 4),
        "Reihenfolge": order.tolist()
    }

# 3. OR-Tools TSP
def tsp_ortools(coords):
    start = time.time()
    n = len(coords)
    dist_matrix = cdist(coords, coords)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return int(dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        route = []
        total_dist = 0
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            total_dist += dist_matrix[manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
        route.append(route[0])
        end = time.time()
        return {
            "Modus": "Google OR-Tools",
            "Distanz": round(total_dist, 2),
            "Rechenzeit (s)": round(end - start, 4),
            "Reihenfolge": route
        }
    else:
        return {"Modus": "Google OR-Tools", "Fehler": "Keine Lösung gefunden"}

# 4. Vergleichsausgabe
if __name__ == "__main__":
    result_sweep = tsp_sweep(coords)
    result_ortools = tsp_ortools(coords)
    print("Sweep-Algorithmus:", result_sweep)
    print("OR-Tools:", result_ortools)
