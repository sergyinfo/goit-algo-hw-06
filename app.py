"""
Додаток для візуалізації транспортної мережі міста та пошуку шляхів між місцями.
"""
import heapq
from typing import Dict, List
import networkx as nx
import matplotlib.pyplot as plt

# Створюємо граф
G = nx.Graph()

# Додаємо вершини (місця)
places = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
# Додаємо ребра (дороги між місцями)
roads = [
    ('A', 'B', 2), ('A', 'C', 3), ('B', 'D', 4), ('B', 'E', 1),
    ('C', 'F', 5), ('C', 'G', 2), ('D', 'H', 1), ('E', 'I', 3),
    ('F', 'J', 2), ('G', 'H', 3), ('H', 'I', 1), ('I', 'J', 4),
    ('A', 'D', 7), ('B', 'F', 6), ('C', 'E', 2), ('D', 'G', 5),
    ('J', 'K', 3), ('K', 'L', 4), ('L', 'M', 2), ('M', 'N', 1),
    ('N', 'O', 2), ('O', 'P', 3), ('P', 'Q', 4), ('Q', 'R', 5),
    ('R', 'S', 1), ('S', 'T', 2), ('J', 'N', 6), ('K', 'P', 2),
    ('L', 'Q', 3), ('M', 'R', 4), ('N', 'S', 1), ('O', 'T', 5)
]
G.add_nodes_from(places)
G.add_weighted_edges_from(roads)

# Фіксуємо розташування вершин
pos = nx.spring_layout(G, seed=42)

# Візуалізуємо граф
plt.figure(figsize=(10, 8))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray')
plt.title('Транспортна мережа міста')
plt.show()

# Аналіз основних характеристик графа
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
degrees = dict(G.degree())

print(f"Кількість вершин: {num_nodes}")
print(f"Кількість ребер: {num_edges}")
print("Ступені вершин:", degrees)

def dfs_paths(graph: nx.Graph, start: str, goal: str) -> List[List[str]]:
    """
    Find all possible paths between start and goal using DFS algorithm.

    :param graph: Graph object
    :param start: Start node
    :param goal: Goal node
    :return: List of all possible paths
    """
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in set(graph.neighbors(vertex)) - set(path):
            if next == goal:
                yield path + [next]
            else:
                stack.append((next, path + [next]))

def bfs_paths(graph: nx.Graph, start: str, goal: str) -> List[List[str]]:
    """
    Find all possible paths between start and goal using BFS algorithm.

    :param graph: Graph object
    :param start: Start node
    :param goal: Goal node
    :return: List of all possible paths
    """
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in set(graph.neighbors(vertex)) - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))

start = 'A'
goal = 'T'

dfs_result = list(dfs_paths(G, start, goal))
bfs_result = list(bfs_paths(G, start, goal))

print("DFS шляхи:", dfs_result)
print("BFS шляхи:", bfs_result)

def dijkstra(graph: nx.Graph, start: str) -> Dict[str, int]:
    """
    Find the shortest paths from the start node to all other nodes using Dijkstra's algorithm.

    :param graph: Graph object
    :param start: Start node
    :return: Dictionary of shortest paths
    """
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight['weight']

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

shortest_paths = {node: dijkstra(G, node) for node in G.nodes}

print("Найкоротші шляхи між усіма вершинами:")
for start_node, paths in shortest_paths.items():
    print(f"Від {start_node}: {paths}")

def visualize_path(graph: nx.Graph, path: List[str], title: str) -> None:
    """
    Visualize the graph with the given path.

    :param graph: Graph object
    :param path: List of nodes in the path
    :param title: Title of the plot
    """
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 8))

    # All nodes and edges
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray')

    # Highlight the path
    if path:
        edges = [(path[n], path[n + 1]) for n in range(len(path) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color='r', width=2)
        nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color='r')

    # Add edge labels
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.title(title)
    plt.show()

# DFS path
dfs_path = next(dfs_paths(G, start, goal))
visualize_path(G, dfs_path, "DFS шлях: " + " -> ".join(dfs_path))

# BFS path
bfs_path = next(bfs_paths(G, start, goal))
visualize_path(G, bfs_path, "BFS шлях: " + " -> ".join(bfs_path))

# Shortest path using Dijkstra's algorithm
def shortest_path_dijkstra(graph: nx.Graph, start: str, goal: str) -> List[str]:
    """
    Find the shortest path between start and goal using Dijkstra's algorithm.

    :param graph: Graph object
    :param start: Start node
    :param goal: Goal node
    :return: List of nodes in the shortest path
    """
    return nx.dijkstra_path(graph, start, goal)

dijkstra_path = shortest_path_dijkstra(G, start, goal)
visualize_path(G, dijkstra_path, "Найкоротший шлях (Дейкстра): " + " -> ".join(dijkstra_path))
