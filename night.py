from utils import player, environment, StardewModdingAPI
import networkx as nx

api = StardewModdingAPI()
environment = environment(api)
player = player(environment)

def tp_default():
    api.reflection(function="invokemethod", args=["game1", "warpFarmer", "64", "15", "true"]) #type: ignore

tp_default() # te lleva afuera a la granja

collision_graph = environment.get_collision_graph() # sin incluir paredes solo el componente del grafo donde esta el jugador
energy_graph = environment.get_energy_graph() # conecta todos los nodos incluyendo lo que sea rompible

connected_components = list(nx.connected_components(energy_graph))[0]
max_distance = 0
furthest_point = None
for point in connected_components:
    distance = nx.shortest_path_length(energy_graph, source=(64, 15), target=point, weight='weight')
    if distance > max_distance:
        print("New furthest point found:", point, "with distance:", distance)
        max_distance = distance
        furthest_point = point
xt, yt = furthest_point #type: ignore
path = nx.dijkstra_path(energy_graph, source=(64, 15), target=(xt, yt), weight='weight') #type: ignore

player.follow_energy_path(path) # type: ignore
environment.print_path(path) 
