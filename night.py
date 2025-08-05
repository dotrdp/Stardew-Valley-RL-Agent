from utils import player, environment, StardewModdingAPI
import networkx as nx

api = StardewModdingAPI()
environment = environment(api)
player = player(environment)

def tp_default():
    api.reflection(function="invokemethod", args=["game1", "warpFarmer", "64", "64", "true"]) #type: ignore

tp_default() # te lleva afuera a la granja

collision_graph = environment.get_collision_graph() # sin incluir paredes solo el componente del grafo donde esta el jugador
energy_graph = environment.get_energy_graph() # conecta todos los nodos incluyendo lo que sea rompible

energy_path = nx.shortest_path(energy_graph, player.position, (30, 30)) 

player.follow_energy_path(energy_path)
environment.print_path(energy_path) 
