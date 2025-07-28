# import torch
from ENV import environment
from player import player
from API import StardewModdingAPI
import networkx as nx

# [NOTE] set your method here
METHOD = "docker"
LOG_LEVEL = "DEBUG"
# [DEBUG]: 0
# [INFO]: 1
# [WARNING]: 2
# [ERROR]: 3
# [FATAL]: 4 ;[NOTE] to call fatal set the string to "CRITICAL"

api = StardewModdingAPI(method=METHOD, loglevel=LOG_LEVEL)
game_environment = environment(api, loglevel=LOG_LEVEL)
player_agent = player(game_environment)
print(game_environment.map)
graph = game_environment.get_collision_graph()
x, y = player_agent.position
x, y = int(x), int(y)
egraph = game_environment.get_energy_graph()
nx.write_graphml(egraph, "./a.graphml")
connected_components = list(nx.connected_components(egraph))[0]
max_distance = 0
furthest_point = None
for point in connected_components:
    distance = nx.shortest_path_length(egraph, source=(x, y), target=point, weight='weight')
    if distance > max_distance:
        max_distance = distance
        furthest_point = point
xt, yt = furthest_point #type: ignore
print(f"Furthest point from player({str(x)},{str(y)}) is ({str(xt)},{str(yt)}) with distance {max_distance}")
p = nx.dijkstra_path(egraph, source=(x, y), target=(xt, yt), weight='weight')
game_environment.draw_path(p)
print(f"Shortest path from player({str(x)},{str(y)}) to target (3, 11):")
player_agent.follow_energy_path(p)
# nx.write_graphml(game_environment.get_collision_graph(), "/home/rd/code/PythonStardewAPI/a.graphml")
# 64 15 Farm, is the default position at map


class RewardSystem:
    def __init__(self, env: environment, map=None):
        self.env = env
        self.cached_state = map

    def get_reward(self, state, action, map=None):
        if map != None:
            self.cached_state = map
            self.env.logger.log(
                "RewardSystem: setting provided map as state", "DEBUG")
        if self.cached_state == None:
            self.env.logger.log(
                "RewardSystem: requires a new map to draw a reward\notherwise it will try to use it's previous cached state as a state", "WARNING")
            self.env.logger.log(
                "RewardSystem: previous state is None", "CRITICAL")
            self.env.logger.log(
                "RewardSystem: cached state is None, instantiating a new Map class\nPLEASE DON'T LET IT TO DO THIS, CACHE THE MAP AS MUCH AS POSSIBLE", "WARNING")
        self.env.logger.log(
            "RewardSystem: using the current map as a state", "DEBUG")


class Agent:
    def __init__(self, env: environment, player: player):
        self.env = env
        self.player = player

    def act(self, state):
        # Convert the state to a tensor
        pass

        # Get the action from the player
