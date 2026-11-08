import networkx as nx
class Env_Graphs:
    def __init__(self, spatial_state, logger):
        self.spatial_state = spatial_state
        self.logger = logger
        self.collision_graph = self.get_collision_graph()
        self.energy_graph = self.get_energy_graph()
        self.logger.log("Environment graphs initialized", "DEBUG")

    def update_spatial_state(self, spatial_state):
        self.spatial_state = spatial_state
        self.collision_graph = self.get_collision_graph()
        self.energy_graph = self.get_energy_graph()

    def get_collision_graph(self):
        graph = nx.Graph()
        for x in range(len(self.spatial_state)):
            for y in range(len(self.spatial_state[x])):
                tile = (x, y)
                if "collision" in self.spatial_state[x][y].properties:
                    continue
                neighbors = []
                if x+1 <= len(self.spatial_state) - 1:
                    neighbors.append((x+1, y))
                if x-1 >= 0:
                    neighbors.append((x-1, y))
                if y+1 <= len(self.spatial_state[x]) - 1:
                    neighbors.append((x, y+1))
                if y-1 >= 0:
                    neighbors.append((x, y-1))
                for neighbor in neighbors:
                    if "collision" not in self.spatial_state[neighbor[0]][neighbor[1]].properties:
                        
                        graph.add_edge(tile, neighbor, weight=1)
        self.logger.log("Walkable graph created", "DEBUG")
        return graph

    def get_energy_graph(self):
        graph = nx.Graph()
        rows = len(self.spatial_state) - 1 
        for x in range(rows):
            cols = len(self.spatial_state[x]) - 1
            for y in range(cols):
                tile = (x, y)

                props = self.spatial_state[x][y].properties

                if "collision" in props and "tool" not in props:
                    continue
                graph.add_node(tile)
                neighbor_positions = [
                    (min(x+1, len(self.spatial_state)), y),
                    (x, min(y+1, cols)),
                    (max(x-1, 0), y),
                    (x, max(y-1, 0))
                ]
                for nx_pos, ny_pos in neighbor_positions:
                    neighbor_props = getattr(self.spatial_state[nx_pos][ny_pos], "properties", [])
                    neighbor = (nx_pos, ny_pos)
                    if "collision" in neighbor_props:
                        if "tool" in props:
                            if "health" in props:
                                graph.add_edge(tile, neighbor, weight=props["health"])
                                continue
                            graph.add_edge(tile, neighbor, weight=2)
                    else:
                        graph.add_edge(tile, neighbor, weight=1)
            # Ensure graph is connected
        if nx.is_connected(graph):
            self.logger.log("Energy graph is connected", "DEBUG")
        self.logger.log("Energy graph created and connected", "DEBUG")
        return graph


