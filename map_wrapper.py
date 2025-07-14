from API import StardewModdingAPI
import torch 

class grid():
    def __init__(self, grid_data):
        self.grid = torch.zeros()

class Map():
    def __init__(self, map_data):
        self.map_data = map_data["MapData"]
        print(self.map_data.keys())
    
    @property
    def grid(self):
        return grid(self.map_data["GridData"])

api = StardewModdingAPI(method="ssh+tty")
a = Map(api.map())
