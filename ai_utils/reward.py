from enum import Enum

class rewardbroken(Enum):
    Tree = 15 #takes 15 hits to take down a tree
    t5p = 5 # this one only takes 5 hits
    t1p = 1 # not sure if this one exists, but it takes only 1 hit
    Stone = 1
    Weeds = 0.5 #remove this if you want, not really sure if it should be here

    # the defaults dictionary define the names that the tiles will have, along with the emoji associated to the tile type


# here are all the tiles data you need to know:
# defaults = {
#     "Building":  "󰆦",
#     "normal":    "·",
#     "player":    "",
#     "Chest":     "󰜦",
#     "Stone":     "",
#     "Weeds":     "󰹧",
#     "Twig":      "󱐚",
#     "Tree":      "",
#     "t5p":       "󰹩",
#     "t1p":       "",
#     "door":      "󰠚",
#     "NpcBarrier":"▫",
#     "Bed":       "",
#     "something": "▫",
#     "PetBowl":   "󰊎",
#     "shipbin":   "",
#     "Seed Spot": "",
#     "NoS":       "✕",
#     "Grass":     "󱔐",
#     "bar":       "▫",
#     "Buildable": "󰦻",
#     "NF":        "󰆷",
#     "NFNS":      "✕",
#     "notbuild":  "✕",
#     "Stone Owl": "󰏒",
# "Artifact Spot": "",
# "debugmarker":   "",
#     "cbuilding": "",
#     "mail":      "󰛮",
#     "f":         "·",
#   "NFF":         "·",
# "debug_red":     "✕",
#     "Water":     "",
# }
#
# def get_logic(type):
#    lgc = {
#         "cbuilding": {"collision": True, "blocks_crops": True},
#         "Building": {"collision": True, "blocks_crops": True},
#         "Water": {"collision": True, "blocks_crops": True},
#         "Stone": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Pickaxe"},
#         "Weeds": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Scythe"},
#         "Tree": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Axe", "health": 15},
#         "t5p": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Axe", "health": 5},
#         "t1p": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Axe", "health": 1},
#         "Twig": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Axe"},
#         "Seed Spot": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Hoe"},
#         "Chest": {"collision": True},
#         "door": {"collision": True, "collisionwarp": True},
#         "shipbin": {"collision": True, "sellplace": True, "blocks_crops": True},
#         "PetBowl": {"collision": True, "blocks_crops": True},
#         "notbuild": {"collision": True, "blocks_crops": True},
#         "NFNS": {"collision": True, "blocks_crops": True},
#         "Stone Owl": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Pickaxe"},
#         "Grass": {"blocks_crops": True, "breakable": True, "tool": "Scythe"},
#         "Artifact Spot": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Hoe"},
#         "mail": {"collision": True, "blocks_crops": True},
#         "unknown": {"collision": True, "blocks_crops": True},
#         }
#    if type in lgc:
#        return lgc[type]
#    else:
#         return {}
# each tile will have a type, feel free to check for the tile type, also VERY IMPORTANT NOTE:
# to avoid lag, the previous state wont be updated after every action automatically
# you'll need to call env.update_spatial_state() after every action or every time you want to update the state, note you might not need to do this now, just letting you know
#

class RewardSystem():
    def __init__(self, env, player):
        self.env = env
        self.player = player
        self.previous_location = player.location
        self.previous_state = {
            "spatial_state": self.env.spatial_state,
            "money": player.money,
            "stamina": player.stamina,
            "health": player.health,
            "location": player.location,
            "position": player.position,
            "action": None  # This can be used to track the last action taken, if needed
        }
        
    def evaluate_action(self, action):
        cumreward = 0
        money = self.player.money
        previous_spatial_state = self.previous_state["spatial_state"]
        previous_money = self.previous_state["money"]
        previous_stamina = self.previous_state["stamina"]
        previous_health = self.previous_state["health"]
        previous_location = self.previous_state["location"]
        previous_position = self.previous_state["position"]
        previous_action = self.previous_state["action"]

        location = self.player.location # NOTE THAT THIS IS A STRING, FOR X;Y REFER TO player.position
        stamina = self.player.stamina #if the player stamina hits 0, he will lose 150 gold which is a huge deal.
        health = self.player.health #if the player health hits 0, he will lose all his money, and items. 
        position = self.player.position # this is a tuple of (x, y) coordinates, e.g (0, 0) for the top left corner of the map

        # if previous_position != position:
        #     # check if the player has moved, or if he's chopping a tree or something

        # if action != previous_action:
        #     return log(range(-100, 100)) # idk bro, maybe you can reduce it's reward if it is always doing the same action

        # checking if broken tiles 
        for row in self.env.spatial_state:
            for tile in row: # this is how you loop through the tiles, trust me
                x, y = tile
                if previous_spatial_state[x][y] != self.env.spatial_state[x][y]:
                    # tile has changed, check if it was broken
                    if previous_spatial_state[x][y] in rewardbroken.__members__:
                        # this tile is a broken tile
                        cumreward += rewardbroken[previous_spatial_state[x][y]].value
        # checking if broken tiles

        if location != previous_location:
            cumreward += 10 # reward the player for changing location, e.g for moving his ass out of the FarmHouse, to the Farm lmao

        # let me know how should we implement the pathfinding, essentially in the original game it is a pain to get it good for a reason, it just kinda works, until a non-realized tile with a building breaks the entire thing, so I was thinking we could evaluate the agent's action at each turn, essentially letting it figure out everything

        return cumreward + money * 0.01  # 1% of the money, github copilot said so

