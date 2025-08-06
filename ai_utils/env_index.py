import torch
import networkx as nx
import numpy as np


items = {
    "Pickaxe": 1.0,
    "Axe": 2.0,
    "MeleeWeapon": 3.0,
    "Empty": 0.0,
    "Wood": 4.0,
    "Stone": 5.0,
    "Hoe": 6.0,
    "WateringCan": 7.0,
    "Slingshot": 8.0,
    "Wand": 9.0,
    "Pan": 10.0,
    "FishingRod": 11.0,
    "MilkPail": 12.0,
    "Lantern": 13.0,
    "Raft": 14.0,
    "Shears": 15.0,
}

# NOTE: do one hot encoding for items, seasons, locations
# str -> index -> mlp

seasons = {
    "spring": 1.0,
    "summer": 2.0,
    "autumn": 3.0,
    "winter": 4.0,
}

locations = {
    "AbandonedJojaMart": 1.0,
    "AdventureGuild": 2.0,
    "BathHousePool": 3.0,
    "Beach": 4.0,
    "BeachNightMarket": 5.0,
    "BoatTunnel": 6.0,
    "BugLand": 7.0,
    "BusStop": 8.0,
    "Cabin": 9.0,
    "Caldera": 10.0,
    "Cellar": 11.0,
    "Club": 12.0,
    "CommunityCenter": 13.0,
    "DecoratableLocation": 14.0,
    "Desert": 15.0,
    "DesertFestival": 16.0,
    "DwarfGate": 17.0,
    "FarmCave": 18.0,
    "FarmHouse": 19.0,
    "FishShop": 20.0,
    "Forest": 21.0,
    "IslandEast": 22.0,
    "IslandFarmCave": 23.0,
    "IslandFarmHouse": 24.0,
    "IslandFieldOffice": 25.0,
    "IslandForestLocation": 26.0,
    "IslandHut": 27.0,
    "IslandLocation": 28.0,
    "IslandNorth": 29.0,
    "IslandSecret": 30.0,
    "IslandShrine": 31.0,
    "IslandSouth": 32.0,
    "IslandSouthEast": 33.0,
    "IslandSouthEastCave": 34.0,
    "IslandWest": 35.0,
    "IslandWestCave1": 36.0,
    "JojaMart": 37.0,
    "LibraryMuseum": 38.0,
    "ManorHouse": 39.0,
    "MermaidHouse": 40.0,
    "Mine": 41.0,
    "MineInfo": 42.0,
    "MineShaft": 43.0,
    "Mountain": 44.0,
    "MovieTheater": 45.0,
    "Racer": 46.0,
    "Railroad": 47.0,
    "SeedShop": 48.0,
    "Sewer": 49.0,
    "ShopLocation": 50.0,
    "Submarine": 51.0,
    "Summit": 52.0,
    "Town": 53.0,
    "VolcanoDungeon": 54.0,
    "Wisp": 55.0,
    "WizardHouse": 56.0,
    "Woods": 57.0,
    "Hospital": 58.0,
}

def get_fixed_neighborhood_vector(energy_graph, player_node, nodes,tile_dataset, radius=20, max_nodes=400):
    # Find nodes within a circle (graph distance) around player_node
    circle_nodes = [
        node for node in nodes
        if node != player_node and
        nx.has_path(energy_graph, player_node, node) and
        nx.shortest_path_length(energy_graph, player_node, node) <= radius
    ]
    # Pad or truncate to fixed size
    circle_nodes = circle_nodes[:max_nodes]
    while len(circle_nodes) < max_nodes:
        circle_nodes.append(None)  # Use None for padding

    lengths = []
    for node in circle_nodes:
        if node is None:
            lengths.append(0.0)  # Padding value
        else:
            try:
                length = nx.shortest_path_length(energy_graph, source=player_node, target=node)
                # Normalize length to [-1, 1] based on radius
                lengths.append(length)
                x, y = node
                prop = tile_dataset.spatial_state[x][y].properties
                if "collision" in prop:
                    if "tool" in prop:
                        if prop["tool"] in items:
                            lengths[-1] = items[prop["tool"]]
                        else:
                            lengths[-1] = float(-1)*float(length)
                else:
                    lengths[-1] = float(length)
            except nx.NetworkXNoPath:
                lengths.append(-1.0)
    return lengths


# vector -> LSTM

# vector to define the world
# implementation details: we need a normalized vector with -1-1 range, note the graph must be normalized into a distance and normalize it's weights
def get_state_embedding(env, player) -> np.ndarray:
    # Normalize time to [-1, 1]
    time_feat = torch.tensor([env.time], dtype=torch.float32)
    snow_feat = torch.tensor([1.0 if env.snow else 0.0], dtype=torch.float32)
    rain_feat = torch.tensor([1.0 if env.raining else 0.0], dtype=torch.float32)
    money_feat = torch.tensor([player.money], dtype=torch.float32)

    # Season as one-hot (already 0/1, so scale to [-1, 1])
    seasons_feat = torch.tensor([seasons[env.season]], dtype=torch.float32) if env.season in seasons else ValueError(f"Unknown season: {env.season}") 

    # Inventory as multi-hot (already 0/1, so scale to [-1, 1])
    inventory_feat = torch.Tensor()
    for item in player.inventory.items:
        if item.name in items:
            inventory_feat = torch.concat((inventory_feat, torch.tensor([items[item.name]], dtype=torch.float32)))
        else:
            print(f"Warning: Item '{item.name}' not recognized in items dictionary.")
    location_feat = torch.Tensor([0.0])
    if player.location in locations:
        location_feat = torch.tensor([locations[player.location]], dtype=torch.float32)
    else:
        print(f"Warning: Location '{player.location}' not recognized in locations dictionary.")

    stamina_feat = torch.tensor([(player.stamina / 270)], dtype=torch.float32)

    # Energy graph: aggregate normalized 'we' within normalized distance <= 1
    energy_graph = env.get_energy_graph()
    nodes = energy_graph.nodes
    player_node = player.position
    lengths = torch.tensor(get_fixed_neighborhood_vector(energy_graph, player_node, nodes, env))

    # suppose we want a circle around the player node, so the amount of nodes feed into the model and the dimensionality of the resultant vector is always the same
    if isinstance(seasons_feat, ValueError):
        raise seasons_feat
    if not isinstance(inventory_feat, torch.Tensor):
        raise TypeError("Inventory feature must be a torch.Tensor")
    if not isinstance(lengths, torch.Tensor):
        raise TypeError("Lengths feature must be a torch.Tensor")
    if not isinstance(time_feat, torch.Tensor):
        raise TypeError("Time feature must be a torch.Tensor")
    if not isinstance(snow_feat, torch.Tensor):
        raise TypeError("Snow feature must be a torch.Tensor")
    if not isinstance(rain_feat, torch.Tensor):
        raise TypeError("Rain feature must be a torch.Tensor")
    if not isinstance(stamina_feat, torch.Tensor):
        raise TypeError("Stamina feature must be a torch.Tensor")


    a = np.array(time_feat.tolist() + snow_feat.tolist() + rain_feat.tolist() + money_feat.tolist() + seasons_feat.tolist() + inventory_feat.tolist() + location_feat.tolist() + stamina_feat.tolist() + lengths.tolist())
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b
