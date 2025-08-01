import torch
import networkx as nx

items = {
    "Pickaxe": 1,
    "Axe": 2,
    "MeeleWeapon": 3,
    "nothing": 0,
    "wood": 4,
    "stone": 5,
}
# str -> index -> mlp

seasons = {
    "Spring": 1,
    "Summer": 2,
    "Autumn": 3,
    "Winter": 4,
}

# vector -> LSTM

# vector to define the world
# implementation details: we need a normalized vector with -1-1 range, note the graph must be normalized into a distance and normalize it's weights
def get_state_embedding(env, player) -> torch.Tensor:
    # Normalize time to [-1, 1]
    time_feat = torch.tensor([(env.time / 24.0) * 2 - 1], dtype=torch.float32)
    snow_feat = torch.tensor([1.0 if env.snow else -1.0], dtype=torch.float32)
    rain_feat = torch.tensor([1.0 if env.raining else -1.0], dtype=torch.float32)

    # Season as one-hot (already 0/1, so scale to [-1, 1])
    season_feat = torch.full((len(seasons),), -1.0, dtype=torch.float32)
    if env.season in seasons:
        season_feat[seasons[env.season]] = 1.0

    # Inventory as multi-hot (already 0/1, so scale to [-1, 1])
    inventory_feat = torch.full((len(items),), -1.0, dtype=torch.float32)
    for item in player.inventory.items:
        if item in items:
            inventory_feat[items[item]] = 1.0

    # Stamina normalized to [-1, 1]
    stamina_feat = torch.tensor([(player.stamina / 256.0) * 2 - 1], dtype=torch.float32)

    # Energy graph: aggregate normalized 'we' within normalized distance <= 1
    energy_graph = env.get_energy_graph()
    nodes = (energy_graph.nodes)
    x, y = player.position
    player_node = nodes[player.position]

    # Find all shortest path lengths from player_node
    lengths = nx.single_source_shortest_path_length(energy_graph, player_node, cutoff=20)
    # Normalize distances to [0, 1]
    norm_distances = {n: l / 20.0 for n, l in lengths.items()}

    # Normalize 'we' values to [-1, 1] if possible
    # such that the minimum 'we' is -1 and maximum 'we' is 1, whereas everything else is in a range of 0-1
    we_values = [energy_graph.nodes[n].get('weight', 0.0) for n in nodes]
    min_we, max_we = min(we_values), max(we_values)
    def norm_we(we):
        if max_we == min_we:
            return 0.0
        return ((we - min_we) / (max_we - min_we)) * 2 - 1

    agg_we = 0.0
    count = 0
    for node, norm_dist in norm_distances.items():
        if norm_dist <= 1.0:
            we = energy_graph.nodes[node].get('weight', 0.0)
            agg_we += norm_we(we)
            count += 1
    agg_we_feat = torch.tensor([agg_we / count if count > 0 else 0.0], dtype=torch.float32)

    # Concatenate all features
    embedding = torch.cat([
        time_feat,
        snow_feat,
        rain_feat,
        season_feat,
        inventory_feat,
        stamina_feat,
        agg_we_feat
    ])
    return embedding
