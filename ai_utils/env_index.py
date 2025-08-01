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
    nodes = energy_graph.nodes
    player_node = player.position

    # Find all shortest path lengths from player_node
    lengths = torch.zeros(len(nodes), dtype=torch.float32)
    for node in nodes:
        print(f"Processing node: {node}, player_node: {player_node}")
        # print(type(node)) this is a trupe, uh forgot what is it called (x, y)
        if node != player_node:
            try:
                length = nx.shortest_path_length(energy_graph, source=player_node, target=node)
                # Normalize length to [-1, 1] based on max distance
                lengths[nodes[node]] = (length / env.max_distance) * 2 - 1
            except nx.NetworkXNoPath:
                lengths[nodes[node]] = -1.0
    result = torch.cat([
        time_feat,
        snow_feat,
        rain_feat,
        season_feat,
        inventory_feat,
        stamina_feat,
        lengths
    ])

    return result


