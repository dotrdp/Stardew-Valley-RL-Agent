import torch
import networkx as nx

items = {
    "Pickaxe": 1.0,
    "Axe": 2.0,
    "MeeleWeapon": 3.0,
    "nothing": 0.0,
    "wood": 4.0,
    "stone": 5.0,
}
# str -> index -> mlp

seasons = {
    "spring": 1.0,
    "summer": 2.0,
    "autumn": 3.0,
    "winter": 4.0,
}

def get_fixed_neighborhood_vector(energy_graph, player_node, nodes,tile_dataset, radius=30, max_nodes=120):
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
            lengths.append(-1.0)  # Padding value
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
                            lengths[-1] = float(length)
                    lengths[-1] = -1.0
            except nx.NetworkXNoPath:
                lengths.append(-1.0)
    return lengths


# vector -> LSTM

# vector to define the world
# implementation details: we need a normalized vector with -1-1 range, note the graph must be normalized into a distance and normalize it's weights
def get_state_embedding(env, player) -> torch.Tensor:
    # Normalize time to [-1, 1]
    time_feat = torch.tensor([env.time], dtype=torch.float32)
    snow_feat = torch.tensor([1.0 if env.snow else -1.0], dtype=torch.float32)
    rain_feat = torch.tensor([1.0 if env.raining else -1.0], dtype=torch.float32)

    # Season as one-hot (already 0/1, so scale to [-1, 1])
    seasons_feat = torch.tensor(seasons[env.season], dtype=torch.float32) if env.season in seasons else ValueError(f"Unknown season: {env.season}") 

    # Inventory as multi-hot (already 0/1, so scale to [-1, 1])
    inventory_feat = torch.arange(len(items), dtype=torch.float32)
    for index, item in enumerate(player.inventory.items):
        if item.name in items:
            inventory_feat[index] = items[item]
        else:
            print(f"Warning: Item '{item}' not recognized in items dictionary.")

    # Stamina normalized to [-1, 1]
    stamina_feat = torch.tensor([(player.stamina / 270)], dtype=torch.float32)

    # Energy graph: aggregate normalized 'we' within normalized distance <= 1
    energy_graph = env.get_energy_graph()
    nodes = energy_graph.nodes
    player_node = player.position
    lengths = torch.tensor(get_fixed_neighborhood_vector(energy_graph, player_node, nodes, env))
    print(f"Lengths: {lengths}")

    # suppose we want a circle around the player node, so the amount of nodes feed into the model and the dimensionality of the resultant vector is always the same
    if isinstance(seasons_feat, ValueError):
        raise seasons_feat
    result = torch.cat([
        time_feat,
        snow_feat,
        rain_feat,
        seasons_feat,
        inventory_feat,
        stamina_feat,
        lengths
    ])
#     # output
# tensor([ 2.3200e+03, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00,
#         -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00,
#         -1.0000e+00, -1.0000e+00, -1.0000e+00,  1.0000e+00, -1.0000e+00,
#         -1.0000e+00,  1.3000e+01,  1.2000e+01,  1.2000e+01,  1.1000e+01,
#          1.1000e+01,  1.0000e+01,  1.0000e+01,  9.0000e+00,  9.0000e+00,
#          8.0000e+00,  8.0000e+00,  7.0000e+00,  9.0000e+00,  8.0000e+00,
#         -1.0000e+00,  1.1000e+01,  1.0000e+01,  9.0000e+00,  7.0000e+00,
#          6.0000e+00,  7.0000e+00, -1.0000e+00,  1.0000e+01,  9.0000e+00,
#          8.0000e+00,  6.0000e+00,  5.0000e+00,  6.0000e+00,  8.0000e+00,
#         -1.0000e+00,  9.0000e+00,  8.0000e+00,  7.0000e+00,  7.0000e+00,
#          6.0000e+00,  5.0000e+00,  4.0000e+00,  5.0000e+00, -1.0000e+00,
#          8.0000e+00,  7.0000e+00,  6.0000e+00,  5.0000e+00,  4.0000e+00,
#          3.0000e+00,  4.0000e+00, -1.0000e+00,  7.0000e+00,  6.0000e+00,
#          5.0000e+00,  4.0000e+00,  3.0000e+00,  2.0000e+00,  3.0000e+00,
#         -1.0000e+00,  6.0000e+00,  5.0000e+00,  4.0000e+00,  3.0000e+00,
#          2.0000e+00,  1.0000e+00,  2.0000e+00, -1.0000e+00,  5.0000e+00,
#          4.0000e+00,  3.0000e+00,  2.0000e+00,  1.0000e+00,  0.0000e+00,
#          1.0000e+00, -1.0000e+00,  6.0000e+00,  5.0000e+00,  4.0000e+00,
#          3.0000e+00,  2.0000e+00,  1.0000e+00,  2.0000e+00])

    return result


