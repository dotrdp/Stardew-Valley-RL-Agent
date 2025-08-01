from ..utils import player, environment # for type hints
import torch
import networkx as nx

items = {
    "Pickaxe": 1,
    "Axe": 2,
    "MeeleWeapon": 2,
    "nothing": 0,
    "wood": 4,
    "stone": 5,
}

seasons = {
    "Spring": 1,
    "Summer": 2,
    "Autumn": 3,
    "Winter": 4,
}

def get_state_embedding(env: environment, player: player) -> torch.Tensor:
    result = torch.tensor(env.time)
    result = torch.concat((result, torch.tensor(1 if env.snow else 0, dtype=torch.float32)))
    result = torch.concat((result, torch.tensor(1 if env.raining else 0, dtype=torch.float32)))
    if env.season in seasons:
        result = torch.concat((result, torch.tensor(seasons[env.season], dtype=torch.float32)))
    else:
        print(f"Unknown season: {env.season}")
        result = torch.concat((result, torch.tensor(0, dtype=torch.float32)))

    for item in player.inventory.items:
        if item in items:
            result = torch.concat((result, torch.tensor(items[item], dtype=torch.float32)))
        else:
            print(f"Unknown item: {item}")
    result = torch.concat((result, torch.tensor(player.stamina, dtype=torch.float32)))
    energy_graph = env.get_energy_graph()
    nodes = list(energy_graph.nodes)
    player_node = nodes[player.position] #type: ignore
    for node in nodes:
        distance = nx.shortest_path_length(energy_graph, source=player_node, target=node)
        print(node.keys())
        if distance <= 20:
            print("OK")
            # result = torch.concat((result, torch.tensor(node.we, dtype=torch.float32)))


    return result

