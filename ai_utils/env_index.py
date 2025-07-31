from ..utils import player, environment # for type hints
import torch

items = {
    "Pickaxe": 1,
    "Axe": 2,
    "MeeleWeapon": 2,
    "nothing": 0,
    "wood": 4,
    "stone": 5,
}

def get_state_embedding(env: environment, player: player) -> torch.Tensor:
    result = torch.Tensor((env.time), dtype=torch.float32)
    for item in player.inventory.items:
        if item in items:
            result = torch.concat((result, torch.tensor(items[item], dtype=torch.float32)))
        else:
            print(f"Unknown item: {item}")


    return result

