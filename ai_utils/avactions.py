from enum import Enum

class world_actions(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    USE_ITEM1 = 4
    USE_ITEM2 = 5
    USE_ITEM3 = 6
    USE_ITEM4 = 7
    USE_ITEM5 = 8
    USE_ITEM6 = 9
    USE_ITEM7 = 10
    USE_ITEM8 = 11
    USE_ITEM9 = 12
    USE_ITEM10 = 13
    USE_ITEM11 = 14
    USE_ITEM12 = 15
    INTERACT = 16

def get_available_actions(player):
    """
    Returns a list of available actions.
    """
    if not player.can_move:
        raise ValueError("Player cannot move.")

    return world_actions

