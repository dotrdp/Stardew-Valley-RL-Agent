from API import StardewModdingAPI

class Player():
    def __init__(self, game_instance, environment=None):
        self.game_instance = game_instance
        self.environment = environment
