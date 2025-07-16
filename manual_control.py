import keyboard 
from map_wrapper import Map
from player import player
from ENV import environment
from API import StardewModdingAPI


api = StardewModdingAPI(method="ssh+tty")
env = environment(api)
doll = player(env)
map = Map(api)


while True:
    keyboard.add_hotkey('w', lambda: doll.normal_action("w", 200))
    keyboard.add_hotkey('s', lambda: doll.normal_action("s", 200))
    keyboard.add_hotkey('a', lambda: doll.normal_action("a", 200))
    keyboard.add_hotkey('d', lambda: doll.normal_action("d", 200))
    keyboard.add_hotkey('c', lambda: doll.normal_action("c", 200))
    keyboard.add_hotkey('1', lambda: doll.inventory.items[0]())
    keyboard.add_hotkey('2', lambda: doll.inventory.items[1]())
    keyboard.add_hotkey('3', lambda: doll.inventory.items[2]())
    keyboard.add_hotkey('4', lambda: doll.inventory.items[3]())
    keyboard.add_hotkey('5', lambda: doll.inventory.items[4]())
    keyboard.add_hotkey('6', lambda: doll.inventory.items[5]())
    keyboard.add_hotkey('7', lambda: doll.inventory.items[6]())
    keyboard.add_hotkey('8', lambda: doll.inventory.items[7]())
    keyboard.add_hotkey('9', lambda: doll.inventory.items[8]())
    keyboard.add_hotkey('0', lambda: doll.inventory.items[9]())
    keyboard.add_hotkey('q', lambda: doll.inventory.items[10]())
    keyboard.add_hotkey('e', lambda: doll.inventory.items[11]())
    keyboard.add_hotkey('m', lambda: print(map.grid))
    keyboard.wait()
