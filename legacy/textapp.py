import keyboard 
from utils import Map, player, environment, StardewModdingAPI


api = StardewModdingAPI()
env = environment(api)
doll = player(env)
map = Map(api)
def tp_player(x, y):
    dollx, dolly = doll.position
    fx, fy = int(dollx + x), int(dolly + y)
    location = doll.location
    print(f"Teleporting to {fx}, {fy} in {location}")
    (api.reflection(function="invokemethod", args=["game1", "warpFarmer", str(location),str(fx), str(fy), "true"])) #type: ignore

def tp_up():
    tp_player(0, 1)
def tp_down():
    tp_player(0, -1)
def tp_left():
    tp_player(-1, 0)
def tp_right():
    tp_player(1, 0)

def close_dialogue():
    doll.close_dialogue()

def print_map():
    print(map)
    print("controls:")
    print("w: up, s: down, a: left, d: right, c: interact")
    print("1-0: use inventory items 1-10")
    print("q: use inventory item 11, e: use inventory item 12")
    print("x interaction")
    print("j: teleport up, k: teleport down, h: teleport left, l: teleport right")
    print("o: close dialogue, m: print map")
    

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
    keyboard.add_hotkey('j', lambda: tp_up())
    keyboard.add_hotkey('k', lambda: tp_down())
    keyboard.add_hotkey('h', lambda: tp_left())
    keyboard.add_hotkey('l', lambda: tp_right())
    keyboard.add_hotkey('o', lambda: close_dialogue())
    keyboard.add_hotkey('m', lambda: print_map())
    keyboard.wait()
