class request():
    def __init__(self, function, taraget, args):
        self.function = function
        self.target = taraget
        self.params = args

####################################################################################################
# Note this initializes an api instance with the default port therefore it cant do multiple instances

METHOD = "ssh+tty"
ALIASES = {
    "isplayerfree": request("reflection", "GetProperty", ["Context", "IsPlayerFree"]),
    "hold": "custom_stuff",
    "clear_dialog": request("reflection", "invokemethod", ["game1", "exitActiveMenu"]),
    "tp": "custom_stuff",
    "map": "custom_stuff",
}

####################################################################################################
import argparse

from API import StardewModdingAPI, read_msgpack_base64
from map_wrapper import Map
parser = argparse.ArgumentParser(description='cli to call methods on the spot')
parser.add_argument('function', type=str, help='API method to call')
parser.add_argument('target', nargs='?', help='Target resource or object')
parser.add_argument('params', nargs='*', help='Additional parameters as key=value', default=[])

api = StardewModdingAPI(method=METHOD, docker_image_name="sdvd-server")

a = (parser.parse_args())
if a.function == "hold":
    result = api.hold_key(str(a.target), int(a.params[0]))
    for k,v in result.items():
        print(f"{k}: {v}")
    raise SystemExit(0)
if a.function == "tp":
    place = a.target
    x = a.params[0] 
    y = a.params[1]
    res = api.reflection("invokemethod", args=["game1", "warpFarmer", place,str(x), str(y), "true"]) # type: ignore
    for k,v in res.items():
        print(f"{k}: {v}")
    raise SystemExit(0)
if a.function == "map":
    map = Map(api)
    a = api.map()
    print(map)
    
if a.function in ALIASES:
    a = ALIASES[a.function]


show = ", ".join([f'"{param}"' for param in a.params]) # type: ignore
print(f"Calling \"{a.function}\" with method/function \"{a.target}\" with params [{show}]")
result = (api.__getattribute__(a.function)(function=a.target, args=[v for v in a.params]))
for k,v in result.items():
    print(f"{k}: {v}")
    if k == "Base64_binary":
        print(f"Decoded: {read_msgpack_base64(v)}")


