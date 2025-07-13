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
}

####################################################################################################
import argparse

from API import StardewModdingAPI
parser = argparse.ArgumentParser(description='cli to call methods on the spot')
parser.add_argument('function', type=str, help='API method to call')
parser.add_argument('target', nargs='?', help='Target resource or object')
parser.add_argument('params', nargs='*', help='Additional parameters as key=value', default=[])

a = (parser.parse_args())
for alias in ALIASES:
    if a.function == alias:
        a = ALIASES[alias]
        
api = StardewModdingAPI(method=METHOD)
show = ", ".join([f'"{param}"' for param in a.params]) # type: ignore
print(f"Calling \"{a.function}\" with method/function \"{a.target}\" with params [{show}]")
result = (api.__getattribute__(a.function)(function=a.target, args=[v for v in a.params]))
for k,v in result.items():
    print(f"{k}: {v}")

