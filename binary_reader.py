import msgpack
import base64
from API import StardewModdingAPI

def read_msgpack_base64(raw_base64_data: str):
    binary_data = base64.b64decode(raw_base64_data)
    unpacked_data = msgpack.unpackb(binary_data, raw=False)
    return unpacked_data

api = StardewModdingAPI(method="ssh+tty")
data = api.__getattribute__("reflection")(function="GetProperty", args=["player", "Items"])["Base64_binary"]

items = read_msgpack_base64(data)
print("Items:", items)

