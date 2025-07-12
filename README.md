# Stardew-Valley-RL-Agent
Artificial Inteligence(AI) agent to playthrough stardew valley.
- Currently implemented .NET 6 RPC protocol over a stardew valley mod.
- Provided API in python for interaction with the environment,
# TO DO?,
Currently the agent itself hasnt been implemented into the environment, allowing a RPC API for reading the game status, however once the agent is wrapped using the RPC protocol, the agent can be then architectured and trained.

To use the StardewModdingAPI over python using the RPC protocol, create a class of it and tell it which method will it use to execute the commands.
simple snippet
    from API import StardewModdingAPI
    stardewapi = StardewModdingAPI(method="tty") # or "ssh+tty" or "docker" 
    response = stardewapi.gamecontenthelper(target="Game1", function="DoesAssetExist", args=["assets/terrainFeatures/grass.png"])

you can see all functions available by:
    print(stardewapi)

also you can see the documentation of each function by:
    print(stardewapi.gamecontenthelper) # for example



FOR DOCKER YOULL NEED TO GIVE IT THE NAME OF THE DOCKER IMAGE
    stardewapi = StardewModdingAPI(method="docker", docker_image_name="menacing racer") #idk man the docker images name are funny

for the rest it works like any other method

Make sure there is an instance of the game running (duh)

# cli python code meant for debugging
<img width="2240" height="1545" alt="image" src="https://github.com/user-attachments/assets/4084d622-8cfb-40db-906d-40fda312d037" />
```python
# cli usecase

# python cli.py function target params
--example
python cli.py reflection GetProperty Context IsPlayerFree

# note that function is an API itself documented here
# https://stardewvalleywiki.com/Modding:Modder_Guide/APIs
        self.helper = [
            "Dispose()",
            "Equals(Object obj)",
            "GetHashCode()",
            "GetType()",
            "ReadConfig()",
            "ToString()",
            "WriteConfig(TConfig config)",
        ]
        self.data = [
            "Equals(Object obj)",
            "GetHashCode()",
            "GetType()",
            "ReadGlobalData(String key)",
            "ReadJsonFile(String path)",
            "ReadSaveData(String key)",
            "ToString()",
            "WriteGlobalData(String key, TModel data)",
            "WriteJsonFile(String path, TModel data)",
            "WriteSaveData(String key, TModel model)",
        ]
        self.events = ["Equals(Object obj)", "GetHashCode()", "GetType()", "ToString()"]
        self.gamecontenthelper = [
            "DoesAssetExist(IAssetName assetName)",
            "Equals(Object obj)",
            "GetHashCode()",
            "GetPatchHelper(T data, String assetName)",
            "GetType()",
            "InvalidateCache()",
            "InvalidateCache(Func\u00602 predicate)",
            "InvalidateCache(IAssetName assetName)",
            "InvalidateCache(String key)",
            "Load(IAssetName assetName)",
            "Load(String key)",
            "ParseAssetName(String rawName)",
            "ToString()",
        ]
        self.input = [
            "Equals(Object obj)",
            "GetCursorPosition()",
            "GetHashCode()",
            "GetState(SButton button)",
            "GetType()",
            "IsDown(SButton button)",
            "IsSuppressed(SButton button)",
            "Suppress(SButton button)",
            "SuppressActiveKeybinds(KeybindList keybindList)",
            "SuppressScrollWheel()",
            "ToString()",
        ]
        self.modcontenthelper = [
            "DoesAssetExist(String relativePath)",
            "Equals(Object obj)",
            "GetHashCode()",
            "GetInternalAssetName(String relativePath)",
            "GetPatchHelper(T data, String relativePath)",
            "GetType()",
            "Load(String relativePath)",
            "ToString()",
        ]
        self.modregistry = [
            "Equals(Object obj)",
            "Get(String uniqueID)",
            "GetAll()",
            "GetApi(String uniqueID)",
            "GetApi(String uniqueID)",
            "GetHashCode()",
            "GetType()",
            "IsLoaded(String uniqueID)",
            "ToString()",
        ]
        self.multiplayer = [
            "Equals(Object obj)",
            "GetActiveLocations()",
            "GetConnectedPlayer(Int64 id)",
            "GetConnectedPlayers()",
            "GetHashCode()",
            "GetNewID()",
            "GetType()",
            "SendMessage(TMessage message, String messageType, String[] modIDs, Int64[] playerIDs)",
            "ToString()",
        ]
        self.reflection = [
            "Equals(Object obj)",
            "GetField(Object obj, String name, Boolean required)",
            "GetField(Type type, String name, Boolean required)",
            "GetHashCode()",
            "GetMethod(Object obj, String name, Boolean required)",
            "GetMethod(Type type, String name, Boolean required)",
            "GetProperty(Object obj, String name, Boolean required)",
            "GetProperty(Type type, String name, Boolean required)",
            "GetType()",
            "ToString()",
        ]
        self.translation = [
            "ContainsKey(String key)",
            "Equals(Object obj)",
            "Get(String key, Object tokens)",
            "Get(String key)",
            "GetHashCode()",
            "GetInAllLocales(String key, Boolean withFallback)",
            "GetKeys()",
            "GetTranslations()",
            "GetType()",
            "ToString()",
        ]
        self.consolecommands = [
            "Add(String name, String documentation, Action\u00602 callback)",
            "Equals(Object obj)",
            "GetHashCode()",
            "GetType()",
            "ToString()",
        ]

```


