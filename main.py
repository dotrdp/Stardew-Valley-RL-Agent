"""
To use the StardewModdingAPI, create a class of it and tell it which method will it use to execute the commands.
simple snippet,. bro
    from API import StardewModdingAPI
    stardewapi = StardewModdingAPI(method="tty") # or "ssh+tty" or "docker" 
    response = stardewapi.gamecontenthelper(target="Game1", function="DoesAssetExist", args=["assets/terrainFeatures/grass.png"])

you can see all functions available by:
    print(stardewapi)

also you can see the documentation of each function by:
    print(stardewapi.gamecontenthelper) # for example

"""

import subprocess
class template:
    def __init__(self, exec_method, provided_docs, port="8080"):
        self.active = True
        self.exec_method = exec_method
        self.docs = provided_docs
        self.port = port

    def __call__(self, target, function, args=None):
        if self.active:
            return self.exec_method(port=self.port, target=target, function=function, args=args)
        else:
            raise Exception("not possible")
    def __str__(self) -> str:
        return "\n".join(self.docs) 

class exec_method:
    def __init__(self, method, docker_image_name = None):
        self.method = method
        self.docker_image_name = docker_image_name

    def __call__(self, **kwargs):
        if self.method == "tty":
            return subprocess.run(list(self.API_wrap(**kwargs).split()))
        elif self.method == "ssh+tty":
            return subprocess.run(["ssh" "rdiaz@mini.lan" f"{self.API_wrap(**kwargs)}"])
        elif self.method == "docker":
            if self.docker_image_name is None:
                raise ValueError("Docker image name must be provided for docker execution method.")
            return subprocess.run(["docker", "exec", "-it", f"{self.docker_image_name}", f"{self.API_wrap(**kwargs)}"]) # NOT TESTED 
    def API_wrap(**kwargs) -> str:
        port = kwargs.pop("port", "8080")
        target = kwargs.pop("target") 
        function = kwargs.pop("function")
        parameters = kwargs.pop("args", "none")
        if parameters == "none":
            parameters = ""
        else:
            parameters = ", ".join([f'"{param}"' for param in parameters]) # type: ignore
        res = f"curl -X POST http://localhost:{port}/api/execute -H \"Content-Type: application/json\" -d" 
        func = " '{\n"+"\"target\": "+f"\"{target}\",\n"+"\"method\": "+f"\"{function}\",\n"+"\"parameters\": "+f"[{parameters}]\n"+"}'"
        return res + func


class DOCS:
    def __init__(self):
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

class StardewModdingAPI:
    def __init__(self, method: str = "tty", docker_image_name = None, port: str = "8080"):
        self.method = exec_method(method, docker_image_name)
        self.docs = DOCS()
        self.port = port
        for target in self.docs.__dict__.keys():
            setattr(self, target, template(self.method, getattr(self.docs, target)))
    def __str__(self) -> str:
        return "\n".join([f"{key}:\n{value}" for key, value in self.docs.__dict__.items() if isinstance(value, list)]) + "\n" + str(self.method)

