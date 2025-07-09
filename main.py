import requests


class helper:
    def __init__(self, exec_method):
        self.active = True
        self.exec_method = exec_method

    def __call__(self, **kwargs):
        if self.active:
            return self.exec_method(**kwargs)
        else:
            raise Exception("NOT POSSIBLE")


class exec_method:
    def __init__(self, method):
        self.method = method

    def __call__(self, *kwargs):
        if self.method == "tty":
            return requests.get(*kwargs)
        elif self.method == "ssh+tty":
            pass
        elif self.method == "docker":
            pass

    def API_wrap(**kwargs):
        exec = kwargs.pop("Command")
        port = kwargs.pop("Port", "8080")
        res = f"curl -X POST http://localhost:{port}/api/execute -H" + " \"Content-Type: application/json\" -d '{\"target\":"+ f" \"translation\","+ \"method\": \"Get\", \"parameters\": [\"menu.title\"]}'"
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
