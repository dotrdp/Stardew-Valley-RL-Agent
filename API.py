""" To use the StardewModdingAPI, create a class of it and tell it which method will it use to execute the commands.
simple snippet,. bro
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
"""

import subprocess
import json
import docker
import msgpack
import base64
from logger import Logger

def read_msgpack_base64(raw_base64_data: str):
    binary_data = base64.b64decode(raw_base64_data)
    unpacked_data = msgpack.unpackb(binary_data, raw=False)
    return unpacked_data

class template:
    def __init__(self, exec_method, provided_docs, target, port="8080"):
        self.active = True
        self.exec_method = exec_method
        self.docs = provided_docs
        self.port = port
        self.target = target

    def __call__(self, function, args=None):
        if self.active:
            return self.exec_method(port=self.port, target=self.target, function=function, args=args)
        else:
            raise Exception("not possible")
    def __str__(self) -> str:
        return "\n".join(self.docs) 

class exec_method:
    def __init__(self, method, docker_image_name = None):
        self.method = method
        self.docker_image_name = docker_image_name
        
        self.ssh_wrapper = ["ssh", "rdiaz@mini.lan"]
        if method == "docker" and docker_image_name:
            try:
                self.docker_client = docker.from_env()
                self.docker_container = self.docker_client.containers.get(docker_image_name)
            except docker.errors.NotFound: # type: ignore 
                raise ValueError(f"Docker container '{docker_image_name}' not found")
            except Exception as e:
                raise ValueError(f"Failed to connect to Docker: {e}")

    def __call__(self, **kwargs):
        if self.method == "tty":
            return self.subprocess_wrap(subprocess.check_output(list(self.API_wrap(**kwargs).split())))
        elif self.method == "ssh+tty":
            return self.subprocess_wrap(subprocess.check_output(self.ssh_wrapper + [self.API_wrap(**kwargs)]))
        elif self.method == "docker":
            if self.docker_image_name is None:
                raise ValueError("Docker image name must be provided for docker execution method.")
            try:
                command = self.API_wrap(wrap_escaping=False, **kwargs)
                result = self.docker_container.exec_run(command, tty=True)
                if result.exit_code != 0:
                    raise RuntimeError(f"Docker command failed with exit code {result.exit_code}: {result.output.decode('utf-8')}")
                return self.subprocess_wrap(result.output)
            except Exception as e:
                raise RuntimeError(f"Docker execution failed: {e}") 
    def API_wrap(self, **kwargs) -> str:
        port = kwargs.pop("port", "8080")
        target = kwargs.pop("target") 
        function = kwargs.pop("function")
        wrap_escaping = kwargs.pop("wrap_escaping", True)
        parameters = kwargs.pop("args", "none")
        if parameters == "none":
            parameters = ""
        else:
            parameters = ", ".join([rf'"{param}"' for param in parameters]) # type: ignore
        res = f"curl -s -X POST http://localhost:{port}/api/execute -H \"Content-Type: application/json\" -d" 
        func = " '{\n"+"\"Target\": "+f"\"{target}\",\n"+"\"Method\": "+f"\"{function}\",\n"+"\"Parameters\": "+f"[{parameters}]\n"+"}'"
        if wrap_escaping != True:
            return res + func.replace('\"', '"')
        return res + func
    def subprocess_wrap(self, bytes_output: bytes) -> dict:
        readable_output = bytes_output.decode("utf-8")
        return dict(json.loads(readable_output))


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
    def __init__(self, method: str = "tty", docker_image_name = None, port: str = "8080", loglevel: str = "CRITICAL"):
        self.method = exec_method(method, docker_image_name)
        self.docs = DOCS()
        self.port = port
        self.logger = Logger(loglevel)
        for target in self.docs.__dict__.keys():
            setattr(self, target, template(self.method, getattr(self.docs, target), target))
        self.logger.log(f"StardewModdingAPI initialized with method: {self.method.method} on port: {self.port}", "DEBUG")
    def __str__(self) -> str:
        return "\n".join([f"{key}:\n{value}" for key, value in self.docs.__dict__.items() if isinstance(value, list)]) + "\n" + str(self.method)
    def hold_key(self, key: str, durationMS = 1000):
        met = self.method.method
        command = None
        self.logger.log(f"Holding key: {key} for {durationMS}ms using method: {met}", "DEBUG")
        res = f"curl -X POST http://localhost:{self.port}/api/keyboard/hold -H \"Content-Type: application/json\" -d" 
        if met == "tty":
            command = res + " '{\n"+"\"Key\": "+f"\"{key}\",\n"+"\"DurationMs\": "+f"{durationMS},\n"+"}'"
            return self.method.subprocess_wrap(subprocess.check_output(command.split()))
        elif met == "ssh+tty":
            command = self.method.ssh_wrapper + (res + " '{\n"+"\"Key\": "+f"\"{key}\",\n"+"\"DurationMs\": "+f"{durationMS}\n"+"}'").split()
            return self.method.subprocess_wrap(subprocess.check_output(command))
        elif met == "docker":
            try:
                command = res + " '{\n"+"\"Key\": "+f"\"{key}\",\n"+"\"DurationMs\": "+f"{durationMS}\n"+"}'"
                result = self.method.docker_container.exec_run(command, tty=True)
                if result.exit_code != 0:
                    raise RuntimeError(f"Docker command failed with exit code {result.exit_code}: {result.output.decode('utf-8')}")
                return self.method.subprocess_wrap(result.output)
            except Exception as e:
                raise RuntimeError(f"Docker execution failed: {e}")
        else:
            raise ValueError("Create an API instance first") 


    def map(self):
        self.logger.log("map fetching, which is very heavy and computer intensive\nConsider caching or sumt idk", "WARNING")
        met = self.method.method
        res = f"curl http://localhost:{self.port}/api/map"
        if met == "tty":
            return self.method.subprocess_wrap(subprocess.check_output(res.split()))
        elif met == "ssh+tty":
            return self.method.subprocess_wrap(subprocess.check_output(self.method.ssh_wrapper+res.split()))
        elif met == "docker":
            try:
                result = self.method.docker_container.exec_run(res, tty=True)
                if result.exit_code != 0:
                    raise RuntimeError(f"Docker command failed with exit code {result.exit_code}: {result.output.decode('utf-8')}")
                return self.method.subprocess_wrap(result.output)
            except Exception as e:
                raise RuntimeError(f"Docker execution failed: {e}")
        else:
            raise ValueError("Create an API instance first") 

            



