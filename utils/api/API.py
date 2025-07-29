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
from logger import Logger
from dotenv import dotenv_values 

from .docs import DOCS
from .execute import exec_method
from .utils import read_msgpack_base64, template # needed here because other modules import it from here, DO NOT REMOVE read_msgpack_base64

prefs = dotenv_values(".env")
docker_image_name = prefs.get("DOCKER_IMAGE_NAME", "sdvd-server")
method = prefs.get("method", "docker")
ssh_host = prefs.get("ssh_host", ".")
ssh_user = prefs.get("ssh_user", ".")
ssh_port = prefs.get("ssh_port", "22")
proxy_port = prefs.get("proxy_port", "8080")
debug_level = prefs.get("debug_level", "ERROR")
debug_level_api = prefs.get("debug_level_api", "ERROR")


if debug_level != "ERROR":
    debug_level_api = debug_level

class StardewModdingAPI:
    def __init__(self, method: str = method, docker_image_name=DOCKER_IMAGE_NAME, port: str = proxy_port, loglevel: str = debug_level_api): # type: ignore
        self.method = exec_method(method, docker_image_name, ssh_user, ssh_host, ssh_port)
        self.docs = DOCS()
        self.port = port
        self.logger = Logger(loglevel)
        for target in self.docs.__dict__.keys():
            setattr(self, target, template(self.method,
                    getattr(self.docs, target), target))
        self.logger.log(
            f"StardewModdingAPI initialized with method: {self.method.method} on port: {self.port}", "DEBUG")

    def __str__(self) -> str:
        return "\n".join([f"{key}:\n{value}" for key, value in self.docs.__dict__.items() if isinstance(value, list)]) + "\n" + str(self.method)

    def hold_key(self, key: str, durationMS=1000):
        met = self.method.method
        command = None
        self.logger.log(
            f"Holding key: {key} for {durationMS}ms using method: {met}", "DEBUG")
        res = f"curl -X POST http://localhost:{self.port}/api/keyboard/hold -H \"Content-Type: application/json\" -d"
        if met == "tty":
            command = res + " '{\n"+"\"Key\": "+f"\"{key}\",\n" + \
                "\"DurationMs\": "+f"{durationMS},\n"+"}'"
            return self.method.subprocess_wrap(subprocess.check_output(command.split()))
        elif met == "ssh+tty":
            command = self.method.ssh_wrapper + \
                (res + " '{\n"+"\"Key\": "+f"\"{key}\",\n" +
                 "\"DurationMs\": "+f"{durationMS}\n"+"}'").split()
            return self.method.subprocess_wrap(subprocess.check_output(command))
        elif met == "docker":
            try:
                command = res + \
                    " '{\n"+"\"Key\": "+f"\"{key}\",\n" + \
                    "\"DurationMs\": "+f"{durationMS}\n"+"}'"
                result = self.method.docker_container.exec_run(
                    command, tty=True)
                if result.exit_code != 0:
                    raise RuntimeError(
                        f"Docker command failed with exit code {result.exit_code}: {result.output.decode('utf-8')}")
                return self.method.subprocess_wrap(result.output)
            except Exception as e:
                raise RuntimeError(f"Docker execution failed: {e}")
        else:
            raise ValueError("Create an API instance first")

    def map(self):
        self.logger.log(
            "map fetching, which is very heavy and computer intensive\nConsider caching or sumt idk", "WARNING")
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
                    raise RuntimeError(
                        f"Docker command failed with exit code {result.exit_code}: {result.output.decode('utf-8')}")
                return self.method.subprocess_wrap(result.output)
            except Exception as e:
                raise RuntimeError(f"Docker execution failed: {e}")
        else:
            raise ValueError("Create an API instance first")

    def skip_events(self, skip: bool = True):
        met = self.method.method
        command = None
        self.logger.log(
            f"skipping events with status {skip} using method: {met}", "DEBUG")
        strskip = "enable" if skip else "disable"

        res = (
            f"curl -X POST http://localhost:{self.port}/api/keyboard/autoskip "
            '-H "Content-Type: application/json" '
            "-d '{"
            f"\"Action\": \"{strskip}\", "
            "\"Events\": true, "
            "\"Dialogues\": true, "
            "\"Speed\": 10"
            "}'"
        )
        if strskip == "disable":
            res = (
                f"curl -X POST http://localhost:{self.port}/api/keyboard/autoskip "
                '-H "Content-Type: application/json" '
                "-d '{"
                f"\"Action\": \"{strskip}\""
                "}'"
            )
        res2 = (
                        f"curl -X POST http://localhost:{self.port}/api/keyboard/skip "
                        '-H "Content-Type: application/json" '
                        "-d '{"
                        "\"Type\": \"event\""
                        "}'"
                    )

        if met == "tty":
            command = res
            self.method.subprocess_wrap(subprocess.check_output(res2.split()))
            return self.method.subprocess_wrap(subprocess.check_output(command.split()))
        elif met == "ssh+tty":
            command = self.method.ssh_wrapper + (res).split()
            self.method.subprocess_wrap(subprocess.check_output(self.method.ssh_wrapper + res2.split()))
            return self.method.subprocess_wrap(subprocess.check_output(command))
        elif met == "docker":
            try:
                command = res
                self.method.docker_container.exec_run(
                    res2, tty=True)
                result = self.method.docker_container.exec_run(
                    command, tty=True)
                if result.exit_code != 0:
                    raise RuntimeError(
                        f"Docker command failed with exit code {result.exit_code}: {result.output.decode('utf-8')}")
                return self.method.subprocess_wrap(result.output)
            except Exception as e:
                raise RuntimeError(f"Docker execution failed: {e}")
        else:
            raise ValueError("Create an API instance first")
