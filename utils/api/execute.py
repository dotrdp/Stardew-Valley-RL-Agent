import json
import subprocess
import docker

class exec_method:
    def __init__(self, method, docker_image_name, ssh_user, ssh_host, ssh_port):
        self.method = method
        self.docker_image_name = docker_image_name
        self.ssh_wrapper = ["ssh", "-p", ssh_port, f"{ssh_user}@{ssh_host}"]
        if method == "docker" and docker_image_name:
            try:
                self.docker_client = docker.from_env()
                self.docker_container = self.docker_client.containers.get(
                    docker_image_name)
            except docker.errors.NotFound:  # type: ignore
                raise ValueError(
                    f"Docker container '{docker_image_name}' not found")
            except Exception as e:
                raise ValueError(f"Failed to connect to Docker: {e}")

    def __call__(self, **kwargs):
        if self.method == "tty":
            return self.subprocess_wrap(subprocess.check_output(list(self.API_wrap(**kwargs).split())))
        elif self.method == "ssh+tty":
            return self.subprocess_wrap(subprocess.check_output(self.ssh_wrapper + [self.API_wrap(**kwargs)]))
        elif self.method == "docker":
            if self.docker_image_name is None:
                raise ValueError(
                    "Docker image name must be provided for docker execution method.")
            try:
                command = self.API_wrap(wrap_escaping=False, **kwargs)
                result = self.docker_container.exec_run(command, tty=True)
                if result.exit_code != 0:
                    raise RuntimeError(
                        f"Docker command failed with exit code {result.exit_code}: {result.output.decode('utf-8')}")
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
            parameters = ", ".join(
                [rf'"{param}"' for param in parameters])  # type: ignore
        res = f"curl -s -X POST http://localhost:{port}/api/execute -H \"Content-Type: application/json\" -d"
        func = " '{\n"+"\"Target\": "+f"\"{target}\",\n"+"\"Method\": " + \
            f"\"{function}\",\n"+"\"Parameters\": "+f"[{parameters}]\n"+"}'"
        if wrap_escaping != True:
            return res + func.replace('\"', '"')
        return res + func

    def subprocess_wrap(self, bytes_output: bytes) -> dict:
        readable_output = bytes_output.decode("utf-8")
        return dict(json.loads(readable_output))


