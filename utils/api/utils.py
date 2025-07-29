import base64
import msgpack

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

