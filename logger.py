
class LogLevel:
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

    @classmethod
    def to_string(cls, level):
        if level == cls.DEBUG:
            return "[DEBUG]"
        elif level == cls.INFO:
            return "[INFO]"
        elif level == cls.WARNING:
            return "[WARNING]"
        elif level == cls.ERROR:
            return "[ERROR]"
        elif level == cls.CRITICAL:
            return "[FATAL]"
        else:
            return "UNKNOWN"
    @classmethod
    def get_value(cls, level_str):
        if level_str == "DEBUG":
            return cls.DEBUG
        elif level_str == "INFO":
            return cls.INFO
        elif level_str == "WARNING":
            return cls.WARNING
        elif level_str == "ERROR":
            return cls.ERROR
        elif level_str == "FATAL":
            return cls.CRITICAL
        else:
            raise ValueError(f"Unknown log level: {level_str}")

class Logger:
    def __init__(self, level):
        self.level = level
    def log(self, message, level):
        if LogLevel.get_value(level) >= self.level:
            print(f"{LogLevel.to_string(LogLevel.get_value(level))}: {message}")

