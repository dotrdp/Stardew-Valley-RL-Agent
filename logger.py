from colorama import Fore
class LogLevel:
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

    @classmethod
    def to_string(cls, level):
        if level == cls.DEBUG:
            return Fore.GREEN+"[DEBUG]"
        elif level == cls.INFO:
            return Fore.LIGHTYELLOW_EX+"[INFO]"
        elif level == cls.WARNING:
            return Fore.YELLOW+"[WARNING]"
        elif level == cls.ERROR:
            return Fore.LIGHTRED_EX+"[ERROR]"
        elif level == cls.CRITICAL:
            return Fore.RED+"[FATAL]"
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
        elif level_str == "CRITICAL":
            return cls.CRITICAL
        else:
            raise ValueError(f"Unknown log level: {level_str}")

class Logger:
    def __init__(self, level):
        self.level = LogLevel.get_value(level)
    def log(self, message, level):
        if LogLevel.get_value(level) >= self.level:
            print(f"{LogLevel.to_string(LogLevel.get_value(level))}: {message}")

