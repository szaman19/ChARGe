import atexit
import readline


def enable_cmd_history_and_shell_integration(history: str):
    try:
        readline.read_history_file(history)
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass

    atexit.register(readline.write_history_file, history)
