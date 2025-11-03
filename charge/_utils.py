import atexit
import readline
import inspect


def enable_cmd_history_and_shell_integration(history: str):
    """Enable persistent command-line history and integrate with the interactive shell.
    Attempts to load an existing readline history file, sets the in-memory history
    length to 1000, and registers an atexit handler to persist history on exit.

    Args:
        history (str): Path to the history file to read from and write to.

    Returns:
        None

    - If the history file exists, it is loaded into readline.
    - If the history file does not exist, the function continues silently.
    - Other exceptions raised by readline.read_history_file may be propagated.
    """
    try:
        readline.read_history_file(history)
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass

    atexit.register(readline.write_history_file, history)


async def maybe_await(func, *args, **kwargs):
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result
