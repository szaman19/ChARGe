import argparse


def add_server_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add standard server arguments to an argparse parser.
    Args:
        parser (argparse.ArgumentParser): The parser to add arguments to.
    """
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--host", type=str, default="http://127.0.0.1", help="Host to run the server on"
    )
