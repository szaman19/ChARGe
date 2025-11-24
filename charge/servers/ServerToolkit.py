################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from mcp.server.fastmcp import FastMCP
from functools import wraps
from typing import Callable, Literal
import time


class ServerToolkit:
    """
    A class that provides a toolkit for registering methods as MCP tools.
    """

    def __init__(self, mcp: FastMCP):
        self._mcp = mcp

    def _register_methods(self):
        """
        Register all methods marked with the @mcp_tool decorator.
        """
        for name in dir(self):
            # Skip private/magic methods
            if name.startswith("_"):
                continue

            attr = getattr(self, name)

            # Check if this is a method marked for registration
            if callable(attr) and hasattr(attr, "_is_mcp_tool"):
                self._register_single_method(attr)

    def _register_single_method(self, method: Callable):
        """
        Register a single method as an MCP tool.

        Args:
            method (Callable): The method to register as an MCP tool.

        Returns:
            None
        """

        # Create a closure that captures the bound method
        @wraps(method)
        def tool_wrapper(*args, **kwargs):
            return method(*args, **kwargs)

        self._mcp.tool()(tool_wrapper)

    @staticmethod
    def mcp_tool(func: Callable) -> Callable:
        """
        Decorator to mark methods for MCP registration.

        Args:
            func (Callable): The function to mark for MCP registration.

        Returns:
            Callable: The marked function.
        """
        func._is_mcp_tool = True
        return func

    @staticmethod
    def register_function_as_tool(mcp: FastMCP, func: Callable) -> None:
        """
        Register an external function as an MCP tool.

        Args:
            mcp (FastMCP): The MCP instance to register the function with.
            func (Callable): The function to register as an MCP tool.

        Returns:
            None
        """
        mcp.tool()(func)

    def register_function_to_server(func: Callable) -> None:
        """
        Register an external function to the server.

        Args:
            func (Callable): The function to register to the server.

        Returns:
            None
        """
        self._mcp.tool()(func)

    def return_mcp(self) -> FastMCP:
        """
        Return the MCP instance.

        Returns:
            FastMCP: The MCP instance.
        """
        self._register_methods()
        return self._mcp

    def run(self, transport: Literal["sse", "stdio"] = "sse") -> None:
        """
        Run the MCP server.

        Args:
            transport (Literal["sse", "stdio"], optional): The transport to use. Defaults to "sse".
        """
        self._register_methods()
        self._mcp.run(transport=transport)

    def update_mcp(self, mcp: FastMCP) -> None:
        """
        Update the MCP instance.

        Args:
            mcp (FastMCP): The new MCP instance.
        """
        self._mcp = mcp
        self._register_methods()


class MultiServerToolkit(ServerToolkit):
    """
    A class to combine multiple servers into a single object that
    can be run as a single MCP server.
    """

    def __init__(
        self,
        servers: list[ServerToolkit],
        description: str,
        host: str,
        port: int,
    ):
        """
        Initialize the MultiServerToolkit.

        Args:
            servers (list[ServerToolkit]): The list of servers to register.
            description (str): The description of the server.
            host (str): The host of the server.
            port (int): The port of the server.
        """
        mcp = FastMCP(description, host=host, port=port)
        super().__init__(mcp)

        self.servers = servers
        # Register all tools from all servers
        for server in self.servers:
            for name in dir(server):
                if name.startswith("_"):
                    continue

                attr = getattr(server, name)
                if callable(attr) and hasattr(attr, "_is_mcp_tool"):
                    self.mcp.tool()(attr)


class ToolKitLauncher:
    """
    A class to launch on a seperate process a server with a toolkit
    """

    def __init__(self, toolkit: ServerToolkit):
        self.toolkit = toolkit
        self.process = None

    def start(self, transport: Literal["sse", "streamable-http"] = "sse"):
        """
        Run the toolkit in a separate process.

        Args:
            transport (Literal["sse", "streamable-http"], optional): The transport to use. Defaults to "sse".
        """

        import multiprocessing

        self.process = multiprocessing.Process(
            target=self.toolkit.run, args=(transport,), daemon=True
        )
        self.process.start()
        self.wait_for_start()

    def wait_for_start(self, timeout: int = 10):
        """
        Wait for the process to start.

        Args:
            timeout (int, optional): The timeout in seconds. Defaults to 10.
        """

        start_time = time.time()
        while not self.is_running():
            if time.time() - start_time > timeout:
                raise TimeoutError("Process did not start within the timeout period.")
            time.sleep(0.1)

    def stop(self):
        """
        Stop the process.
        """
        if self.process is None:
            return
        self.process.terminate()
        self.process.join()
        self.process = None

    def is_running(self) -> bool:
        """
        Check if the process is running.

        Returns:
            bool: True if the process is running, False otherwise.
        """
        return self.process is not None and self.process.is_alive()

    def get_process(self) -> Optional[multiprocessing.Process]:
        """
        Get the process.

        Returns:
            Optional[multiprocessing.Process]: The process.
        """
        return self.process

    def __del__(self):
        """
        Destructor to stop the process.
        """
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
