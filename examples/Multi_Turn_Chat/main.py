import argparse
import asyncio
from charge.tasks.Task import Task
from typing import Optional, Union
from charge.clients.Client import Client
from charge.clients.autogen import AutoGenPool

parser = argparse.ArgumentParser()

# Add system prompt argument
parser.add_argument(
    "--system-prompt",
    type=str,
    default=None,
    help="Custom system prompt (optional, uses default chemistry prompt if not provided)",
)

# Add standard CLI arguments
Client.add_std_parser_arguments(parser)

# Default system prompt
DEFAULT_SYSTEM_PROMPT = (
    "You are a world-class chemist. Your task is to generate unique molecules "
    "based on the lead molecule provided by the user. The generated molecules "
    "should be chemically valid and diverse, exploring different chemical spaces "
    "while maintaining some structural similarity to the lead molecule. "
    "Provide the final answer in a clear and concise manner."
)


class ChargeChatTask(Task):
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        server_urls: Optional[Union[str, list]] = None,
    ):
        # Use provided system prompt or fall back to default
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        super().__init__(system_prompt=system_prompt, user_prompt=None)
        print("ChargeChatTask initialized with the provided prompts.")

        self.system_prompt = system_prompt
        self.user_prompt = None


if __name__ == "__main__":

    args = parser.parse_args()
    server_url = args.server_urls

    mytask = ChargeChatTask(
        system_prompt=args.system_prompt,
        server_urls=server_url,
    )

    agent_pool = AutoGenPool(model=args.model, backend=args.backend)

    agent = agent_pool.create_agent(task=mytask)

    agent_state = asyncio.run(agent.chat())

    print(f"Task completed. Results: {agent_state}")
