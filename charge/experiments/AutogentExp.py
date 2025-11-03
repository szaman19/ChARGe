from Experiment import Experiment
from charge.clients.AgentPool import Agent, AgentPool
from charge.tasks.Task import Task
from typing import Any, List, Union

try:
    import autogen
    from charge.clients.autogen import AutoGenPool
except ImportError:

    raise ImportError(
        "The autogen package is required for AutogenExperiment. Please install it via 'pip install autogen'."
    )


class AutogenExperiment(Experiment):
    def __init__(
        self, task: Union[Task, List[Task]], agent_pool: AutoGenPool, *args, **kwargs
    ):
        super().__init__(task=task, agent_pool=agent_pool, *args, **kwargs)
        # Initialize Autogen specific parameters here
        # For example:

    def save_agent_state(self, agent):
        # Implement saving the state of the Autogen agent
        pass

    def add_to_context(self, agent: Agent, task: Task, result):
        # Implement adding the result to the context of the Autogen task
        pass

    def save_state(self):
        # Implement saving the state of the Autogen experiment
        pass

    def load_state(self, state):
        # Implement loading the state of the Autogen experiment
        pass
