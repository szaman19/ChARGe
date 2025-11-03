from Experiment import Experiment
from charge.clients.AgentPool import Agent, AgentPool
from charge.tasks.Task import Task
from typing import Any, List, Union
import asyncio

try:
    from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
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
        self.model_context = ListMemory()

    def save_agent_state(self, agent):
        # Implement saving the state of the Autogen agent
        pass

    def add_to_context(self, agent: Agent, task: Task, result):
        # Implement adding the result to the context of the Autogen task

        # This is tricky and should be customized based on the use case
        # We set the default behavior to add the instruction
        #  result to the model context

        instruction = task.get_user_prompt()
        result = str(result)

        content = MemoryContent(
            content=f"Instruction: {instruction}\nResponse: {result}",
            mime_type=MemoryMimeType.TEXT,
        )
        asyncio.run(self.model_context.add(content))

    def save_state(self):
        # Implement saving the state of the Autogen experiment

        # Get the current memory content
        all_memory_content = self.model_context.content
        state = {
            "memory": all_memory_content,
            "finished_tasks": self.finished_tasks,
            "remaining_tasks": self.tasks,
            "agent_pool": self.agent_pool,  # May need to customize based on agent pool implementation
        }
        return state

    def load_state(self, state):
        # Implement loading the state of the Autogen experiment

        memory_content = state.get("memory", [])
        self.model_context = ListMemory()
        for content in memory_content:
            asyncio.run(self.model_context.add(content))

        self.finished_tasks = state.get("finished_tasks", [])
        self.tasks = state.get("remaining_tasks", [])
        self.agent_pool = state.get("agent_pool", self.agent_pool)

    def create_agent_with_experiment_state(self, task):
        return self.agent_pool.create_agent(task=task, memory=[self.model_context])
