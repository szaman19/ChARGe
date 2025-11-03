import pytest
import os


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY", None) is None, reason="OPENAI_API_KEY not set"
)
class TestOpenAISimpleTask:
    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        from charge.tasks.Task import Task
        from charge.clients.autogen import AutoGenPool
        from charge.experiments import AutogenExperiment
        from pydantic import BaseModel

        self.agent_pool = AutoGenPool(model="gpt-5", backend="openai")

        first_task = Task(
            system_prompt="You are a helpful assistant, that is capable of"
            + "doing arithmetic and returning an explanation of how you arrived at the"
            + "answer.",
            user_prompt="What is 10 plus 5?",
        )

        class MathExplanationSchema(BaseModel):
            answer: int
            explanation: str

        second_task = Task(
            system_prompt="You are a helpful assistant, that is capable of"
            + "take an answer and explanation and convert it a structured JSON format.",
            user_prompt="Take the previous answer and explanation"
            + "and convert it into a JSON",
            structured_output_schema=MathExplanationSchema,
        )

        self.experiment = AutogenExperiment(
            task=[first_task, second_task], agent_pool=self.agent_pool
        )

        third_task = Task(
            system_prompt="You are a helpful assistant that can parse JSON from text.",
            user_prompt="Extract the 'answer' field from the previous JSON.",
        )

        # Don't add the third task to the experiment yet
        self.third_task = third_task

    @pytest.mark.asyncio
    async def test_linear_experiment_run(self):
        import re

        await self.experiment.run()
        finished_tasks = self.experiment.get_finished_tasks()
        assert len(finished_tasks) == 2

        first_task_result = finished_tasks[0]
        second_task_result = finished_tasks[1]

        print("First Task Result:", first_task_result)
        print("Second Task Result:", second_task_result)

        assert "15" in first_task_result
        assert re.search(r'"answer":\s*15', second_task_result)

        self.state = self.experiment.save_state()

        self.experiment.load_state(self.state)

        self.experiment.add_task(self.third_task)
        assert self.experiment.remaining_tasks() == 1

        await self.experiment.run()
        finished_tasks = self.experiment.get_finished_tasks()
        assert len(finished_tasks) == 3

        third_task_result = finished_tasks[2]
        print("Third Task Result:", third_task_result)
        assert "15" in third_task_result
