import asyncio
from typing import Sequence
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import TerminationCondition
from autogen_agentchat.messages import BaseChatMessage, ToolCallExecutionEvent, StopMessage
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Custom termination: stop when a specific function is called
class FunctionCallTermination(TerminationCondition):
    def __init__(self, function_name: str) -> None:
        self._terminated = False
        self._function_name = function_name

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(self, messages: Sequence[BaseChatMessage]) -> StopMessage | None:
        if self._terminated:
            return StopMessage(
                content=f"Function '{self._function_name}' was executed.",
                source="FunctionCallTermination",
            )
        for message in messages:
            if isinstance(message, ToolCallExecutionEvent):
                for execution in message.content:
                    if execution.name == self._function_name:
                        self._terminated = True
                        return StopMessage(
                            content=f"Function '{self._function_name}' was executed.",
                            source="FunctionCallTermination",
                        )
        return None

    async def reset(self) -> None:
        self._terminated = False

# Example tool function
def approve() -> None:
    """Approve the message when all feedbacks have been addressed."""
    pass

# Model client
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Agents
primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant."
)
critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    tools=[approve],
    system_message="Provide constructive feedback. Use the approve tool to approve when all feedbacks are addressed."
)

# Termination condition: stop when 'approve' is called
function_call_termination = FunctionCallTermination(function_name="approve")

# Team
round_robin_team = RoundRobinGroupChat(
    [primary_agent, critic_agent],
    termination_condition=function_call_termination
)

async def main():
    await Console(round_robin_team.run_stream(task="Write a unique, Haiku about the weather in Paris"))
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
