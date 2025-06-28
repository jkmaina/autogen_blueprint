import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Create a model client with parallel tool calls disabled
model_client = OpenAIChatCompletionClient(model="gpt-4o", parallel_tool_calls=False)

# Create agents with handoff capabilities
agent1 = AssistantAgent(
    name="agent1",
    model_client=model_client,
    handoffs=["agent2", "agent3"],
    system_message="You are agent1. First, introduce yourself by saying 'This is agent1 speaking. I am the first agent.' Then hand off to agent2 if they haven't introduced themselves yet."
)
agent2 = AssistantAgent(
    name="agent2",
    model_client=model_client,
    handoffs=["agent1", "agent3"],
    system_message="You are agent2. First, introduce yourself by saying 'This is agent2 speaking. I am the second agent.' Then hand off to agent3 if they haven't introduced themselves yet."
)
agent3 = AssistantAgent(
    name="agent3",
    model_client=model_client,
    handoffs=["agent1", "agent2"],
    system_message="You are agent3. First, introduce yourself by saying 'This is agent3 speaking. I am the third agent.' After introducing yourself, say 'TERMINATE' since all agents have now introduced themselves."
)

# Create a Swarm
swarm = Swarm(
    participants=[agent1, agent2, agent3],
    termination_condition=TextMentionTermination("TERMINATE"),
)

async def main():
    stream = swarm.run_stream(task="Each agent should introduce themselves only ONCE and hand off to another agent who has not introduced themself. Terminate when all agents have introduced themselves.")
    await Console(stream)
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())