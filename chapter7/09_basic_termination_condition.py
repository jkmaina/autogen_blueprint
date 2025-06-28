import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console

# Create a model client
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Define three agents
agent1 = AssistantAgent(
    name="agent1",
    model_client=model_client,
    system_message=(
        "You are agent1. Participate in the round robin chat. "
        "For each response, read the previous messages, find the highest number used, "
        "and respond with the next number and a new discussion point. "
        "Stop when 8 points have been made."
    )
)
agent2 = AssistantAgent(
    name="agent2",
    model_client=model_client,
    system_message=(
        "You are agent2. Participate in the round robin chat. "
        "For each response, read the previous messages, find the highest number used, "
        "and respond with the next number and a new discussion point. "
        "Stop when 8 points have been made."
    )
)

# To end after 8 messages, we use 9 to account for the user message
max_messages = MaxMessageTermination(9)

# Use the termination condition in a group chat
group_chat = RoundRobinGroupChat(
    [agent1, agent2],
    termination_condition=max_messages,    
)

async def main():
    stream = group_chat.run_stream(
        task="Discuss the benefits of multi-agent collaboration. Number each discussion point sequentially, starting from 1, and stop after 8 points."
    )
    await Console(stream)
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
