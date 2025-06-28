import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console

# Create a model client
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Define three agents
agent1 = AssistantAgent(
    name="agent1",
    model_client=model_client,
    system_message="You are agent1. Participate in the round robin chat. Each time, propose a step to complete the task, numbering your step. When you reach step 5, say 'TASK COMPLETE' and stop."
)
agent2 = AssistantAgent(
    name="agent2",
    model_client=model_client,
    system_message="You are agent2. Participate in the round robin chat. Each time, propose a step to complete the task, numbering your step. When you reach step 5, say 'TASK COMPLETE' and stop."
)
agent3 = AssistantAgent(
    name="agent3",
    model_client=model_client,
    system_message="You are agent3. Participate in the round robin chat. Each time, propose a step to complete the task, numbering your step. When you reach step 5, say 'TASK COMPLETE' and stop."
)

# End when "TASK COMPLETE" is mentioned
task_complete = TextMentionTermination("TASK COMPLETE")

# Use the termination condition in a group chat
group_chat = RoundRobinGroupChat(
    [agent1, agent2, agent3],
    termination_condition=task_complete,
)

async def main():
    stream = group_chat.run_stream(
        task="Discuss the steps to complete the assigned task. Number each step. When you reach step 5, stop."
    )
    await Console(stream)
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
