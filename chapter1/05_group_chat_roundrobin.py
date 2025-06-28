"""
Chapter 1: AutoGen Fundamentals
Example 5: Group Chat Round Robin

Description:
Demonstrates a conversation between multiple agents using RoundRobinGroupChat.
Shows how to create a group chat with multiple agents that take turns responding
in a structured round-robin format.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed

Usage:
```bash
python -m chapter1.05_group_chat_roundrobin
```

Expected Output:
A group chat conversation between a poet, scientist, and philosopher discussing the concept of time,
with each agent responding in turn according to their unique persona. The conversation will
terminate after 9 total messages (3 rounds).

Key Concepts:
- Round-robin group chat mechanics
- Multiple agent coordination
- Turn-based conversation flow
- Agent persona differentiation
- Conversation termination conditions

AutoGen Version: 0.5+
"""

import sys
from pathlib import Path
import asyncio

# Setup imports and environment
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main() -> None:
    # Load config
    config = get_openai_config()

    # Create shared OpenAI model client
    model_client = OpenAIChatCompletionClient(
        model=config["model"],
        api_key=config["api_key"]
    )

    # Define agents with unique personas
    poet = AssistantAgent(
        name="poet",
        system_message="You are a creative poet who speaks in verse. Keep your responses brief and poetic.",
        model_client=model_client,
    )

    scientist = AssistantAgent(
        name="scientist",
        system_message="You are a precise scientist who explains concepts clearly with facts. Keep your responses brief and factual.",
        model_client=model_client,
    )

    philosopher = AssistantAgent(
        name="philosopher",
        system_message="You are a thoughtful philosopher who considers the deeper meaning of concepts. Keep your responses brief and thought-provoking.",
        model_client=model_client,
    )

    # Terminate after 9 total messages (3 rounds)
    termination = MaxMessageTermination(9)

    # Set up RoundRobinGroupChat
    group_chat = RoundRobinGroupChat(
        participants =[poet, scientist, philosopher],
        termination_condition=termination,
    )

    # Start the group chat
    print("Starting group chat...\n")
    task = "Discuss the concept of time from your unique perspectives."

    # Stream the conversation to the console
    await Console(group_chat.run_stream(task=task))

    # Clean up
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
