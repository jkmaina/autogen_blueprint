"""
Chapter 7: Advanced Patterns and Error Handling
Example 6: Basic Round Robin

Description:
Demonstrates round-robin agent coordination patterns for balanced workload
distribution. Shows how to create multi-agent systems where agents take
turns handling tasks in a structured, cyclic manner.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter7.06_basic_round_robin
```

Expected Output:
Round-robin coordination demonstration:
1. Three agents are configured in round-robin pattern
2. Task is initiated and agents take turns responding
3. Each agent contributes based on their specialized role
4. Conversation continues in predictable rotation
5. Process terminates after maximum message limit

Key Concepts:
- Round-robin scheduling algorithms
- Fair agent participation patterns
- Structured multi-agent conversations
- Turn-based collaboration
- Message limit termination conditions
- Balanced workload distribution
- Predictable agent ordering

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

async def main():
    # Create a model client (replace with your actual model and API key if needed)
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Define three simple agents
    agent1 = AssistantAgent(
        name="Agent1",
        system_message="You are Agent1, a helpful assistant in a round robin group.",
        model_client=model_client
    )
    agent2 = AssistantAgent(
        name="Agent2",
        system_message="You are Agent2, a helpful assistant in a round robin group.",
        model_client=model_client
    )
    agent3 = AssistantAgent(
        name="Agent3",
        system_message="You are Agent3, a helpful assistant in a round robin group.",
        model_client=model_client
    )

    # Create a RoundRobinGroupChat
    group_chat = RoundRobinGroupChat(
        [agent1, agent2, agent3],
        termination_condition=MaxMessageTermination(9),
    )

    # Run the group chat and print the result
    response = await group_chat.run(task="Discuss the main challenges in reinforcement learning.")
    print("\nRound Robin Group Chat response:")
    print(response)

    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
