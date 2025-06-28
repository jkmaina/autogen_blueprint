"""
Chapter 1: AutoGen Fundamentals
Example 6: Complete Multi-Agent System

Description:
Demonstrates a complete multi-agent system with tools and group chat coordination.
Combines multiple concepts: specialized agents, tool usage, and round-robin group chat 
to create a comprehensive multi-agent workflow.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed

Usage:
```bash
python -m chapter1.06_complete_multi_agent_system
```

Expected Output:
A group chat conversation between a coordinator, math specialist, and science specialist.
Each agent will use their specialized tools to solve specific problems:
1. The coordinator will get the current time
2. The math specialist will calculate the area of a circle
3. The science specialist will convert temperature from Celsius to Fahrenheit
The coordinator will summarize the findings at the end.

Key Concepts:
- Multi-agent group chat coordination
- Specialized agent roles with custom tools
- Round-robin conversation management
- Tool integration and usage
- Agent termination conditions

AutoGen Version: 0.5+
"""

import asyncio
import sys
from datetime import datetime
import math
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to sys.path for utils import
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_openai_config

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ----------------------- Tool Definitions -----------------------

async def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"The current date and time is {now.strftime('%Y-%m-%d %H:%M:%S')}"

async def calculate_circle_area(radius: float) -> float:
    """Calculate the area of a circle given its radius."""
    return math.pi * (radius ** 2)

async def convert_temperature(celsius: float) -> float:
    """Convert temperature from Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32

# ----------------------- Main Entry -----------------------

async def main() -> None:
    # Load OpenAI config
    load_dotenv()
    config = get_openai_config()

    # Create shared OpenAI client
    model_client = OpenAIChatCompletionClient(
        model=config["model"],
        api_key=config["api_key"]
    )

    # Define agents with tools and personas
    coordinator = AssistantAgent(
        name="coordinator",
        system_message="""
        You are the coordinator who manages the conversation.
        Start by introducing the task and then ask specific questions to the appropriate specialist.
        Summarize findings at the end.
        """,
        model_client=model_client,
        tools=[get_current_time]
    )

    math_specialist = AssistantAgent(
        name="math_specialist",
        system_message="""
        You are a mathematics specialist who can perform calculations.
        Explain your approach clearly when solving problems.
        """,
        model_client=model_client,
        tools=[calculate_circle_area],
        reflect_on_tool_use=True
    )

    science_specialist = AssistantAgent(
        name="science_specialist",
        system_message="""
        You are a science specialist who can explain scientific concepts
        and perform scientific calculations.
        """,
        model_client=model_client,
        tools=[convert_temperature],
        reflect_on_tool_use=True
    )

    # Define termination condition (12 total messages)
    termination = MaxMessageTermination(12)

    # Set up group chat with participants
    group_chat = RoundRobinGroupChat(
        participants=[coordinator, math_specialist, science_specialist],
        termination_condition=termination
    )

    # Define task
    task = """
    We need to solve the following problems:
    1. What is the current time?
    2. What is the area of a circle with radius 5 meters?
    3. Convert 25 degrees Celsius to Fahrenheit.

    Each specialist should handle the appropriate tasks, and the coordinator should summarize the findings.
    """

    print("Starting specialized group chat...\n")
    await Console(group_chat.run_stream(task=task))

    # Cleanup
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
