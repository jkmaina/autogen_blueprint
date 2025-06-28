"""
Chapter 1: AutoGen Fundamentals
Example 2: Agent with Tools

Description:
Demonstrates how to equip an AssistantAgent with custom tools it can use to perform tasks.
Shows how to define tool functions and make them available to the agent for dynamic execution.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed

Usage:
```bash
python -m chapter1.02_agent_with_tools
```

Expected Output:
The assistant will use the provided tools to:
1. Get the current time using the get_current_time tool
2. Calculate the area of a rectangle with length 5 and width 3 using the calculate_area tool
The responses will be streamed to the console, showing tool usage and results.

Key Concepts:
- Tool function definition with type hints
- Agent tool integration and configuration
- Streaming responses with Console UI
- Tool reflection and usage tracking
- Async tool execution

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

# --- Define Tools (as async functions with type hints) ---

async def get_current_time() -> str:
    """Return current date and time as a string."""
    now = datetime.now()
    return f"The current date and time is {now.strftime('%Y-%m-%d %H:%M:%S')}"

async def calculate_area(length: float, width: float) -> float:
    """Return the area of a rectangle."""
    return length * width

# --- Main async function ---

async def main() -> None:
    # Load config
    config = get_openai_config()

    # Create model client
    model_client = OpenAIChatCompletionClient(
        model=config["model"],
        api_key=config["api_key"]
    )

    # Create assistant with tools
    assistant = AssistantAgent(
        name="assistant_with_tools",
        system_message="You are a helpful AI assistant that can use tools to perform tasks.",
        model_client=model_client,
        model_client_stream=True,
        reflect_on_tool_use=True,
        tools=[get_current_time, calculate_area]
    )

    # Run task requiring tool usage
    print("Assistant is responding...\n")
    stream = assistant.run_stream(
        task="What time is it now? Also, calculate the area of a rectangle with length 5 and width 3."
    )

    # Display streamed output
    await Console(stream)

    # Cleanup
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
