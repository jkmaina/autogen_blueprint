"""
Chapter 7: Advanced Patterns and Error Handling
Example 4: Basic Recovery Agent

Description:
Demonstrates error handling and recovery patterns using specialized recovery agents.
Shows how to implement automatic error detection, handler delegation, and
recovery strategies in multi-agent systems.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter7.04_basic_recovery_agent
```

Expected Output:
Recovery agent demonstration:
1. Main agent encounters an error condition
2. Error signal triggers handler agent activation
3. Error handler analyzes the situation
4. Recovery strategy is suggested and implemented
5. System continues operation with recovered state

Key Concepts:
- Error detection and signaling patterns
- Specialized recovery agent roles
- Automatic error handler delegation
- Recovery strategy implementation
- Multi-agent error coordination
- Fault tolerance in agent systems
- Error analysis and resolution

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

# Create a model client
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Main agent
main_agent = AssistantAgent(
    name="main_agent",
    model_client=model_client,
    system_message="You are a research assistant. If you encounter a problem, say 'ERROR' and the error_handler agent will help."
)

# Recovery/Error handler agent
error_handler = AssistantAgent(
    name="error_handler",
    model_client=model_client,
    system_message="You are an Error Handler agent. When the main_agent says 'ERROR', analyze the conversation, identify the issue, and suggest a recovery strategy."
)

def selector(messages):
    # If the last message contains 'ERROR', let the error_handler respond
    if "ERROR" in messages[-1].content:
        return "error_handler"
    # Otherwise, let the main agent respond
    return "main_agent"

team = SelectorGroupChat(
    [main_agent, error_handler],
    selector_func=selector,
    model_client=model_client,
    max_turns=4
)

async def main():
    # Simulate a task that may fail and require recovery
    task = (
        "Try to summarize the latest research on 'quantum teleportation' using the quantum tool. "
        "If you encounter an error or don't know the answer, say 'ERROR'."
    )
    response = await team.run(task=task)
    print("\nTeam Response:")
    print(response)
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
