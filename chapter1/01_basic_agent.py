"""
Chapter 1: AutoGen Fundamentals
Example 1: Basic Assistant Agent

Description:
Demonstrates the minimal setup required to create and use a basic AssistantAgent.
This is the foundational example showing how to initialize an agent with OpenAI
and execute a simple task.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed

Usage:
```bash
python -m chapter1.01_basic_agent
```

Expected Output:
The script creates a basic assistant agent and asks it to explain what AutoGen is.
The agent's response will be printed to the console, demonstrating successful
agent initialization and task execution.

Key Concepts:
- AssistantAgent creation and configuration
- OpenAI model client integration
- Basic task execution with asyncio
- Environment configuration management

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

# Load OpenAI API configuration
config = get_openai_config()

# Create a basic assistant agent
openai_client = OpenAIChatCompletionClient(
    model=config['model'],
    api_key=config['api_key'],
)
assistant = AssistantAgent(
    name="BasicAssistant",    
    model_client=openai_client,
    system_message="You are a helpful assistant."
)

# Send a message and get a response
response = asyncio.run(assistant.run(task="What is AutoGen?"))
print("\nAssistant’s response:")
print(response)
