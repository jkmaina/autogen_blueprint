"""
Chapter 2: Advanced Agent Concepts
Example 1: AgentChat High-Level API

Description:
Demonstrates the high-level API provided by AgentChat for creating sophisticated
assistant agents. Shows how to build an architecture expert agent with streaming
capabilities and specialized knowledge.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed

Usage:
```bash
python -m chapter2.01_assistant_agent
```

Expected Output:
An architecture expert agent will stream its response explaining the relationship
between AutoGen Core and AgentChat components. The response will be displayed
in real-time using the Console UI.

Key Concepts:
- AgentChat high-level API usage
- Specialized agent persona creation
- Streaming responses with Console UI
- Model client configuration
- Expert domain knowledge integration

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config



async def main() -> None:
    # Create a model client
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)
    
    # Create an AssistantAgent
    assistant = AssistantAgent(
        name="architecture_expert",
        system_message="""You are an expert on AutoGen v0.6 architecture.
        Explain concepts clearly and provide examples when helpful.""",
        model_client=model_client,
        model_client_stream=True,
    )
    
    # Run the agent with a task related to AutoGen architecture
    print("Assistant is responding...")
    stream = assistant.run_stream(
        task="Explain the relationship between AutoGen Core and AgentChat components."
    )
    await Console(stream)
    
    # Close the model client connection
    await model_client.close()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
