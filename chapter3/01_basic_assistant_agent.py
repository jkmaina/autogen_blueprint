"""
Chapter 3: Agent Communication Patterns
Example 1: Basic Assistant Agent

Description:
Demonstrates a basic Hello World AgentChat application using the high-level
AssistantAgent API. This is the foundational example for AgentChat patterns.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed

Usage:
```bash
python -m chapter3.01_basic_assistant_agent
```

Expected Output:
The agent will respond with "Hello World!" demonstrating basic agent
task execution and response generation.

Key Concepts:
- Basic AgentChat application setup
- AssistantAgent instantiation
- Simple task execution
- Agent response handling
- Client lifecycle management

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


async def main():
    """Main execution function demonstrating basic agent usage."""
    # Configuration and setup
    config = get_openai_config()
    
    # Create OpenAI client
    client = OpenAIChatCompletionClient(**config)
    
    # Create basic assistant agent
    agent = AssistantAgent(
        name="assistant",           # Agent identifier
        model_client=client         # OpenAI client instance
    )
    
    # Execute task and get response
    result = await agent.run(task="Say Hello World!")
    
    # Display results
    print(result.messages[-1].content)
    
    # Cleanup
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())