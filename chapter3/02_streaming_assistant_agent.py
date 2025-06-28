"""
Chapter 3: Agent Communication Patterns
Example 2: Streaming Assistant Agent

Description:
Demonstrates how to use streaming with an AssistantAgent to see responses as they're generated.
Streaming provides a better user experience by showing content as it's created in real-time
rather than waiting for the complete response.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed

Usage:
```bash
python -m chapter3.02_streaming_assistant_agent
```

Expected Output:
The assistant will stream its response about the benefits of streaming in LLM applications,
showing the text as it's being generated in real-time rather than waiting for the complete
response. Demonstrates the streaming capabilities of AgentChat.

Key Concepts:
- Streaming response generation
- Real-time content display with Console UI
- Agent streaming configuration
- Async stream handling
- Enhanced user experience patterns

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

async def main():
    # Create a model client
    config = get_openai_config()
    client = OpenAIChatCompletionClient(**config)
    
    # Create an AssistantAgent with streaming enabled
    agent = AssistantAgent(
        name="streaming_assistant",
        model_client=client,
        model_client_stream=True  # Enable streaming
    )
    
    # Run the agent with a task and stream the response
    print("Assistant is responding with streaming...\n")
    stream = agent.run_stream(
        task="Explain the benefits of streaming in LLM applications."
    )
    
    # Display the streaming output
    await Console(stream)
    
    # Clean up
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
