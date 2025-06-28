"""
Chapter 1: AutoGen Fundamentals
Example 3: Basic Streaming

Description:
Demonstrates how to use streaming with an AssistantAgent to see responses as they're generated.
Streaming provides a better user experience by showing content as it's created rather than 
waiting for the complete response.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed

Usage:
```bash
python -m chapter1.03_basic_streaming
```

Expected Output:
The assistant will stream its response about multi-agent systems in AI to the console,
showing the text as it's being generated in real-time rather than waiting for the 
complete response. This demonstrates the streaming capabilities of AutoGen.

Key Concepts:
- Streaming response generation
- Console UI for real-time display
- Agent configuration for streaming
- Async stream handling
- Real-time user experience

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
    # Load config with environment variables
    config = get_openai_config()

    # Create OpenAI client with streaming
    model_client = OpenAIChatCompletionClient(
        model=config["model"],
        api_key=config["api_key"]
    )

    # Create assistant with streaming enabled
    assistant = AssistantAgent(
        name="StreamingAssistant",
        system_message="You are a helpful AI assistant that provides detailed explanations.",
        model_client=model_client,
        model_client_stream=True  # 🔑 Enables streaming
    )

    # Run the agent task with streaming
    print("Assistant is responding...\n")
    stream = assistant.run_stream(task="Explain the concept of multi-agent systems in AI.")
    
    # Display streaming output using the console
    await Console(stream)

    # Clean up
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
