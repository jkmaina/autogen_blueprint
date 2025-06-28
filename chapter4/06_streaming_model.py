"""
Chapter 4: AutoGen Core Fundamentals
Example 6: Streaming Model Integration

Description:
Demonstrates integrating streaming language models with AutoGen Core agents.
Shows how to create an assistant agent with OpenAI streaming capabilities for
real-time content generation and display through the Console UI.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed with extensions
- autogen-ext package for OpenAI integration

Usage:
```bash
python -m chapter4.06_streaming_model
```

Expected Output:
A streaming sci-fi story generation demonstration:
1. Assistant agent receives creative writing task
2. OpenAI model generates story content in real-time
3. Console UI displays streaming text as it arrives
4. Complete robot story with creative narrative elements

Key Concepts:
- Streaming language model integration
- OpenAI client configuration
- Real-time content generation
- Assistant agent with model streaming
- Console UI for stream display
- Creative writing task execution
- Model client lifecycle management

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import os
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
    """Main execution function demonstrating streaming model integration."""
    try:
        # Configuration and setup
        config = get_openai_config()
        client = OpenAIChatCompletionClient(**config)
        
        # Create assistant agent with streaming capabilities
        agent = AssistantAgent(
            name="assistant",
            system_message="You are a creative sci-fi writer.",
            model_client=client,
            model_client_stream=True     # Enable streaming output
        )
        
        # Execute streaming creative writing task
        print("=== Streaming Sci-Fi Story Generation ===")
        await Console(agent.run_stream(
            task="Tell me a story about a robot."
        ))
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Cleanup
        try:
            await client.close()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(main())
