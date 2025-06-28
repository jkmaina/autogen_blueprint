"""
Chapter 3: Agent Communication Patterns
Example 8: Streaming Assistant Agent Advanced

Description:
Demonstrates advanced streaming output options for AgentChat with multiple approaches
to handle streaming responses. Shows three different patterns for processing and
displaying streaming content from agents.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- dotenv package installed

Usage:
```bash
python -m chapter3.08_streaming_assistant_agent
```

Expected Output:
Three different streaming examples demonstrating:
1. Real-time message printing as content arrives
2. Console UI with statistics (tokens used, duration)
3. Collecting streaming chunks for post-processing
Each approach shows different ways to handle streaming agent responses.

Key Concepts:
- Multiple streaming output patterns
- Real-time content processing
- Console UI with output statistics
- Stream event handling and filtering
- Chunk collection and aggregation
- Performance monitoring in streaming

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
    """Main execution function demonstrating multiple streaming patterns."""
    # Configuration and setup
    config = get_openai_config()
    client = OpenAIChatCompletionClient(**config)
    
    # Create assistant agent with streaming enabled
    agent = AssistantAgent(
        name="assistant",
        model_client=client,
        model_client_stream=True           # Enable streaming output
    )
    
    print("=== Streaming Pattern 1: Real-time Event Processing ===")
    # Option 1: Print each message as it arrives
    async for event in agent.run_stream(task="List 10 cities in North America"):
        # Check if the event has content to print
        if hasattr(event, 'content') and event.content:
            print(event, end='', flush=True)
    
    print("\n\n=== Streaming Pattern 2: Console UI with Statistics ===")
    # Option 2: Use Console helper with statistics
    # Displays live output plus summary with tokens used and duration
    await Console(
        agent.run_stream(task="List 10 cities in Europe"),
        output_stats=True                  # Show performance statistics
    )
    
    print("\n=== Streaming Pattern 3: Chunk Collection ===")
    # Option 3: Collect all streamed chunks for post-processing
    # Useful for applications that need the complete response
    chunks = []
    async for event in agent.run_stream(task="List 10 cities in Africa"):
        if hasattr(event, 'content'):
            chunks.append(event.content)
    
    final_response = ''.join(chunks)
    print(f"Collected Response Length: {len(final_response)} characters")
    print(f"Final Response: {final_response}")
    
    # Cleanup
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())

