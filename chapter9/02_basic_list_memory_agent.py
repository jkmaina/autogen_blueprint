"""
Chapter 9: Memory and Context Management
Example 2: Basic List Memory Agent

Description:
Demonstrates integrating ListMemory with assistant agents for context-aware
conversations. Shows how agents can remember user preferences and maintain
contextual information across interactions.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ with memory extensions
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter9.02_basic_list_memory_agent
```

Expected Output:
Memory-enabled agent demonstration:
1. Creates ListMemory for user preferences
2. Stores user preference information
3. Agent integrates memory into responses
4. Context-aware conversation handling
5. Memory-driven personalization
6. Persistent preference application

Key Concepts:
- Agent memory integration
- Context-aware conversations
- User preference storage
- Memory-driven responses
- Personalized agent behavior
- Contextual information retention
- Agent memory systems

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

async def basic_memory_example():
    # Step 1: Create a memory store
    user_memory = ListMemory(name="user_preferences")
    
    # Step 2: Add some user preferences to memory
    await user_memory.add(MemoryContent(
        content="The user prefers metric units for temperature",
        mime_type=MemoryMimeType.TEXT
    ))
   
    def get_weather(location: str, unit: str) ->str:
        if unit == "metric":
            return "It's 25°C in " + location
        return "It's 75°F in " + location
    
    # Step 3: Create an agent with memory
    assistant = AssistantAgent(
        name="helpful_assistant",
        model_client=OpenAIChatCompletionClient(model="gpt-4o"),
        memory=[user_memory],  # Pass memory as a list
        tools=[get_weather]
    )


    
    # Step 4: Test the agent - it should remember preferences
    stream = assistant.run_stream(task="What's the weather like in London?")
    await Console(stream)
    
    print("\n" + "="*50)
    print("Memory contents:")

#run the example
asyncio.run(basic_memory_example())