"""
Chapter 9: Memory and Context Management
Example 3: Basic Memory Context

Description:
Demonstrates different MIME types and content formats in memory systems.
Shows how to store and manage various types of content including text,
JSON data, and structured information.

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import json
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

async def mime_types_example():
    memory = ListMemory(name="mixed_content")
    
    # Text content
    await memory.add(MemoryContent(
        content="User loves pizza",
        mime_type=MemoryMimeType.TEXT
    ))
    
    # JSON content
    user_profile = {
        "name": "Alice",
        "age": 30,
        "dietary_restrictions": ["vegetarian", "no_nuts"]
    }
    await memory.add(MemoryContent(
        content=json.dumps(user_profile),
        mime_type=MemoryMimeType.JSON,
        metadata={"type": "user_profile"}
    ))
    
    # Markdown content
    await memory.add(MemoryContent(
        content="## Meeting Notes\n- Discussed project timeline\n- Next deadline: March 15th",
        mime_type=MemoryMimeType.MARKDOWN,
        metadata={"type": "meeting_notes", "date": "2025-01-15"}
    ))
    
    # Create an agent with this memory
    assistant = AssistantAgent(
        name="memory_assistant",
        model_client=OpenAIChatCompletionClient(model="gpt-4o"),
        memory=[memory],
    )
    
    # The agent can use the context from memory to answer questions
    print("Testing agent's ability to access different memory types...")
    print("-" * 50)
    
    # Test 1: Accessing JSON and text data
    print("User: What does the user like to eat, and what are their dietary restrictions?")
    stream = assistant.run_stream(task="What does the user like to eat, and what are their dietary restrictions?")
    await Console(stream)
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Accessing Markdown data
    print("User: What was discussed in the last meeting?")
    stream = assistant.run_stream(task="What was discussed in the last meeting?")
    await Console(stream)


asyncio.run(mime_types_example())
