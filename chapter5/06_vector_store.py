"""
Chapter 5: Code Execution and Tool Integration
Example 6: Vector Store Memory Integration

Description:
Demonstrates integrating vector-based memory storage with assistant agents using
ChromaDB. Shows how agents can remember user preferences and use them to provide
personalized responses across conversations.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ with memory extensions
- ChromaDB package installed
- Write permissions for persistent storage

Usage:
```bash
python -m chapter5.06_vector_store
```

Expected Output:
Vector memory integration demonstration:
1. Initializes ChromaDB vector memory with persistence
2. Stores user preference for metric units
3. Agent retrieves weather information
4. Memory influences response format (metric vs imperial)
5. Shows personalized weather response in Celsius

Key Concepts:
- Vector-based memory storage
- Persistent memory across sessions
- User preference learning
- Contextualized agent responses
- ChromaDB integration
- Memory-driven personalization
- Tool integration with memory

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
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

async def main():
    """Main execution function demonstrating vector memory integration."""
    
    async def get_weather(city: str, units: str = "imperial") -> str:
        """Sample weather tool that responds with different units."""
        if units == "imperial":
            return f"The weather in {city} is 73 °F and Sunny."
        elif units == "metric":
            return f"The weather in {city} is 23 °C and Sunny."
        else:
            return f"Sorry, I don't know the weather in {city}."

    try:
        print("=== Vector Memory Integration with ChromaDB ===")
        
        # Set up vector memory with persistent storage
        chroma_user_memory = ChromaDBVectorMemory(
            config=PersistentChromaDBVectorMemoryConfig(
                collection_name="preferences",
                persistence_path=os.path.join(str(Path.home()), ".chromadb_autogen"),
                k=2,                           # Retrieve top 2 similar memories
                score_threshold=0.2,           # Minimum similarity threshold
            )
        )

        # Add user preference entries
        await chroma_user_memory.clear()  # Clean slate for demonstration
        await chroma_user_memory.add(MemoryContent(
            content="The weather should be in metric units",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "preferences", "type": "units"},
        ))
        
        print("🧠 User preference stored: Weather should be in metric units")

        # Set up model and agent
        config = get_openai_config()
        model_client = OpenAIChatCompletionClient(**config)

        # Create assistant agent with memory and weather tool
        assistant_agent = AssistantAgent(
            name="WeatherAssistant",
            model_client=model_client,
            memory=[chroma_user_memory],       # Enable vector memory
            tools=[get_weather],               # Weather tool integration
            system_message="You are a helpful weather assistant. Use stored preferences when available."
        )

        # Task that should trigger memory retrieval
        task = "What is the weather in New York?"
        
        print(f"🌤️ Task: {task}")
        print("(Agent should remember user prefers metric units)")

        # Run the assistant with memory integration
        stream = assistant_agent.run_stream(task=task)
        await Console(stream)
        
        # Cleanup
        await model_client.close()
        await chroma_user_memory.close()
        
        print("
✅ Vector memory integration demonstration complete!")
        
    except Exception as e:
        print(f"Error during vector memory demonstration: {e}")


if __name__ == "__main__":
    asyncio.run(main())
