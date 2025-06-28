"""
Simple AutoGen ContextPlus example for AutoGen v0.5

This demonstrates how to use context enhancement for agents in a simple, learner-friendly way.
"""

import asyncio
import sys
import os
import logging

# Add the parent directory to the path so we can import the utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autogen_agentchat.agents import AssistantAgent
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient

from utils.config import get_openai_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """
    Main function to demonstrate a simple context-enhanced agent.
    """
    # Create a model client using the configuration from utils
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)
    
    # Create a simple memory with programming language information
    memory = ListMemory(name="programming_knowledge")
    
    # Add programming language information to memory
    await memory.add(
        MemoryContent(
            content="""
            Python is a high-level, interpreted programming language known for its readability and simplicity.
            Key features include: easy to learn syntax, dynamic typing, object-oriented programming support,
            extensive standard library, and large community ecosystem.
            """,
            mime_type=MemoryMimeType.TEXT,
            metadata={"topic": "python"}
        )
    )
    
    await memory.add(
        MemoryContent(
            content="""
            JavaScript is primarily used for web development. Key features include:
            client-side scripting, event-driven programming, prototype-based object orientation,
            and first-class functions.
            """,
            mime_type=MemoryMimeType.TEXT,
            metadata={"topic": "javascript"}
        )
    )
    
    # Create an AssistantAgent with memory
    assistant = AssistantAgent(
        name="programming_assistant",
        system_message="""You are a helpful programming assistant.
        Use the information in your memory to provide accurate responses about programming languages.
        If you don't have information about a topic, simply acknowledge that limitation.""",
        model_client=model_client,
        memory=[memory],  # Attach the memory to the agent
    )
    
    # Ask questions about programming languages
    questions = [
        "What are the key features of Python?",
        "What is JavaScript primarily used for?",
        "Tell me about Rust programming language."  # Not in our memory
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        response = await assistant.run(task=question)
        print(f"Answer: {response}")
    
    # Close the model client connection
    await model_client.close()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
