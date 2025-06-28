"""
Performance Optimization example for AutoGen v0.5

This demonstrates various techniques for optimizing performance in AutoGen applications.
"""

import asyncio
import sys
import os
import time
from typing import Dict, Any, List, Optional
import logging
from functools import wraps

# Add the parent directory to the path so we can import the utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

from utils.config import get_openai_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define a decorator for timing functions
def timing_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

@timing_decorator
async def run_single_agent() -> None:
    """
    Run a single agent example with performance monitoring.
    """
    logger.info("Starting single agent example")
    
    # Create a model client using the configuration from utils
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)
    
    # Create an AssistantAgent with the model client
    assistant = AssistantAgent(
        name="optimized_assistant",
        system_message="You are a helpful assistant focused on providing concise responses.",
        model_client=model_client,
    )
    
    # Run the agent with a task
    task = "Explain the concept of performance optimization in software development in one paragraph."
    logger.info(f"Running task: {task}")
    
    start_time = time.time()
    response = await assistant.run(task=task)
    end_time = time.time()
    
    logger.info(f"Task completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Response received: {response}")
    
    # Close the model client connection
    await model_client.close()

@timing_decorator
async def run_with_streaming() -> None:
    """
    Run an example with streaming to improve perceived performance.
    """
    logger.info("Starting streaming example")
    
    # Create a model client using the configuration from utils
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)
    
    # Create an AssistantAgent with streaming enabled
    assistant = AssistantAgent(
        name="streaming_assistant",
        system_message="You are a helpful assistant that provides detailed explanations.",
        model_client=model_client,
        model_client_stream=True,  # Enable streaming
    )
    
    # Run the agent with a task that requires a longer response
    task = "Explain the history and evolution of artificial intelligence from its inception to the present day."
    logger.info(f"Running streaming task: {task}")
    
    # Stream the response to the console
    await Console(assistant.run_stream(task=task))
    
    # Close the model client connection
    await model_client.close()

@timing_decorator
async def run_parallel_tasks() -> None:
    """
    Run multiple tasks in parallel for better throughput.
    """
    logger.info("Starting parallel tasks example")
    
    # Create a model client using the configuration from utils
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)
    
    # Create an AssistantAgent with the model client
    assistant = AssistantAgent(
        name="parallel_assistant",
        system_message="You are a helpful assistant that provides concise responses.",
        model_client=model_client,
    )
    
    # Define multiple tasks
    tasks = [
        "What is machine learning?",
        "Explain natural language processing.",
        "What is computer vision?",
    ]
    
    logger.info(f"Running {len(tasks)} tasks in parallel")
    
    # Run tasks in parallel
    start_time = time.time()
    responses = await asyncio.gather(*(assistant.run(task=task) for task in tasks))
    end_time = time.time()
    
    logger.info(f"All tasks completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Average time per task: {(end_time - start_time) / len(tasks):.2f} seconds")
    
    # Close the model client connection
    await model_client.close()

async def main() -> None:
    """
    Main function to demonstrate performance optimization techniques.
    """
    # Run examples with different optimization techniques
    await run_single_agent()
    await run_with_streaming()
    await run_parallel_tasks()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
