"""
ChatCompletionCache example for AutoGen v0.5

This demonstrates how to use ChatCompletionCache to improve performance by caching model responses.
"""

import asyncio
import sys
import os
import time
import logging

# Add the parent directory to the path so we can import the utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.cache import ChatCompletionCache, CHAT_CACHE_VALUE_TYPE
from autogen_ext.cache_store.diskcache import DiskCacheStore
from diskcache import Cache

from utils.config import get_openai_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

async def run_with_cache() -> None:
    """
    Run an example with ChatCompletionCache to demonstrate caching of model responses.
    """
    logger.info("Starting ChatCompletionCache example")
    
    # Create a model client using the configuration from utils
    config = get_openai_config()
    base_model_client = OpenAIChatCompletionClient(**config)
    
    # Initialize the CacheStore with diskcache
    cache_store = DiskCacheStore[CHAT_CACHE_VALUE_TYPE](Cache(CACHE_DIR))
    cached_model_client = ChatCompletionCache(base_model_client, cache_store)
    
    # Create an AssistantAgent with the cached model client
    assistant = AssistantAgent(
        name="cached_assistant",
        system_message="You are a helpful assistant focused on providing concise responses.",
        model_client=cached_model_client,
    )
    
    # Define a set of questions to ask
    questions = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of France?",  # Repeated to demonstrate cache hit
        "What is the capital of Germany?",
        "What is the capital of Japan?",    # Repeated to demonstrate cache hit
    ]
    
    # Run through the questions and measure response times
    for i, question in enumerate(questions):
        logger.info(f"Question {i+1}: {question}")
        
        start_time = time.time()
        response = await assistant.run(task=question)
        end_time = time.time()
        
        logger.info(f"Response time: {end_time - start_time:.2f} seconds")
        logger.info(f"Response: {response}")
        logger.info("-" * 50)
    
    # Close the model client connection
    await cached_model_client.close()

async def run_with_seed() -> None:
    """
    Run an example with ChatCompletionCache using a seed for deterministic responses.
    """
    logger.info("Starting ChatCompletionCache with seed example")
    
    # Create a model client using the configuration from utils
    config = get_openai_config()
    config["seed"] = 42  # Set a seed for deterministic responses
    base_model_client = OpenAIChatCompletionClient(**config)
    
    # Initialize the CacheStore with diskcache
    cache_store = DiskCacheStore[CHAT_CACHE_VALUE_TYPE](Cache(os.path.join(CACHE_DIR, "seeded")))
    cached_model_client = ChatCompletionCache(base_model_client, cache_store)
    
    # Create an AssistantAgent with the seeded model client
    assistant = AssistantAgent(
        name="seeded_assistant",
        system_message="You are a creative assistant that generates ideas.",
        model_client=cached_model_client,
    )
    
    # Run the same creative task multiple times to demonstrate deterministic outputs
    creative_task = "Generate a unique name for a sci-fi spaceship"
    
    logger.info(f"Creative task: {creative_task}")
    logger.info("Running the same task multiple times with seed=42:")
    
    for i in range(3):
        start_time = time.time()
        response = await assistant.run(task=creative_task)
        end_time = time.time()
        
        logger.info(f"Run {i+1} - Response time: {end_time - start_time:.2f} seconds")
        logger.info(f"Response: {response}")
    
    # Close the model client connection
    await cached_model_client.close()

async def main() -> None:
    """
    Main function to demonstrate ChatCompletionCache.
    """
    await run_with_cache()
    await run_with_seed()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
