"""
Chapter 7: Advanced Patterns and Error Handling
Example 5: Basic Retry Logic

Description:
Demonstrates implementing retry patterns with exponential backoff for resilient
agent operations. Shows how to handle transient failures automatically with
configurable retry strategies and comprehensive logging.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- tenacity package for retry logic
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter7.05_basic_retry_logic
```

Expected Output:
Retry logic demonstration:
1. Initial attempts fail with simulated errors
2. Exponential backoff delays between retries
3. Comprehensive logging of retry attempts
4. Eventual success after configured attempts
5. Graceful handling of persistent failures

Key Concepts:
- Exponential backoff retry strategies
- Tenacity library integration
- Retry attempt configuration
- Transient failure handling
- Retry logging and monitoring
- Failure simulation and testing
- Resilient service integration

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import logging
import sys
from pathlib import Path

# Third-party imports
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paper_fetch")

# Tool function with retry logic and logging
attempt_counter = {"count": 0}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def fetch_paper_with_retry(paper_id: str) -> str:
    """Fetch paper details with retry logic and exponential backoff."""
    attempt_counter["count"] += 1
    logger.info(f"Attempt {attempt_counter['count']} to fetch paper {paper_id}")
    if attempt_counter["count"] < 3:
        raise ValueError("Simulated fetch failure.")
    return f"Details for paper {paper_id}"

# Create a model client
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Create an agent with the retry tool
agent = AssistantAgent(
    name="paper_agent",
    model_client=model_client,
    tools=[fetch_paper_with_retry],
    system_message="You are a research assistant. Use the 'fetch_paper_with_retry' tool to fetch paper details. If an error occurs, retry up to 3 times with exponential backoff. Report the result or any error to the user."
)

async def main():
    paper_id = "12345"
    logger.info(f"Requesting details for paper: {paper_id}")
    try:
        response = await agent.run(task=f"Use the fetch_paper_with_retry tool to get details for paper {paper_id}.")
        logger.info("Agent completed the fetch task.")
        print(response)
    except RetryError as e:
        logger.error(f"All retries failed: {e}")
    except Exception as e:
        logger.error(f"Agent failed to complete the task: {e}")
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
