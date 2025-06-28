"""
Chapter 7: Advanced Patterns and Error Handling
Example 1: Basic Circuit Breaker Pattern

Description:
Demonstrates implementing a circuit breaker pattern for reliable agent operations.
Shows how to protect against cascading failures when external services become
unreliable, with automatic recovery and failure threshold management.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter7.01_basic_circuit_breaker
```

Expected Output:
Circuit breaker demonstration:
1. Successful search operation with circuit closed
2. Multiple failures trigger circuit breaker to open
3. Subsequent calls are blocked while circuit is open
4. Automatic reset after timeout period
5. Resilient error handling for unreliable services

Key Concepts:
- Circuit breaker pattern implementation
- Failure threshold management
- Automatic recovery mechanisms
- Service resilience patterns
- Error handling and logging
- Agent tool protection
- Cascading failure prevention

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import logging
import sys
import time
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("circuit_breaker_demo")

class CircuitBreaker:
    """Circuit breaker implementation for protecting against service failures."""
    
    def __init__(self, failure_threshold=3, reset_timeout=10):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
        self.is_open = False

    async def execute(self, func, *args, **kwargs):
        current_time = time.time()
        # Reset circuit if enough time has passed
        if self.is_open and current_time - self.last_failure_time > self.reset_timeout:
            self.is_open = False
            self.failure_count = 0
        # If circuit is open, block calls
        if self.is_open:
            raise Exception("Circuit breaker is open")
        try:
            result = await func(*args, **kwargs)
            self.failure_count = 0  # Reset on success
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
            raise e

# Simulated flaky search function
async def search_academic_papers(query: str) -> str:
    if "fail" in query:
        raise ValueError("Simulated search failure.")
    return f"Results for '{query}'"

# Create a circuit breaker for API calls
paper_search_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=10)

# Define a safer search function
async def search_papers_with_circuit_breaker(query: str) -> str:
    try:
        return await paper_search_breaker.execute(search_academic_papers, query)
    except Exception as e:
        logger.error(f"Error searching papers: {str(e)}")
        return "Unable to search for papers at this time. Let's try a different approach."

async def main():
    """Main execution function demonstrating circuit breaker pattern."""
    try:
        print("=== Circuit Breaker Pattern Demonstration ===")
        
        # Try a successful search
        logger.info("Testing successful search operation")
        result = await search_papers_with_circuit_breaker("ai agents")
        print(f"✅ Success: {result}")
        
        # Simulate failures to trigger the circuit breaker
        logger.info("Testing circuit breaker with failures")
        for i in range(4):
            logger.info(f"Failure test {i+1}/4")
            result = await search_papers_with_circuit_breaker("fail this search")
            print(f"❌ Attempt {i+1}: {result}")
            
        logger.info("Circuit breaker demonstration complete")
        
    except Exception as e:
        logger.error(f"Error in circuit breaker demo: {e}")
        print(f"Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
