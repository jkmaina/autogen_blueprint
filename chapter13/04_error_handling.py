"""
Chapter 13: Performance Optimization and Deployment
Example 4: Error Handling and Resilience

Description:
Demonstrates robust error handling and resilience patterns including retry
mechanisms with exponential backoff, circuit breaker patterns, graceful
degradation strategies, and comprehensive failure recovery techniques.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support
- OAI_CONFIG_LIST configuration file

Usage:
```bash
python -m chapter13.04_error_handling
```

Expected Output:
Error handling and resilience demonstration:
1. Retry mechanisms with exponential backoff
2. Circuit breaker pattern activation
3. Graceful degradation with fallback systems
4. Failure recovery and state management
5. Performance statistics and metrics
6. Resilience pattern effectiveness analysis

Key Concepts:
- Retry mechanisms with exponential backoff
- Circuit breaker patterns
- Graceful degradation strategies
- Fallback system implementation
- Failure rate monitoring
- System resilience design
- Error propagation control
- Recovery time optimization

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Third-party imports
import autogen
from autogen import AssistantAgent, UserProxyAgent

# Local imports
sys.path.append(str(Path(__file__).parent.parent))

# Simulated API client with potential failures
class SimulatedLLMClient:
    def __init__(self, failure_rate=0.3, max_consecutive_failures=3):
        self.failure_rate = failure_rate
        self.consecutive_failures = 0
        self.max_consecutive_failures = max_consecutive_failures
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.circuit_open = False
        self.circuit_reset_time = None
    
    async def generate_response(self, prompt, model="primary"):
        self.total_calls += 1
        
        # Check if circuit breaker is open
        if self.circuit_open:
            if datetime.now() < self.circuit_reset_time:
                print(f"Circuit breaker open. Failing fast.")
                self.failed_calls += 1
                raise Exception("Circuit breaker open - failing fast")
            else:
                print(f"Circuit breaker reset time reached. Closing circuit.")
                self.circuit_open = False
                self.consecutive_failures = 0
        
        # Simulate API call with potential failure
        if random.random() < self.failure_rate:
            self.consecutive_failures += 1
            self.failed_calls += 1
            
            # Check if we should open the circuit breaker
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.circuit_open = True
                self.circuit_reset_time = datetime.now().replace(second=datetime.now().second + 10)
                print(f"Too many consecutive failures ({self.consecutive_failures}). Opening circuit breaker until {self.circuit_reset_time}.")
            
            raise Exception(f"Simulated API failure (attempt {self.consecutive_failures})")
        
        # Successful response
        self.consecutive_failures = 0
        self.successful_calls += 1
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        if model == "primary":
            return f"Detailed response to: {prompt}"
        else:
            return f"Simple response to: {prompt}"

# Retry mechanism with exponential backoff
async def retry_with_exponential_backoff(func, max_retries=5, base_delay=1):
    retries = 0
    while retries < max_retries:
        try:
            return await func()
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                print(f"Maximum retries ({max_retries}) reached. Giving up.")
                raise
            
            # Calculate backoff delay: 2^retries + random jitter
            delay = (2 ** retries) * base_delay + random.uniform(0, 1)
            print(f"Attempt {retries} failed with error: {e}. Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
    
    raise Exception("Retry mechanism failed")

# Graceful degradation with fallback
async def with_fallback(primary_func, fallback_func, max_primary_attempts=3):
    try:
        # Try primary function with retries
        return await retry_with_exponential_backoff(primary_func, max_retries=max_primary_attempts)
    except Exception as e:
        print(f"Primary function failed after {max_primary_attempts} attempts. Using fallback.")
        # Use fallback function
        return await fallback_func()

async def main():
    """Main function to demonstrate error handling and resilience patterns."""
    # Create simulated LLM client
    llm_client = SimulatedLLMClient(failure_rate=0.4)
    
    # Load model configurations
    config_list = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-3.5-turbo"],
        },
    )
    
    # Create assistant agent
    assistant = AssistantAgent(
        name="assistant",
        llm_config={"config_list": config_list},
    )
    
    # Create user proxy agent
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
    )
    
    # 1. Demonstrate retry mechanism
    print("\n=== Retry Mechanism with Exponential Backoff ===")
    
    query = "Explain the concept of machine learning"
    
    async def primary_function():
        return await llm_client.generate_response(query)
    
    try:
        start_time = time.time()
        result = await retry_with_exponential_backoff(primary_function, max_retries=4)
        elapsed_time = time.time() - start_time
        print(f"Success after {elapsed_time:.2f} seconds: {result}")
    except Exception as e:
        print(f"All retries failed: {e}")
    
    # 2. Demonstrate circuit breaker pattern
    print("\n=== Circuit Breaker Pattern ===")
    
    # Reset client for demonstration
    llm_client.consecutive_failures = 0
    llm_client.circuit_open = False
    
    # Make several calls to trigger circuit breaker
    for i in range(10):
        try:
            result = await llm_client.generate_response(f"Query {i}")
            print(f"Call {i} succeeded: {result}")
        except Exception as e:
            print(f"Call {i} failed: {e}")
    
    # 3. Demonstrate graceful degradation with fallback
    print("\n=== Graceful Degradation with Fallback ===")
    
    # Reset client for demonstration
    llm_client.consecutive_failures = 0
    llm_client.circuit_open = False
    
    async def complex_query():
        return await llm_client.generate_response("Provide a detailed analysis of quantum computing", model="primary")
    
    async def simple_fallback():
        return await llm_client.generate_response("What is quantum computing?", model="fallback")
    
    result = await with_fallback(complex_query, simple_fallback)
    print(f"Final result: {result}")
    
    # 4. Performance statistics
    print("\n=== Performance Statistics ===")
    print(f"Total API calls: {llm_client.total_calls}")
    print(f"Successful calls: {llm_client.successful_calls}")
    print(f"Failed calls: {llm_client.failed_calls}")
    success_rate = llm_client.successful_calls / llm_client.total_calls * 100 if llm_client.total_calls > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    # Summary of error handling benefits
    print("\n=== Error Handling Benefits ===")
    print("1. Retry Mechanism: Handles transient failures automatically")
    print("2. Circuit Breaker: Prevents cascading failures and system overload")
    print("3. Graceful Degradation: Ensures system continues functioning with reduced capabilities")
    print("4. Combined Approach: Maximizes reliability while maintaining performance")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nError handling demo interrupted by user")
    except Exception as e:
        print(f"Error running error handling demo: {e}")
        raise
