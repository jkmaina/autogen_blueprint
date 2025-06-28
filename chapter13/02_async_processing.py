"""
Chapter 13: Performance Optimization and Deployment
Example 2: Asynchronous Processing Optimization

Description:
Demonstrates performance optimization through asynchronous processing patterns
including concurrent agent task execution, non-blocking I/O operations, and
comparative analysis of sequential vs parallel execution strategies.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support
- OAI_CONFIG_LIST configuration file

Usage:
```bash
python -m chapter13.02_async_processing
```

Expected Output:
Async processing optimization demonstration:
1. Sequential execution baseline measurement
2. Parallel execution with asyncio.gather()
3. Performance improvement calculations
4. Non-blocking file operation examples
5. Concurrent task completion timing
6. Optimization strategy comparison

Key Concepts:
- Asynchronous processing patterns
- Concurrent agent task execution
- asyncio.gather() for parallel operations
- Non-blocking I/O operations
- Sequential vs parallel performance
- Task concurrency optimization
- Performance measurement and benchmarking
- Scalable async architectures

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
import time
from pathlib import Path

# Third-party imports
import autogen
from autogen import AssistantAgent, UserProxyAgent

# Local imports
sys.path.append(str(Path(__file__).parent.parent))

async def main():
    """Main function to demonstrate asynchronous processing optimization."""
    # Load model configurations
    config_list = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-3.5-turbo"],
        },
    )
    
    # Create three specialized agents
    research_agent = AssistantAgent(
        name="research_agent",
        llm_config={"config_list": config_list},
        system_message="You are a research assistant who provides information on specific topics."
    )
    
    writing_agent = AssistantAgent(
        name="writing_agent",
        llm_config={"config_list": config_list},
        system_message="You are a writing assistant who creates well-structured content."
    )
    
    fact_checking_agent = AssistantAgent(
        name="fact_checking_agent",
        llm_config={"config_list": config_list},
        system_message="You are a fact-checking assistant who verifies information."
    )
    
    # Create user proxy agent
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
    )
    
    # Define queries for each agent
    queries = {
        "research": "Provide three key facts about renewable energy sources.",
        "writing": "Write a short paragraph about the importance of exercise.",
        "fact_checking": "Verify if the following statement is true: 'The Great Wall of China is visible from space with the naked eye.'"
    }
    
    # Function to run a single agent query
    async def run_agent_query(agent, query):
        start_time = time.time()
        await user_proxy.a_initiate_chat(agent, message=query)
        end_time = time.time()
        return end_time - start_time
    
    # 1. Sequential execution (non-optimized)
    print("\n=== Sequential Execution (Non-Optimized) ===")
    sequential_start = time.time()
    
    research_time = await run_agent_query(research_agent, queries["research"])
    print(f"Research query completed in {research_time:.2f} seconds")
    
    writing_time = await run_agent_query(writing_agent, queries["writing"])
    print(f"Writing query completed in {writing_time:.2f} seconds")
    
    fact_checking_time = await run_agent_query(fact_checking_agent, queries["fact_checking"])
    print(f"Fact-checking query completed in {fact_checking_time:.2f} seconds")
    
    sequential_total = time.time() - sequential_start
    print(f"Total sequential execution time: {sequential_total:.2f} seconds")
    
    # 2. Parallel execution (optimized with asyncio)
    print("\n=== Parallel Execution (Optimized with asyncio) ===")
    parallel_start = time.time()
    
    # Run all queries concurrently
    tasks = [
        run_agent_query(research_agent, queries["research"]),
        run_agent_query(writing_agent, queries["writing"]),
        run_agent_query(fact_checking_agent, queries["fact_checking"])
    ]
    
    # Wait for all tasks to complete
    task_times = await asyncio.gather(*tasks)
    
    parallel_total = time.time() - parallel_start
    print(f"Research query completed in {task_times[0]:.2f} seconds")
    print(f"Writing query completed in {task_times[1]:.2f} seconds")
    print(f"Fact-checking query completed in {task_times[2]:.2f} seconds")
    print(f"Total parallel execution time: {parallel_total:.2f} seconds")
    
    # Performance improvement summary
    improvement = (sequential_total - parallel_total) / sequential_total * 100
    print("\n=== Performance Improvement Summary ===")
    print(f"Sequential execution: {sequential_total:.2f} seconds")
    print(f"Parallel execution: {parallel_total:.2f} seconds")
    print(f"Time saved: {sequential_total - parallel_total:.2f} seconds ({improvement:.1f}%)")
    
    # Additional async optimization example: non-blocking file operations
    print("\n=== Non-Blocking File Operations Example ===")
    
    async def write_results_to_file(filename, content):
        # Simulate file I/O with sleep
        await asyncio.sleep(0.5)  # Non-blocking sleep
        print(f"Results written to {filename}")
    
    # Run file operations concurrently
    file_tasks = [
        write_results_to_file("research_results.txt", "Research data"),
        write_results_to_file("writing_results.txt", "Written content"),
        write_results_to_file("fact_check_results.txt", "Verification results")
    ]
    
    await asyncio.gather(*file_tasks)
    print("All file operations completed concurrently")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAsync processing demo interrupted by user")
    except Exception as e:
        print(f"Error running async processing demo: {e}")
        raise
