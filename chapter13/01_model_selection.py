"""
Chapter 13: Performance Optimization and Deployment
Example 1: Model Selection Optimization

Description:
Demonstrates strategic model selection for performance optimization including
cost-performance balancing, tiered approaches based on task complexity, and
intelligent routing to appropriate models for different types of queries.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support
- OAI_CONFIG_LIST configuration file

Usage:
```bash
python -m chapter13.01_model_selection
```

Expected Output:
Model selection optimization demonstration:
1. Simple queries routed to GPT-3.5 Turbo (cost-effective)
2. Complex reasoning queries routed to GPT-4 (high capability)
3. Code generation using GPT-4 (best quality)
4. Performance timing comparisons
5. Cost-performance optimization strategy
6. Tiered model selection summary

Key Concepts:
- Strategic model selection
- Cost-performance optimization
- Tiered model routing
- Task complexity assessment
- Response time measurement
- Multi-model agent systems
- Performance benchmarking
- Cost-effective AI deployment

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
    """Main function to demonstrate model selection optimization strategies."""
    # Load model configurations
    config_list = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-4", "gpt-3.5-turbo"],
        },
    )
    
    # Filter configurations for specific models
    gpt4_config = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-4"],
        },
    )
    
    gpt35_config = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-3.5-turbo"],
        },
    )
    
    # Create agents with different models for different purposes
    
    # 1. Basic assistant using GPT-3.5 Turbo for simple tasks
    assistant_basic = AssistantAgent(
        name="assistant_basic",
        llm_config={"config_list": gpt35_config},
        system_message="You are a helpful assistant who provides brief, concise answers."
    )
    
    # 2. Advanced assistant using GPT-4 for complex reasoning
    assistant_advanced = AssistantAgent(
        name="assistant_advanced",
        llm_config={"config_list": gpt4_config},
        system_message="You are a sophisticated assistant who provides detailed analysis and reasoning."
    )
    
    # 3. Code assistant using GPT-4 for programming tasks
    code_assistant = AssistantAgent(
        name="code_assistant",
        llm_config={"config_list": gpt4_config},
        system_message="You are a coding expert who writes efficient, well-documented code."
    )
    
    # Create user proxy agent
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
    )
    
    # Function to measure response time
    async def measure_response_time(agent, query):
        start_time = time.time()
        await user_proxy.a_initiate_chat(agent, message=query)
        end_time = time.time()
        return end_time - start_time
    
    # Test with different queries to demonstrate model selection optimization
    
    # 1. Simple factual query - use GPT-3.5 Turbo (faster, cheaper)
    simple_query = "What is the capital of France?"
    print(f"\n=== Simple Query (GPT-3.5 Turbo) ===")
    simple_time = await measure_response_time(assistant_basic, simple_query)
    print(f"Response time: {simple_time:.2f} seconds")
    
    # 2. Complex reasoning query - use GPT-4 (more capable)
    complex_query = "Analyze the potential long-term economic impacts of increasing automation in manufacturing industries."
    print(f"\n=== Complex Query (GPT-4) ===")
    complex_time = await measure_response_time(assistant_advanced, complex_query)
    print(f"Response time: {complex_time:.2f} seconds")
    
    # 3. Coding query - use GPT-4 (better for code)
    code_query = "Write a Python function that implements the merge sort algorithm."
    print(f"\n=== Code Query (GPT-4) ===")
    code_time = await measure_response_time(code_assistant, code_query)
    print(f"Response time: {code_time:.2f} seconds")
    
    # Summary of optimization strategy
    print("\n=== Model Selection Optimization Summary ===")
    print(f"Simple query (GPT-3.5): {simple_time:.2f} seconds")
    print(f"Complex query (GPT-4): {complex_time:.2f} seconds")
    print(f"Code query (GPT-4): {code_time:.2f} seconds")
    print("\nOptimization strategy:")
    print("1. Use GPT-3.5 Turbo for simple, factual queries (faster, cheaper)")
    print("2. Use GPT-4 for complex reasoning and analysis (more capable)")
    print("3. Use GPT-4 for code generation (better quality code)")
    print("\nThis tiered approach optimizes for both cost and performance.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nModel selection demo interrupted by user")
    except Exception as e:
        print(f"Error running model selection demo: {e}")
        raise
