"""
Chapter 10: Workflow Patterns and Graph Execution
Example 3: Parallel Flow

Description:
Demonstrates parallel execution patterns where multiple agents work
concurrently on different aspects of the same task. Shows how to
coordinate simultaneous agent activities and aggregate results.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter10.03_parallel_flow
```

Expected Output:
Parallel workflow demonstration:
1. Multiple specialized researchers work concurrently
2. Technology, market, and social impact analysis in parallel
3. Simultaneous task execution patterns
4. Coordinated parallel agent activities
5. Aggregated multi-perspective results

Key Concepts:
- Parallel execution patterns
- Concurrent agent coordination
- Multi-threaded workflow design
- Specialized agent roles
- Parallel task distribution
- Simultaneous processing
- Result aggregation patterns

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

async def main():
    """Run a parallel execution flow where multiple agents work concurrently"""
    
    """Run a parallel execution flow where multiple agents work concurrently."""
    
    # Create an OpenAI model client
    config = get_openai_config()
    client = OpenAIChatCompletionClient(**config)
    
    # Create specialized research agents
    tech_researcher = AssistantAgent(
        "tech_researcher", 
        model_client=client, 
        system_message="You are a technology researcher. Research and provide insights on the technical aspects of the given topic."
    )
    
    market_researcher = AssistantAgent(
        "market_researcher", 
        model_client=client, 
        system_message="You are a market researcher. Research and provide insights on the market aspects of the given topic."
    )
    
    social_researcher = AssistantAgent(
        "social_researcher", 
        model_client=client, 
        system_message="You are a social impact researcher. Research and provide insights on the social implications of the given topic."
    )
   
    
    

  

    
    # Create the synthesizer agent
    synthesizer = AssistantAgent(
        "synthesizer", 
        model_client=client, 
        system_message="""You are a research synthesizer. 
        Combine the insights from multiple research perspectives into a comprehensive report.
        Clearly identify the technical, market, and social aspects in your synthesis."""
    )
    
    # Build the graph with parallel execution
    builder = DiGraphBuilder()
    
    # Add all nodes
    builder.add_node(tech_researcher)
    builder.add_node(market_researcher)
    builder.add_node(social_researcher)
    builder.add_node(synthesizer)
    
    # Fan out to all researchers (parallel execution)
    # In this case, all researchers are source nodes
    
    # Fan in from all researchers to synthesizer
    builder.add_edge(tech_researcher, synthesizer)
    builder.add_edge(market_researcher, synthesizer)
    builder.add_edge(social_researcher, synthesizer)
    
    # Build and validate the graph
    graph = builder.build()
    
    # Create the flow with a termination condition
    flow = GraphFlow(
        participants=[tech_researcher, market_researcher, social_researcher, synthesizer], 
        graph=graph,
        termination_condition=MaxMessageTermination(5)  # Terminate after 5 messages
    )
    
    # Run the flow
    async for event in flow.run_stream(task="Research the impact of artificial intelligence on healthcare."):
        print(f"{event}")