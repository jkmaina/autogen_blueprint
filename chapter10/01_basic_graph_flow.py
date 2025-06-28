"""
Chapter 10: Workflow Patterns and Graph Execution
Example 1: Basic Graph Flow

Description:
Demonstrates fundamental graph-based workflow patterns with sequential
agent execution. Shows how to create simple linear workflows where agents
pass tasks to each other in a structured, predictable manner.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter10.01_basic_graph_flow
```

Expected Output:
Basic sequential workflow demonstration:
1. Writer agent creates initial content draft
2. Reviewer agent analyzes and provides feedback
3. Sequential task progression with clear handoffs
4. Structured workflow execution patterns
5. Linear agent coordination example

Key Concepts:
- Sequential workflow patterns
- Linear agent coordination
- Graph-based task progression
- Structured agent handoffs
- Basic workflow construction
- Sequential execution flows
- Simple graph patterns

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
    """Run a basic sequential flow with writer and reviewer agents"""
    
    # Create an OpenAI model client
    client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    
    # Create the writer agent
    writer = AssistantAgent(
        name="writer", 
        model_client=client, 
        system_message="You are a professional writer. Draft a short paragraph on the requested topic."
    )
    
    # Create the reviewer agent
    reviewer = AssistantAgent(
        name="reviewer", 
        model_client=client, 
        system_message="You are an editor. Review the draft and suggest specific improvements for clarity and impact."
    )
    
    # Build the graph: writer -> reviewer
    builder = DiGraphBuilder()
    builder.add_node(writer).add_node(reviewer)
    
    # Set writer as the entry point
    builder.set_entry_point(writer)
    
    # Add edge from writer to reviewer
    builder.add_edge(writer, reviewer)
    
    # Build and validate the graph
    graph = builder.build()
    
    # Create the flow with a termination condition
    flow = GraphFlow(
        participants=[writer, reviewer],
        graph=graph,
        termination_condition=MaxMessageTermination(5)  # Terminate after 5 messages
    )
    
    # Run the flow
    async for event in flow.run_stream(task="Write a short paragraph about artificial intelligence."):
        print(f"{event}")

if __name__ == "__main__":
    asyncio.run(main())