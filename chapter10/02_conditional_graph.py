"""
Chapter 10: Workflow Patterns and Graph Execution
Example 2: Conditional Graph

Description:
Demonstrates conditional branching workflows with decision-based routing.
Shows how to create adaptive workflows that change execution paths based
on agent responses and evaluation criteria.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter10.02_conditional_graph
```

Expected Output:
Conditional workflow demonstration:
1. Writer creates initial content
2. Reviewer evaluates using specific criteria
3. Conditional branching based on review feedback
4. Different paths for revision, editing, or approval
5. Adaptive workflow execution patterns

Key Concepts:
- Conditional workflow branching
- Decision-based routing
- Adaptive execution paths
- Review-driven workflows
- Conditional graph construction
- Dynamic workflow adaptation
- Response-based flow control

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
    """Run a conditional branching flow with different paths based on review feedback"""
    
    # Create an OpenAI model client
    client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    
    # Create the writer agent
    writer = AssistantAgent(
        "writer", 
        model_client=client, 
        system_message="You are a professional writer. Draft content based on the given topic. Write a clear, well-structured paragraph."
    )
    
    # Create the reviewer agent with instructions to use specific phrases
    reviewer = AssistantAgent(
        "reviewer", 
        model_client=client, 
        system_message="""You are an editor who reviews content.
        You must use one of these exact phrases in your response:
        - 'major revision needed' (if content needs significant improvements)
        - 'minor edits suggested' (if content is good with minor changes)
        - 'approved as is' (if content is excellent)
        
        Be specific in your feedback and always include one of these phrases."""
    )
    
    # Create the reviser agent
    reviser = AssistantAgent(
        "reviser", 
        model_client=client, 
        system_message="You are a content reviser. Make substantial improvements to the content based on the reviewer's feedback. After making changes, clearly state what you've improved."
    )
    
    # Create the editor agent
    editor = AssistantAgent(
        "editor", 
        model_client=client, 
        system_message="You are a copy editor. Make minor edits and polish the content based on the reviewer's feedback. After making changes, clearly state what you've edited."
    )
    
    # Create the publisher agent
    publisher = AssistantAgent(
        "publisher", 
        model_client=client, 
        system_message="You are a publisher. Format the approved content for publication and add a publication note. Include 'PUBLICATION COMPLETE' at the end of your response."
    )
    
    # Build the graph with conditional branches
    builder = DiGraphBuilder()
    
    # Add all nodes
    builder.add_node(writer).add_node(reviewer)
    builder.add_node(reviser).add_node(editor).add_node(publisher)
    
    # Set writer as the entry point
    builder.set_entry_point(writer)
    
    # Add edges with string conditions
    builder.add_edge(writer, reviewer)
    
    # Use individual add_edge calls with condition parameter for conditional branching
    builder.add_edge(reviewer, reviser, condition="major revision needed")
    builder.add_edge(reviewer, editor, condition="minor edits suggested")
    builder.add_edge(reviewer, publisher, condition="approved as is")
    
    # Continue the flow after revisions/edits
    builder.add_edge(reviser, reviewer)  # Reviser sends back to reviewer
    builder.add_edge(editor, publisher)  # Editor sends to publisher
    
    # Build and validate the graph
    graph = builder.build()
    
    print("Graph structure:")
    print(f"Nodes: {[node for node in graph.nodes]}")
    print(f"Entry point: {graph.entry_point}")
    print(f"Edges: {[(edge.source, edge.target, edge.condition) for edge in graph.edges]}")
    print("-" * 50)
    
    # Create the flow with a termination condition
    flow = GraphFlow(
        participants=[writer, reviewer, reviser, editor, publisher],
        graph=graph,
        termination_condition=MaxMessageTermination(15)  # Increased limit for debugging
    )
    
    # Run the flow with a scenario
    scenario = "Write a paragraph about artificial intelligence that needs minor edits."
    print(f"Starting flow with scenario: {scenario}")
    print("=" * 50)
    
    try:
        async for event in flow.run_stream(task=scenario):
            print(f"Event: {event}")
    except Exception as e:
        print(f"Error during flow execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
