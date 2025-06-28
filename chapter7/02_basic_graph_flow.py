"""
Chapter 7: Advanced Patterns and Error Handling
Example 2: Basic Graph Flow

Description:
Demonstrates directed graph workflows with conditional routing between agents.
Shows how to create complex multi-agent workflows with approval processes,
revision loops, and conditional branching based on agent responses.

Prerequisites:
- OpenAI API key set in .env file
- AutoGen v0.5+ installed
- Python 3.9+ with asyncio support

Usage:
```bash
python -m chapter7.02_basic_graph_flow
```

Expected Output:
Graph flow demonstration with document review process:
1. Agent1 creates initial document draft
2. Agent2 reviews and passes to Agent3 for decision
3. Agent3 either approves (routes to finalizer) or requests revision (routes back to Agent1)
4. Process continues until approval is achieved
5. Finalizer confirms completion

Key Concepts:
- Directed graph workflow construction
- Conditional routing between agents
- Multi-agent collaboration patterns
- Approval and revision workflows
- Graph-based agent orchestration
- Conditional edge routing
- Workflow state management

AutoGen Version: 0.5+
"""

# Standard library imports
import asyncio
import sys
from pathlib import Path

# Third-party imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_openai_config

async def create_agents():
    """Create all agents for the document review workflow."""
    config = get_openai_config()
    model_client = OpenAIChatCompletionClient(**config)

# Define agents including a terminating agent
agent1 = AssistantAgent(
    name="agent1",
    model_client=model_client,
    system_message="You are agent1. Start the document review process. Create a draft document and pass to agent2. Always identify yourself."
)
agent2 = AssistantAgent(
    name="agent2",
    model_client=model_client,
    system_message="You are agent2. Review the document from agent1 and pass to agent3 for decision. Always identify yourself."
)
agent3 = AssistantAgent(
    name="agent3",
    model_client=model_client,
    system_message="You are agent3. Review the document. If it needs revision, say 'NEEDS_REVISION' and explain why. If it's good, say 'APPROVED' and pass to finalizer. Always identify yourself."
)
# Add a finalizer agent as the leaf node (no outgoing edges)
finalizer = AssistantAgent(
    name="finalizer",
    model_client=model_client,
    system_message="You are the finalizer. Acknowledge that the document review process is complete. Say 'Process completed successfully.'"
)

# Build the graph
builder = DiGraphBuilder()
builder.add_node(agent1).add_node(agent2).add_node(agent3).add_node(finalizer)
builder.set_entry_point(agent1)
builder.add_edge(agent1, agent2)
builder.add_edge(agent2, agent3)

# Use conditional edges - string conditions look for keywords in message content
builder.add_edge(agent3, agent1, condition="NEEDS_REVISION")
builder.add_edge(agent3, finalizer, condition="APPROVED")  # Route to finalizer on approval
# finalizer has no outgoing edges (leaf node) - this satisfies the requirement
graph = builder.build()

# Create a GraphFlow
flow = GraphFlow(
    participants=[agent1, agent2, agent3, finalizer],
    graph=graph,
)

async def main():
    """Main execution function demonstrating graph flow workflow."""
    try:
        print("=== Graph Flow Document Review Process ===")
        
        config = get_openai_config()
        model_client = OpenAIChatCompletionClient(**config)
        
        # Create agents (moved inline for proper cleanup)
        agent1 = AssistantAgent(
            name="agent1",
            model_client=model_client,
            system_message="You are agent1. Start the document review process. Create a draft document and pass to agent2. Always identify yourself."
        )
        agent2 = AssistantAgent(
            name="agent2",
            model_client=model_client,
            system_message="You are agent2. Review the document from agent1 and pass to agent3 for decision. Always identify yourself."
        )
        agent3 = AssistantAgent(
            name="agent3",
            model_client=model_client,
            system_message="You are agent3. Review the document. If it needs revision, say 'NEEDS_REVISION' and explain why. If it's good, say 'APPROVED' and pass to finalizer. Always identify yourself."
        )
        finalizer = AssistantAgent(
            name="finalizer",
            model_client=model_client,
            system_message="You are the finalizer. Acknowledge that the document review process is complete. Say 'Process completed successfully.'"
        )
        
        # Build the workflow graph
        builder = DiGraphBuilder()
        builder.add_node(agent1).add_node(agent2).add_node(agent3).add_node(finalizer)
        builder.set_entry_point(agent1)
        builder.add_edge(agent1, agent2)
        builder.add_edge(agent2, agent3)
        builder.add_edge(agent3, agent1, condition="NEEDS_REVISION")
        builder.add_edge(agent3, finalizer, condition="APPROVED")
        graph = builder.build()
        
        # Create and run the graph flow
        flow = GraphFlow(
            participants=[agent1, agent2, agent3, finalizer],
            graph=graph,
        )
        
        stream = flow.run_stream(
            task="Start a document review process. Agent3 should approve or request revision."
        )
        await Console(stream)
        
        # Cleanup
        await model_client.close()
        print("\n✅ Graph flow demonstration complete!")
        
    except Exception as e:
        print(f"Error in graph flow demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())